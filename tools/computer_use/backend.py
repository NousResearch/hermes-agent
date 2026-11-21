"""Abstract backend interface for computer use. Any implementation (cua-driver over MCP, a plugin
driver, the test noop) returns the shapes below. All methods are synchronous; async is handled inside
the backend implementation if needed. :class:`ComputerUseProvider` is the factory a provider plugin
registers (see :mod:`plugins.computer_use`); ``computer_use.backend`` selects exactly one."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.provider_base import ProviderBase


# Base64 prefixes of the image magics we can resolve from bytes alone. GIF,
# WebP and BMP join PNG/JPEG because a capture source that mislabels bytes
# (e.g. a CLI fallback that hardcodes `image/png`) would otherwise produce a
# data URL whose declared media type contradicts the payload — Anthropic
# rejects that mismatch with a non-retryable HTTP 400, and because the image
# is already in history it is re-sent every turn (NousResearch/hermes-agent#157848).
_IMAGE_MAGIC_MIMES: tuple[tuple[str, str], ...] = (
    ("iVBOR", "image/png"),
    ("/9j/", "image/jpeg"),
    ("R0lGOD", "image/gif"),
    ("UklGR", "image/webp"),
    ("Qk", "image/bmp"),
)


def resolve_image_mime(b64: str, declared: Optional[str]) -> str:
    """Resolve the MIME type for a base64 capture, trusting bytes over labels.

    Sniffs the base64 payload for a known image magic first; a declared MIME
    that disagrees with recognizable bytes is a mislabel (the payload is what
    the provider validates), so the sniffed type wins. When the bytes carry
    no recognizable magic, the declared type is preserved — back-compat for
    drivers that emit formats outside the table (or pre-mimeType builds).
    """
    prefix = (b64 or "")[:8]
    for magic, mime in _IMAGE_MAGIC_MIMES:
        if prefix.startswith(magic):
            return mime
    return (declared or "").strip().lower() or "image/png"


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int                       # 1-based SOM index
    role: str                        # AX role (AXButton, AXTextField, ...)
    label: str = ""                  # AXTitle / AXDescription / AXValue snippet
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (logical px)
    app: str = ""                    # owning bundle ID or app name
    pid: int = 0                     # owning process PID
    window_id: int = 0               # SkyLight / CG window ID
    attributes: dict[str, Any] = field(default_factory=dict)
    # Opaque per-snapshot handle from cua-driver, passed alongside `index` for explicit stale-detection: a
    # stale token errors instead of silently re-resolving to a different element. None on older drivers.
    # None for pre-#1961 drivers that didn't carry the field.
    element_token: Optional[str] = None


@dataclass
class CaptureResult:
    """Result of a screen capture call. mode="vision" → png_b64 only; mode="ax" → elements
    only; mode="som" (default) → both: the PNG already carries numbered overlays drawn by
    the backend and `elements` holds the matching index → element mapping."""

    mode: str
    width: int                      # screenshot width (logical px, pre-Anthropic-scale)
    height: int
    png_b64: Optional[str] = None
    elements: list[UIElement] = field(default_factory=list)
    app: str = ""                   # target app/window the elements were captured for
    window_title: str = ""
    png_bytes_len: int = 0          # raw bytes sent to Anthropic, for token estimation
    # Explicit MIME type for `png_b64` when the backend supplied it (cua-driver-rs emits `mimeType` on every image
    # part as of trycua/cua#1961 — Surface 7 of NousResearch/hermes-agent#47072). Downstream consumers resolve the
    # final type through `resolve_image_mime`, which trusts sniffable bytes over this label so a mislabeled capture
    # (e.g. JPEG bytes tagged `image/png`) never ships as a contradicting data URL (#157848); a declared type the
    # bytes cannot confirm is still honored for formats outside the sniff table.
    image_mime_type: Optional[str] = None
    # Guidance appended to the summary by capture lanes that intentionally return no elements (e.g.
    # full-screen composited grabs) to point the model at an interactive lane.
    note: str = ""
    # ``max_elements`` the backend asked the driver's AX walk to stop at (0 = unbounded / not applicable);
    # ``len(elements) >= ax_max_elements > 0`` means the tree may be truncated.
    ax_max_elements: int = 0


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key / wait). ``ok`` is
    tool/transport success only — NOT the semantic verdict; read ``effect`` / ``escalation``
    (cua-driver's structured verdict) to pick the next rung of the verify → escalate ladder.
    Structured fields are optional and additive: an older driver that omits
    ``structuredContent`` leaves them ``None``, behavior unchanged.

    Beyond the transport-level ``ok`` flag, this carries cua-driver's structured action verdict so the model
    can follow the documented verify → escalate ladder (NousResearch/hermes-agent#67052).
    """

    ok: bool
    action: str
    message: str = ""                # human-readable summary
    capture: Optional[CaptureResult] = None  # trailing screenshot, when requested / always-on
    meta: dict[str, Any] = field(default_factory=dict)  # debugging / telemetry extras
    verified: Optional[bool] = None  # AX read-back: True confirmed, False unconfirmed, None n/a
    effect: Optional[str] = None     # "confirmed" | "unverifiable" | "suspected_noop"
    # {"recommended": "px"|"foreground"|"page", "reason": str} — only when driver recommends climbing
    escalation: Optional[dict[str, Any]] = None
    path: Optional[str] = None       # delivery rung that ran (e.g. "ax", "x11_pixel", "cgevent_fg")
    degraded: Optional[bool] = None  # AX walk found no actionable elements (act by px instead)
    delivery_mode: Optional[str] = None  # the delivery_mode the caller requested, echoed back
    code: Optional[str] = None       # refusal code, e.g. "background_unavailable", "desktop_scope_disabled"


class ComputerUseBackend(ABC):
    """Lifecycle: `start()` before first use, `stop()` at shutdown. Pointer/keyboard actions
    take ``delivery_mode`` (background (default) | foreground) and ``bring_to_front``;
    ``button`` is left | right | middle; ``modifiers`` a list of key names. ``element`` args
    are 1-based SOM indices from a prior capture. `direction` is up | down | left | right and
    `amount` is wheel ticks; `keys` is a combo such as 'cmd+s', 'ctrl+alt+t', 'return'."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool: ...  # usable on this host right now (check_fn gating, setup wizard)

    @abstractmethod
    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult: ...

    @abstractmethod
    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", click_count: int = 1, modifiers: Optional[list[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_xy: Optional[tuple[int, int]] = None, to_xy: Optional[tuple[int, int]] = None,
             button: str = "left", modifiers: Optional[list[str]] = None,
             delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def scroll(self, *, direction: str, amount: int = 3, element: Optional[int] = None,
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[list[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def type_text(self, text: str, *, delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def key(self, keys: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def list_apps(self) -> list[dict[str, Any]]: ...  # running apps with bundle IDs, PIDs, window counts

    def list_windows(self) -> list[dict[str, Any]]:
        """Visible native windows with PID and window identifiers. Optional compatibility hook: backends that
        predate window discovery stay instantiable and report none."""
        return []

    @abstractmethod
    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult: ...  # route input to `app` (name / bundle ID)

    @abstractmethod
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult: ...  # e.g. AXPopUpButton selection

    def wait(self, seconds: float) -> ActionResult:  # default implementation
        time.sleep(max(0.0, min(seconds, 30.0)))
        return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")


class ComputerUseProvider(ProviderBase):
    """Factory for one computer_use driver, registered via ``ctx.register_computer_use_provider`` from a provider
    plugin whose directory name is the ``computer_use.backend`` value (built-in: ``plugins/computer_use/cua``). Only
    the selected provider is imported and instantiated. The model-facing tool schema is identical for every backend:
    an action the driver cannot perform returns ``ActionResult(ok=False, code="unsupported_action", message=...)``
    rather than changing the schema."""

    @abstractmethod
    def create_backend(self, *, permission_mode: str) -> ComputerUseBackend:
        """A fresh, unstarted backend for one Hermes session (the tool calls ``start()``/``stop()``).
        ``permission_mode`` is ``standard`` | ``bounded`` | ``unrestricted`` (approval bypass active)."""

    def is_available(self) -> bool:
        """Usable on this host right now (the tool's ``check_fn``). Cheap: no network, no spawning."""
        return True

    def doctor(self) -> Optional[int]:
        """``hermes computer-use doctor`` for this backend: print findings, return the exit code.
        ``None`` (default) means the backend ships no doctor."""
        return None
