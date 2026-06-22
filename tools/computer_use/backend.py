"""Abstract backend interface for computer use.

Any implementation (cua-driver over MCP, pyautogui, noop, future Linux/Windows)
must return the shape described below. All methods synchronous; async is
handled inside the backend implementation if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int                       # 1-based SOM index
    role: str                        # AX role (AXButton, AXTextField, ...)
    label: str = ""                  # AXTitle / AXDescription / AXValue snippet
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (logical px)
    app: str = ""                    # owning bundle ID or app name
    pid: int = 0                     # owning process PID
    window_id: int = 0               # SkyLight / CG window ID
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Opaque per-snapshot element handle from cua-driver
    # (trycua/cua#1961 — Surface 6 of NousResearch/hermes-agent#47072).
    # When set, downstream calls can pass it alongside `index` for
    # explicit stale-detection: a stale token returns an error from
    # cua-driver rather than silently re-resolving to a different
    # element. None for pre-#1961 drivers that didn't carry the field.
    element_token: Optional[str] = None

    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bounds
        return x + w // 2, y + h // 2


@dataclass
class CaptureResult:
    """Result of a screen capture call.

    At least one of png_b64 / elements is populated depending on capture mode:
      * mode="vision" → png_b64 only
      * mode="ax"     → elements only
      * mode="som"    → both (default): PNG already has numbered overlays
                         drawn by the backend, and `elements` holds the
                         matching index → element mapping.
    """

    mode: str
    width: int                      # screenshot width (logical px, pre-Anthropic-scale)
    height: int
    png_b64: Optional[str] = None
    elements: List[UIElement] = field(default_factory=list)
    # Optional: the target app/window the elements were captured for.
    app: str = ""
    window_title: str = ""
    # Raw bytes we sent to Anthropic, for token estimation.
    png_bytes_len: int = 0
    # Explicit MIME type for `png_b64` when the backend supplied it
    # (cua-driver-rs emits `mimeType` on every image part as of
    # trycua/cua#1961 — Surface 7 of NousResearch/hermes-agent#47072).
    # When None, downstream consumers fall back to base64-prefix
    # sniffing for back-compat with older drivers.
    image_mime_type: Optional[str] = None


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key / wait)."""

    ok: bool
    action: str
    message: str = ""                # human-readable summary
    # Optional trailing screenshot — set when the caller asked for a
    # post-action capture or the backend always returns one.
    capture: Optional[CaptureResult] = None
    # Arbitrary extra fields for debugging / telemetry.
    meta: Dict[str, Any] = field(default_factory=dict)


# Conservative macOS app-name alias matching shared by the safety wrapper and
# concrete backends. A false positive could route input to the wrong process;
# keep aliases explicit rather than broad/fuzzy.
_APP_ALIASES: Dict[str, Set[str]] = {
    "システム設定": {
        "system settings",
        "system preferences",
        "システム環境設定",
        "com.apple.systemsettings",
        "com.apple.systempreferences",
    },
    "システム環境設定": {
        "system settings",
        "system preferences",
        "システム設定",
        "com.apple.systemsettings",
        "com.apple.systempreferences",
    },
    "system settings": {
        "システム設定",
        "システム環境設定",
        "system preferences",
        "com.apple.systemsettings",
        "com.apple.systempreferences",
    },
    "system preferences": {
        "システム設定",
        "システム環境設定",
        "system settings",
        "com.apple.systemsettings",
        "com.apple.systempreferences",
    },
    "com.apple.systemsettings": {
        "システム設定",
        "システム環境設定",
        "system settings",
        "system preferences",
        "com.apple.systempreferences",
    },
    "com.apple.systempreferences": {
        "システム設定",
        "システム環境設定",
        "system settings",
        "system preferences",
        "com.apple.systemsettings",
    },
}


def _normalize_app_name(value: str) -> str:
    normalized = (value or "").casefold().strip()
    if normalized.endswith(".app"):
        normalized = normalized[:-4]
    return re.sub(r"[\s_\-.]+", " ", normalized).strip()


def _app_name_variants(value: str) -> Set[str]:
    normalized = _normalize_app_name(value)
    if not normalized:
        return set()
    variants = {normalized}
    for alias in _APP_ALIASES.get(normalized, set()):
        variants.add(_normalize_app_name(alias))
    return variants


def app_matches_requested(requested_app: str, actual_app: str) -> bool:
    """Return True when an actual app/window matches a requested app name.

    Prefer exact names and explicit localized aliases. Allow only conservative
    one-way containment for multi-word app names so short/generic requests like
    ``Mail`` or ``Settings`` cannot accidentally match unrelated apps.
    """
    requested = _app_name_variants(requested_app)
    actual = _app_name_variants(actual_app)
    if not requested or not actual:
        return False
    if requested & actual:
        return True
    for req in requested:
        if " " not in req or len(req) < 10:
            continue
        if any(req in act for act in actual):
            return True
    return False


class ComputerUseBackend(ABC):
    """Lifecycle: `start()` before first use, `stop()` at shutdown."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can be used on this host right now.

        Used by check_fn gating and by the post-setup wizard.
        """

    # ── Capture ─────────────────────────────────────────────────────
    @abstractmethod
    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult: ...

    # ── Pointer actions ─────────────────────────────────────────────
    @abstractmethod
    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",           # left | right | middle
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult: ...

    @abstractmethod
    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult: ...

    @abstractmethod
    def scroll(
        self,
        *,
        direction: str,                 # up | down | left | right
        amount: int = 3,                # wheel ticks
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult: ...

    # ── Keyboard ────────────────────────────────────────────────────
    @abstractmethod
    def type_text(self, text: str) -> ActionResult: ...

    @abstractmethod
    def key(self, keys: str) -> ActionResult:
        """Send a key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return'."""

    # ── Introspection ───────────────────────────────────────────────
    @abstractmethod
    def list_apps(self) -> List[Dict[str, Any]]:
        """Return running apps with bundle IDs, PIDs, window counts."""

    @abstractmethod
    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Route input to `app` (by name or bundle ID). Default: focus without raise."""

    # ── Native-value mutation ────────────────────────────────────────
    @abstractmethod
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a native value on an element (e.g. AXPopUpButton selection).

        `element` is the 1-based SOM index returned by a prior capture call.
        """

    # ── Timing ──────────────────────────────────────────────────────
    def wait(self, seconds: float) -> ActionResult:
        """Default implementation: time.sleep."""
        import time
        time.sleep(max(0.0, min(seconds, 30.0)))
        return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")
