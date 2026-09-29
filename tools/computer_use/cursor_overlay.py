"""Agent-cursor overlay: policy + driver for cua-driver's NATIVE session cursor.

cua-driver (>= 0.29) owns a click-through, always-on-top agent cursor overlay
(``Cua.AgentCursorOverlay.<theme>``) whose lifecycle is tied to a public session
label. Hermes chooses to *use* it rather than re-implement one, because the
driver's overlay is already:

* **click-through + focus-safe** — the Win32 window is created with
  ``WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED``,
  so it never receives a mouse click, never takes keyboard focus, and stays out
  of Alt-Tab. Verified on a live 1920x1080 Windows desktop (see the Win32 probe
  in the accompanying report).
* **per-session and self-cleaning** — the cursor is keyed by the ``session``
  label Hermes already sends (``CuaDriverBackend._session_id``) and is removed
  by ``end_session``, which ``stop()`` already calls.
* **cross-platform** — same binary drives macOS, Windows and Linux; a hand-rolled
  overlay would need per-OS window/DPI/multi-monitor work plus its own process.

What Hermes has to do (and previously did not) is *turn it on and make it
visible*:

* ``set_agent_cursor_enabled`` — show/hide the session cursor. The live 0.29.1
  schema is ``additionalProperties: false`` over ``{session, enabled}``; the
  previous call site also forwarded ``cursor_id``, so the driver rejected the
  whole call and the belt-and-suspenders disable never applied. See
  ``CuaDriverBackend.set_agent_cursor_enabled``.
* ``set_agent_cursor_motion`` — **the actual root cause of "I can't see the
  mouse move"**: the driver's default motion has ``glide_duration_ms = 0.0``
  (measured live via ``get_agent_cursor_state``), i.e. the cursor *teleports*
  to the action target instead of travelling. With no glide there is nothing to
  watch, and a pure accessibility action additionally only plays a brief pulse.
  Configuring a 150-350 ms glide (plus a click dwell and a longer idle-hide)
  turns the driver's own per-action cursor movement into the animation the user
  asked for — for *every* action, element-addressed ones included, with no
  coordinate-space guessing on our side.
* ``move_cursor`` — glide to an action's explicit pixel coordinates *before*
  the action runs, so a coordinate click is seen travelling and then landing.
  Always ``scope="window"``: ``scope="desktop"`` moves the **real** OS pointer,
  which would steal the user's mouse. If the live schema does not advertise
  ``scope`` the glide fails closed rather than risk the real pointer.

Config surface
--------------
``computer_use.cursor_overlay`` (bool, default ``true``) or the process-level
override ``HERMES_CUA_CURSOR_OVERLAY`` (accepts ``1/true/yes/on`` and
``0/false/no/off``; an unparseable value falls through to config). Precedence:
env > config > auto-detect, where auto-detect is "on" unless the ``--no-overlay``
policy (``computer_use.no_overlay``) has suppressed the overlay rendering loop
for this host — macOS, headless/WSL2 and Linux/X11 suppress it for known
CPU/lifecycle regressions (#28152, #47032). On those hosts the existing
``computer_use.no_overlay: false`` remains the escape hatch.

This module is deliberately free of any rendering code, any new process, and any
Electron/renderer dependency.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.computer_use.cua_backend")

#: Process-level override for :func:`cua_cursor_overlay_enabled`.
CURSOR_OVERLAY_ENV_VAR = "HERMES_CUA_CURSOR_OVERLAY"

_TRUTHY = frozenset({"1", "true", "yes", "y", "on", "enable", "enabled"})
_FALSY = frozenset({"0", "false", "no", "n", "off", "disable", "disabled"})

# Motion tuning applied at session start. ``glide_duration_ms`` sits in the
# 150-350 ms window that reads as "smooth" rather than "sliding" or "instant";
# ``dwell_after_click_ms`` is the click pulse the user can follow; ``idle_hide_ms``
# is the driver's maximum (it clamps larger values) and keeps the cursor on screen
# across a slow capture so it does not blink out mid-task.
GLIDE_DURATION_MS = 260.0
DWELL_AFTER_CLICK_MS = 220.0
IDLE_HIDE_MS = 60_000.0
# A shallow arc reads as a deliberate hand movement without overshooting narrow UI.
ARC_FLOW = 0.35
ARC_SIZE = 0.28

#: The only overlay scope Hermes may ever request: ``desktop`` moves the real OS pointer.
OVERLAY_SCOPE = "window"


def _cb():
    """Facade module (config helpers), looked up lazily to avoid an import cycle."""
    from tools.computer_use import cua_backend
    return cua_backend


def _parse_bool(raw: Any) -> Optional[bool]:
    """``True``/``False`` for a recognised on/off token, ``None`` for anything else (incl. real bools' absence)."""
    if isinstance(raw, bool):
        return raw
    if not isinstance(raw, str):
        return None
    token = raw.strip().lower()
    return True if token in _TRUTHY else False if token in _FALSY else None


def cua_cursor_overlay_requested() -> Optional[bool]:
    """The *explicit* opt-in/opt-out value only: env override, then ``computer_use.cursor_overlay``.

    ``None`` when the user never expressed a preference and auto-detection applies.
    """
    explicit = _parse_bool(os.environ.get(CURSOR_OVERLAY_ENV_VAR))
    if explicit is not None:
        return explicit
    return _parse_bool(_cb()._computer_use_cfg().get("cursor_overlay"))


def cua_cursor_overlay_enabled() -> bool:
    """Should the agent cursor overlay be shown for this run?

    ``HERMES_CUA_CURSOR_OVERLAY`` wins, then ``computer_use.cursor_overlay``; both
    default to ``not _cua_no_overlay()`` so an explicit ``cursor_overlay: true``
    still enables the cursor on a host whose ``--no-overlay`` policy would
    otherwise suppress the rendering loop.
    """
    requested = cua_cursor_overlay_requested()
    if requested is not None:
        return requested
    return not _cb()._cua_no_overlay()


class AgentCursorOverlay:
    """Drives one cua-driver session cursor: visibility, motion physics, and per-action glides.

    Every driver call is best-effort — a missing or older driver, a rejected key,
    or a transport hiccup must never fail a computer-use action or a session
    start. Nothing here requests foreground delivery, focuses a window, or
    touches the real pointer.
    """

    def __init__(self, session: Any, session_id: str, *, enabled: Optional[bool] = None) -> None:
        self._session = session
        self._session_id = session_id
        self.enabled = cua_cursor_overlay_enabled() if enabled is None else bool(enabled)
        # Glides are only issued once configure() has actually applied the policy,
        # so a glide can never race ahead of the visibility decision.
        self._configured = False

    # ── capability gates (fail closed) ───────────────────────────────────────
    def _known(self, tool: str) -> bool:
        """Tool advertised by ``tools/list``; before discovery treat it as unknown-but-possible (mirrors capture())."""
        return self._session._has_tool(tool) or not self._session.capabilities_discovered

    def _can_glide(self) -> bool:
        """``move_cursor`` is usable AND advertises ``scope``.

        Fails closed on purpose: without a ``scope`` argument the driver's
        default may be the desktop scope, which moves the *real* OS pointer and
        would fight the user for their own mouse. Better no cursor than that.
        """
        return self._session.supports_input_property("move_cursor", "scope") and self._known("move_cursor")

    # ── policy application ───────────────────────────────────────────────────
    def configure(self) -> None:
        """Apply the overlay policy at session start: show/hide the cursor and set its motion physics."""
        self._configured = True
        self.set_visible(self.enabled)
        if self.enabled:
            self.set_motion()

    def set_visible(self, visible: bool) -> None:
        """Show or hide this session's cursor. Sends only schema-valid keys (``enabled`` + injected ``session``)."""
        self._call("set_agent_cursor_enabled", {"enabled": bool(visible)})

    def set_motion(self) -> None:
        """Install the glide/click-dwell/idle-hide tuning that makes the cursor watchable."""
        if not self._known("set_agent_cursor_motion"):
            return
        self._call("set_agent_cursor_motion", {
            "glide_duration_ms": GLIDE_DURATION_MS,
            "dwell_after_click_ms": DWELL_AFTER_CLICK_MS,
            "idle_hide_ms": IDLE_HIDE_MS,
            "arc_flow": ARC_FLOW,
            "arc_size": ARC_SIZE,
        })

    def glide_to(self, x: Any, y: Any, *, pid: Any = None, window_id: Any = None) -> None:
        """Glide the agent cursor overlay to a point the action is about to touch.

        Coordinates use the same window-local pixel space the input actions use,
        so ``target`` is attached whenever the live schema advertises it; the
        overlay-only ``scope`` is always set. A no-op when the overlay is off,
        unconfigured, or the driver cannot be trusted to keep the real pointer
        still.
        """
        if not (self.enabled and self._configured and self._can_glide()):
            return
        with contextlib.suppress(TypeError, ValueError):
            args: Dict[str, Any] = {"x": float(x), "y": float(y), "scope": OVERLAY_SCOPE}
            if isinstance(pid, int) and isinstance(window_id, int) and self._session.supports_input_property("move_cursor", "target"):
                args["target"] = {"kind": "window", "pid": pid, "window_id": window_id}
            self._call("move_cursor", args)

    def teardown(self) -> None:
        """Stop gliding for this session; the driver removes the cursor itself on ``end_session``."""
        self._configured = False

    # ── transport ────────────────────────────────────────────────────────────
    def _call(self, name: str, args: Dict[str, Any]) -> None:
        """Best-effort driver call; ``session`` is always attached and failures are logged, never raised."""
        try:
            self._session.call_tool(name, {**args, "session": self._session_id})
        except Exception as exc:  # a broken overlay must never break the action it decorates
            logger.debug("cua-driver agent cursor %s failed: %s", name, exc)
