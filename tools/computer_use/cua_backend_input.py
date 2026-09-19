"""Input side of the cua-driver backend: delivery-mode handling and the pointer / keyboard /
value-setter methods (mixed into ``CuaDriverBackend``)."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_parse import _parse_key_combo

logger = logging.getLogger(__name__)

_NO_TARGET_MSG = "No active window — call capture() first."
_BTF_UNSUPPORTED_MSG = "The connected cua-driver does not advertise the standalone bring_to_front tool."
_FOREGROUND_UNSUPPORTED_MSG = ("The connected cua-driver action schema does not accept delivery_mode, so foreground "
                               "delivery is unavailable. Use another verified rung without assuming the reported "
                               "package version describes the live schema.")
# (what, extra args) pointer addressing form; ``extra`` is None when the caller did not supply that form and
# may be a callable when computing it has side effects (capability probes) that must follow the refusal checks.
_Variant = Tuple[str, Union[None, Dict[str, Any], Callable[[], Dict[str, Any]]]]

def _refuse(action: str, message: str, **fields: Any) -> ActionResult:
    return ActionResult(ok=False, action=action, message=message, **fields)


def _flush_modifier_after(action: str, args: Dict[str, Any]) -> bool:
    """Return True when an injected action carried modifier state and should
    be followed by a no-op key flush.

    cua-driver injects chords (hotkey) and modifier-bearing clicks/drags/
    scrolls via CGEvent on macOS. A lost key-up leaves the OS believing the
    modifier is still held, which turns the user's own typing into hotkey
    behaviour until the stuck state is re-synced. Only actions that actually
    carried modifiers need the flush; plain clicks/keys/type_text are already
    self-contained key-up events.

    The check is intentionally conservative and silent-failing: a flush is a
    no-op if the driver rejects the extra key, and it must never change the
    outcome of the action it follows.
    """
    if sys.platform != "darwin":
        return False
    if action in {"hotkey", "press_key"}:
        keys = args.get("keys")
        if keys:
            return True  # hotkey is by construction a modifier chord
        # press_key with modifiers is ALSO a chord and can lose its modifier
        # key-up — flush regardless of whether a main key accompanies the
        # modifiers (review #93702-3: previously this only flushed when
        # `key is None`, leaving press_key(key='x', modifiers=['cmd']) with
        # the same stuck-state risk as hotkey).
        return bool(args.get("modifiers") or args.get("modifier"))
    return bool(args.get("modifiers")) or bool(args.get("modifier"))


@functools.lru_cache(maxsize=1)
def _resolve_stuck_key_helper() -> Optional[str]:
    """Resolve the trusted release-stuck-keys helper path, or None.

    The helper is the compiled product of the macos-input-troubleshooting
    skill's `scripts/release-stuck-keys.swift` (installed per its SKILL.md
    via `swiftc -O ... -o ~/bin/release-stuck-keys`). We only trust the
    ~/bin copy when the skill's source file is still present on disk —
    otherwise an arbitrary executable dropped into ~/bin would be silently
    executed by the backend on every modifier-bearing action (review
    #93702-1: supply-chain surface). Resolved once and cached: per-action
    stat/spawn during drags and scroll loops is wasted work.
    """
    helper = os.path.expanduser("~/bin/release-stuck-keys")
    if not (os.path.exists(helper) and os.access(helper, os.X_OK)):
        return None
    # Provenance gate: the helper may only come from the skill we ship.
    skill_src = os.path.expanduser(
        "~/.hermes/skills/devops/macos-input-troubleshooting/scripts/release-stuck-keys.swift"
    )
    if not os.path.exists(skill_src):
        return None
    return helper


class _InputMixin:
    """Pointer / keyboard / value-setter actions against the sticky target."""

    def _target_args(self, action: str, *, need_window: bool = False) -> Tuple[Optional[ActionResult], Dict[str, Any]]:
        """``(refusal, base args)`` for an input action against the sticky target."""
        if self._active_pid is None or (need_window and self._active_window_id is None):
            return _refuse(action, _NO_TARGET_MSG), {}
        return None, {"pid": self._active_pid, **({"window_id": self._active_window_id} if need_window else {})}

    def _pointer_args(self, tool: str, args: Dict[str, Any], variants: Sequence[_Variant],
                      missing_msg: Optional[str]) -> Optional[ActionResult]:
        """Fill *args* from the first supplied addressing variant (element or coordinates) plus ``window_id``; refuse
        when the target has a pid but no window_id yet. No variant -> refuse with *missing_msg* (None = bare window)."""
        for what, extra in variants:
            if extra is not None:
                if self._active_window_id is None:
                    return _refuse(tool, f"No active window_id for {what}.")
                args.update(extra() if callable(extra) else extra, window_id=self._active_window_id)
                return None
        return _refuse(tool, missing_msg) if missing_msg else None

    def _apply_delivery(self, action: str, args: Dict[str, Any], delivery_mode: Optional[str]) -> Optional[ActionResult]:
        """Attach delivery_mode to an input-action args dict. Background is the default and needs no flag.
        Foreground is only sent when the live action schema accepts it; on an older driver we refuse with
        ``foreground_unsupported`` instead of silently downgrading to background (which would land input
        where the model didn't expect).

        Returns an ActionResult to short-circuit on refusal, or None to proceed. See
        NousResearch/hermes-agent#67052 phase B.
        """
        if not delivery_mode or delivery_mode == "background":
            return None
        if delivery_mode != "foreground":
            return _refuse(action, f"unknown delivery_mode {delivery_mode!r} — use background|foreground.",
                           code="bad_delivery_mode")
        if not self._session.supports_input_property(action, "delivery_mode"):
            return _refuse(action, _FOREGROUND_UNSUPPORTED_MSG, code="foreground_unsupported", delivery_mode="foreground")
        args["delivery_mode"] = "foreground"
        return None

    def _run_input_action(self, action: str, args: Dict[str, Any], delivery_mode: Optional[str],
                          bring_to_front: bool) -> ActionResult:
        """Apply one delivery rung, optionally focusing via its own tool. ``bring_to_front`` is never an
        input-action property: when requested, the separately approved standalone focus action runs first,
        then the original foreground input runs unchanged."""
        refusal = self._apply_delivery(action, args, delivery_mode)
        if refusal is not None:
            return refusal
        if bring_to_front:
            if delivery_mode != "foreground":
                return _refuse(action, "bring_to_front requires delivery_mode='foreground'.",
                               code="bring_to_front_requires_foreground")
            if not self._session._has_tool("bring_to_front"):
                return _refuse(action, _BTF_UNSUPPORTED_MSG, code="bring_to_front_unsupported", delivery_mode="foreground")
            if self._active_pid is None or self._active_window_id is None:
                return _refuse(action, "Capture an exact target before requesting persistent foreground focus.",
                               code="bring_to_front_target_required", delivery_mode="foreground")
            focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
            if not focused.ok:
                return focused
        result = self._action(action, args)
        # Post-injection hygiene: cua-driver injects modifier key chords via
        # CGEvent, and a lost key-up leaves macOS believing ctrl/cmd/fn is
        # still held — the user's typing then behaves as hotkeys and input
        # feels "stuck" (see macos-input-troubleshooting skill). Flush runs
        # even when the injection reported failure: a partially-injected
        # chord (lost key-up) is exactly the stuck-state scenario (review
        # #93702 minor).
        if _flush_modifier_after(action, args):
            self._flush_stuck_modifiers()
        if bring_to_front:
            result.meta["foreground_focus"] = {"invoked": True, "tool": "bring_to_front"}
        return result

    def _flush_stuck_modifiers(self) -> None:
        """Re-sync macOS modifier state after a modifier-bearing injection.

        cua-driver's CGEvent injection can lose a modifier key-up under
        race (window switch, IME toggle, event contention). macOS then
        believes ctrl/cmd/fn is still held and the user's own keyboard
        input acts as hotkeys until the state is cleared.

        This prefers the standalone `release-stuck-keys --fix` helper
        (ships with the macos-input-troubleshooting skill): it re-posts
        every modifier key-up via CGEvent, which re-syncs the OS state
        machine without producing any visible key. The helper path is
        provenance-gated (only trusted when the skill's source file is
        present) and resolved once via lru_cache — per-action stat/spawn
        during drags and scroll loops is wasted work (review #93702-1).

        When the helper is not installed (e.g. fresh machines / CI), fall
        back to pressing F12 with no modifiers through the live session —
        F12 has no default binding on macOS, produces no text, and nudges
        the state machine to drop stale presses. Known trade-off (review
        #93702-2): a synthetic F12 is delivered to the focused window, and
        apps that bind F12 (DevTools, IDE actions) could react; the
        injection is targeted at the active session's pid/window_id, macOS
        has no system-level F12 default, and this path only runs when the
        CGEvent helper is unavailable. An in-process CGEvent re-post would
        be strictly safer but requires Quartz bindings the backend does not
        ship. Failures are silent: the flush must never change the outcome
        of the action it follows.
        """
        if sys.platform != "darwin":
            return
        # Preferred: local helper re-posts all modifier key-ups.
        try:
            helper = _resolve_stuck_key_helper()
            if helper:
                subprocess.run(
                    [helper, "--fix"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                return
        except Exception:
            pass
        # Fallback: no-op F12 through the live session (self-contained).
        try:
            if self._active_pid is None:
                return
            args: Dict[str, Any] = {"pid": self._active_pid, "key": "f12"}
            if self._active_window_id is not None:
                args["window_id"] = self._active_window_id
            self._action("press_key", args)
        except Exception:
            pass

    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", click_count: int = 1, modifiers: Optional[List[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("click")
        if refusal is not None:
            return refusal
        # Tool is chosen by click_count only; `button` goes through click's enum (the driver rejects unknown
        # buttons). `right_click` / `middle_click` MCP tools are deprecated aliases and never invoked here.
        # Choose tool by click_count only — single-vs-double — and pass the button through to `click`'s
        # `button` enum (Surface 5 of NousResearch/hermes-agent#47072). cua-driver-rs gained an explicit
        # `button: "left"|"right"|"middle"` arg on `click` in trycua/cua#1961 which rejects unknown buttons;
        # before that, `middle` was silently mapped to a left-click via name-routing through `right_click`.
        button_norm = (button or "left").lower()
        if button_norm not in {"left", "right", "middle"}:
            return _refuse("click", f"unknown button {button!r} — expected left, right, middle.")
        tool, args["button"] = ("double_click" if click_count == 2 else "click"), button_norm
        refusal = self._pointer_args(tool, args, (
            ("element_index click", {"element_index": element} if element is not None else None),
            ("coordinate click", {"x": x, "y": y} if x is not None and y is not None else None),
        ), "click requires element= or x/y.")
        if modifiers:
            args["modifier"] = modifiers
        return refusal if refusal is not None else self._run_input_action(tool, args, delivery_mode, bring_to_front)

    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_xy: Optional[Tuple[int, int]] = None, to_xy: Optional[Tuple[int, int]] = None,
             button: str = "left", modifiers: Optional[List[str]] = None,
             delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("drag")
        if refusal is None:
            refusal = self._pointer_args("drag", args, (
                ("element-based drag", {"from_element": from_element, "to_element": to_element}
                 if from_element is not None and to_element is not None else None),
                ("coordinate drag", {"from_x": int(from_xy[0]), "from_y": int(from_xy[1]),
                                     "to_x": int(to_xy[0]), "to_y": int(to_xy[1])}
                 if from_xy is not None and to_xy is not None else None),
            ), "drag requires from_element/to_element or from_coordinate/to_coordinate.")
        return refusal if refusal is not None else self._run_input_action("drag", args, delivery_mode, bring_to_front)

    def scroll(self, *, direction: str, amount: int = 3, element: Optional[int] = None,
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[List[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("scroll")
        if refusal is not None:
            return refusal
        args.update(direction=direction, amount=max(1, min(50, amount)))
        # An element without a known window_id is not an addressing form here; scrolling then falls through
        # to the coordinate form or the bare window. Some driver schemas reject x/y on scroll: only send
        # coordinates when the driver advertises support; otherwise it scrolls the targeted window
        # (window_id is still sent for routing).
        xy = lambda: ({"x": x, "y": y}  # noqa: E731
                      if self._session.supports_capability("input.scroll.coordinates", tool="scroll") else {})
        refusal = self._pointer_args("scroll", args, (
            ("element scroll", {"element_index": element}
             if element is not None and self._active_window_id is not None else None),
            ("coordinate scroll", xy if x is not None and y is not None else None),
        ), None)
        return refusal if refusal is not None else self._run_input_action("scroll", args, delivery_mode, bring_to_front)

    def type_text(self, text: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("type_text", need_window=True)
        return refusal if refusal is not None else self._run_input_action("type_text", {**args, "text": text},
                                                                          delivery_mode, bring_to_front)

    def key(self, keys: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("key", need_window=True)
        if refusal is not None:
            return refusal
        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return _refuse("key", f"Could not parse key from '{keys}'.")
        if modifiers:  # hotkey requires at least one modifier + one key
            return self._run_input_action("hotkey", {**args, "keys": modifiers + [key_name]}, delivery_mode, bring_to_front)
        return self._run_input_action("press_key", {**args, "key": key_name}, delivery_mode, bring_to_front)

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        refusal, args = self._target_args("set_value", need_window=True)
        if refusal is not None:
            return refusal
        if element is None:
            return _refuse("set_value", "set_value requires element= (element index).")
        return self._action("set_value", {**args, "element_index": element, "value": value})
