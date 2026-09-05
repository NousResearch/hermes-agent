"""Input side of the cua-driver backend: delivery-mode handling and the pointer / keyboard /
value-setter methods (mixed into ``CuaDriverBackend``)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_parse import _parse_key_combo

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
        if bring_to_front:
            result.meta["foreground_focus"] = {"invoked": True, "tool": "bring_to_front"}
        return result

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
        # coordinates when the driver advertises support — via the capability vocabulary, or (defensive
        # second path, #89527) via the live scroll inputSchema declaring `x`. cua-driver 0.23.x retired
        # the `input.scroll.coordinates` capability token while keeping x/y in the schema; inputSchema is
        # a core MCP tools/list field that no strict client model can drop, so the schema probe recovers
        # coordinate scrolling there. Neither signal → omit x/y and let the driver scroll the targeted
        # window (window_id is still sent for routing). This is the safe default when capabilities
        # haven't been discovered yet (older drivers).
        schema_has_coordinates = (
            self._session.supports_input_property("scroll", "x")
            and self._session.supports_input_property("scroll", "y")
        )
        xy = lambda: ({"x": x, "y": y}  # noqa: E731
                      if self._session.supports_capability("input.scroll.coordinates", tool="scroll")
                      or schema_has_coordinates else {})
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
