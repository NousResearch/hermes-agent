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
_Variant = tuple[str, None | dict[str, Any] | Callable[[], dict[str, Any]]]

def _refuse(action: str, message: str, **fields: Any) -> ActionResult:
    return ActionResult(ok=False, action=action, message=message, **fields)


class _InputMixin:
    """Pointer / keyboard / value-setter actions against the sticky target."""

    def _target_args(self, action: str, *, need_window: bool = False) -> tuple[Optional[ActionResult], dict[str, Any]]:
        """``(refusal, base args)`` for an input action against the sticky target."""
        if self._active_pid is None or (need_window and self._active_window_id is None):
            return _refuse(action, _NO_TARGET_MSG), {}
        return None, {"pid": self._active_pid, **({"window_id": self._active_window_id} if need_window else {})}

    def _element_addressing(self, action: str, element: Optional[int],
                            element_token: Optional[str], snapshot_id: Optional[str]) -> tuple[Optional[ActionResult], dict[str, Any]]:
        """``(refusal, extra args)`` for element addressing: the genuine per-snapshot token from the latest
        capture rides along so the driver detects staleness instead of silently re-resolving. A caller-supplied
        token/id is validated against the latest capture — never forwarded from an invented or superseded one
        (tokens come from `structuredContent.elements[].element_token`, not from screenshot or cache names).
        Only properties the LIVE schema accepts are emitted; on drivers that mint neither handle the bare
        index goes through unchanged."""
        tokens = getattr(self, "_snapshot_tokens", {}) or {}
        snap = getattr(self, "_snapshot_id", None)
        extra: dict[str, Any] = {"element_index": int(element)} if element is not None else {}
        if element_token is not None:
            token = str(element_token).strip()
            # Only an exact token returned by this backend's latest AX capture is trusted.
            # A caller-provided prefix is not proof, nor is an empty/uninitialised snapshot.
            if token not in tokens.values():
                return _refuse(action, (f"{token!r} is not from the latest capture (current snapshot "
                                        f"{snap!r}). Re-run capture() and use the fresh element_token/element."),
                               code="stale_snapshot"), {}
            if element is not None and (mapped := tokens.get(int(element))) and mapped != token:
                return _refuse(action, f"element={element} does not map to element_token {token!r} in the latest "
                                       f"snapshot ({snap!r}). Re-run capture() and use one consistent handle."), {}
            extra["element_token"] = token
        elif snapshot_id is not None and element is not None:
            sid = str(snapshot_id).strip()
            if not snap or sid != snap:
                return _refuse(action, f"snapshot_id {sid!r} is not the latest driver capture ({snap!r}). Re-run capture() first.",
                               code="stale_snapshot"), {}
            extra["snapshot_id"] = sid
        # Down-select to what the live schema accepts, preferring the richer handle.
        wants_token = self._session.supports_input_property(action, "element_token") or \
            self._session.supports_capability("accessibility.element_tokens", tool=action)
        wants_snap = self._session.supports_input_property(action, "snapshot_id")
        if "element_token" in extra:
            derived_snap = extra["element_token"].split(":", 1)[0] if ":" in extra["element_token"] else snap
            if not wants_token:
                extra.pop("element_token")
                if wants_snap and derived_snap:
                    extra["snapshot_id"] = extra.get("snapshot_id") or derived_snap
        elif "snapshot_id" not in extra and wants_snap:
            if (token := tokens.get(int(element)) if element is not None else None):
                if wants_token:
                    extra["element_token"] = token
                elif (derived := token.split(":", 1)[0] if ":" in token else snap):
                    extra["snapshot_id"] = derived
            elif snap and element is not None:
                extra["snapshot_id"] = snap
        return None, extra

    def _validate_xy(self, action: str, x: Optional[int], y: Optional[int]) -> Optional[ActionResult]:
        """Reject coordinates outside the last capture's screenshot space BEFORE sending input. `x`/`y` are
        window-local screenshot pixels (the driver's documented contract); values in the tens of thousands on a
        ~1456px frame are native-desktop coordinates typed into the wrong space, which the driver would
        silently re-scale to an off-window screen point."""
        size = getattr(self, "_last_capture_size", None)
        if x is None or y is None or not size:
            return None
        img_w, img_h = size
        if x < 0 or y < 0 or x > img_w or y > img_h:
            frame = getattr(self, "_active_frame", None)
            convert = (f" Convert bounds→coordinate: x=(bounds_x - {frame[0]}) * {img_w}/{frame[2]},"
                       f" y=(bounds_y - {frame[1]}) * {img_h}/{frame[3]}." if frame and frame[2] > 0 and frame[3] > 0
                       else " Re-capture with app= or pid=/window_id= to get the frame for conversion.")
            return _refuse(action, f"coordinate ({x},{y}) is outside the last capture's screenshot "
                                   f"({img_w}x{img_h}). Coordinates are window-local screenshot pixels, NOT "
                                   "screen-absolute element bounds — prefer element=/element_token=." + convert,
                           code="coordinate_out_of_bounds")
        return None

    def _pointer_args(self, tool: str, args: dict[str, Any], variants: Sequence[_Variant],
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

    def _apply_delivery(self, action: str, args: dict[str, Any], delivery_mode: Optional[str]) -> Optional[ActionResult]:
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

    def _run_input_action(self, action: str, args: dict[str, Any], delivery_mode: Optional[str],
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
              button: str = "left", click_count: int = 1, modifiers: Optional[list[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False,
              element_token: Optional[str] = None, snapshot_id: Optional[str] = None) -> ActionResult:
        refusal, args = self._target_args("click")
        if refusal is not None:
            return refusal
        if (bad := self._validate_xy("click", x, y)) is not None:
            return bad
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
        if element is not None or element_token is not None or snapshot_id is not None:
            refusal, addressing = self._element_addressing(tool, element, element_token, snapshot_id)
            if refusal is not None:
                return refusal
            if not addressing:
                return _refuse(tool, "element addressing needs element= (index) or element_token=.")
            variants: Sequence[_Variant] = ((f"element #{element if element is not None else 'token'} click", addressing),)
        else:
            variants = (("coordinate click", {"x": x, "y": y} if x is not None and y is not None else None),)
        refusal = self._pointer_args(tool, args, variants, "click requires element= or x/y.")
        if modifiers:
            args["modifier"] = modifiers
        return refusal if refusal is not None else self._run_input_action(tool, args, delivery_mode, bring_to_front)

    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_xy: Optional[tuple[int, int]] = None, to_xy: Optional[tuple[int, int]] = None,
             button: str = "left", modifiers: Optional[list[str]] = None,
             delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("drag")
        if refusal is None:
            for pt in (from_xy, to_xy):
                if pt is not None and (bad := self._validate_xy("drag", pt[0], pt[1])) is not None:
                    return bad
            refusal = self._pointer_args("drag", args, (
                ("element-based drag", {"from_element": from_element, "to_element": to_element}
                 if from_element is not None and to_element is not None else None),
                ("coordinate drag", {"from_x": int(from_xy[0]), "from_y": int(from_xy[1]),
                                     "to_x": int(to_xy[0]), "to_y": int(to_xy[1])}
                 if from_xy is not None and to_xy is not None else None),
            ), "drag requires from_element/to_element or from_coordinate/to_coordinate.")
        return refusal if refusal is not None else self._run_input_action("drag", args, delivery_mode, bring_to_front)

    def scroll(self, *, direction: str, amount: int = 3, element: Optional[int] = None,
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[list[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False,
               element_token: Optional[str] = None, snapshot_id: Optional[str] = None) -> ActionResult:
        refusal, args = self._target_args("scroll")
        if refusal is not None:
            return refusal
        if (bad := self._validate_xy("scroll", x, y)) is not None:
            return bad
        args.update(direction=direction, amount=max(1, min(50, amount)))
        # An element without a known window_id is not an addressing form here; scrolling then falls through
        # to the coordinate form or the bare window. Some driver schemas reject x/y on scroll: only send
        # coordinates when the driver advertises support; otherwise it scrolls the targeted window
        # (window_id is still sent for routing).
        xy = lambda: ({"x": x, "y": y}
                      if self._session.supports_capability("input.scroll.coordinates", tool="scroll")
                      or self._session.supports_input_property("scroll", "x") else {})
        element_variant = None
        if element is not None or element_token is not None:
            if self._active_window_id is not None:
                refusal, addressing = self._element_addressing("scroll", element, element_token, snapshot_id)
                if refusal is not None:
                    return refusal
                element_variant = addressing
        refusal = self._pointer_args("scroll", args, (
            ("element scroll", element_variant),
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

    def set_value(self, value: str, element: Optional[int] = None, *, element_token: Optional[str] = None,
                  snapshot_id: Optional[str] = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        refusal, args = self._target_args("set_value", need_window=True)
        if refusal is not None:
            return refusal
        if element is None and element_token is None:
            return _refuse("set_value", "set_value requires element= (element index).")
        refusal, addressing = self._element_addressing("set_value", element, element_token, snapshot_id)
        if refusal is not None:
            return refusal
        if not addressing:
            return _refuse("set_value", "set_value requires element= (element index).")
        return self._action("set_value", {**args, **addressing, "value": value})
