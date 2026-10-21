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
             from_xy: Optional[tuple[int, int]] = None, to_xy: Optional[tuple[int, int]] = None,
             button: str = "left", modifiers: Optional[list[str]] = None,
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
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[list[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("scroll")
        if refusal is not None:
            return refusal
        args.update(direction=direction, amount=max(1, min(50, amount)))
        # An element without a known window_id is not an addressing form here; scrolling then falls through
        # to the coordinate form or the bare window. Some driver schemas reject x/y on scroll: only send
        # coordinates when the driver advertises support; otherwise it scrolls the targeted window
        # (window_id is still sent for routing).
        xy = lambda: ({"x": x, "y": y}
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

    # ── Batch (run_steps) ───────────────────────────────────────────────────
    # Hermes exposes the driver's native run_steps tool so an ordered sequence of actions runs in ONE
    # driver call. Separate calls are not a batch: they have no shared up-front validation or single
    # end-of-batch observation, and parallel tool calls do not promise an input order. A wrong result
    # alone does not identify which layer omitted or changed a step. The driver validates every step
    # before running the first one, so a malformed step changes nothing.
    _RUN_STEP_TOOLS = ("click", "double_click", "right_click", "set_value", "type_text", "press_key",
                       "hotkey", "scroll", "drag", "move_cursor")
    # Hermes-level action names the model naturally reaches for, mapped onto the driver's tool names.
    # `key` and `type` are Hermes spellings; the driver splits press_key (one key) from hotkey (modifiers)
    # and calls typed text `type_text`. `middle_click` is a `click` with button=middle.
    _RUN_STEP_ALIASES = {"key": "press_key", "type": "type_text", "middle_click": "click"}

    def _run_step_args(self, step: dict[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
        """Translate one Hermes step into a driver step: ``(tool, args)``.

        A step is ``{"tool": t, "args": {...}}`` or the short form ``{"<tool>": {...}}``. Hermes-level
        argument names are mapped onto the driver's own schema: ``element`` (index) becomes
        ``element_token`` on a strict token-only driver, ``keys``/``key`` become ``press_key``/``hotkey``,
        and coordinates become ``x``/``y``.
        """
        if not isinstance(step, dict):
            return None, {}
        accepted = self._RUN_STEP_TOOLS + tuple(self._RUN_STEP_ALIASES)
        if "tool" in step:
            tool = step.get("tool")
            raw = step.get("args") or {}
        else:
            named = [k for k in step if k in accepted]
            if len(named) != 1:
                return None, {}
            tool = named[0]
            raw = step.get(tool) or {}
        if not isinstance(tool, str) or not isinstance(raw, dict):
            return None, {}
        requested_tool = tool
        tool = self._RUN_STEP_ALIASES.get(tool, tool)
        if tool not in self._RUN_STEP_TOOLS:
            return None, {}
        args = dict(raw)
        if requested_tool == "middle_click":
            args.setdefault("button", "middle")
        # Hermes `element` is a 1-based SOM index; the driver wants the token from the latest capture.
        idx = args.pop("element", None)
        if isinstance(idx, int):
            token = self._snapshot_tokens.get(idx)
            if token and self._session.supports_input_property(tool, "element_token"):
                args["element_token"] = token
            elif not self._session.supports_input_property(tool, "element_token"):
                args["element_index"] = idx
            else:
                return None, {}
        # Hermes spells the combo `keys="cmd+s"`; the driver splits press_key (one key) from hotkey (modifiers).
        combo = args.pop("keys", None)
        if isinstance(combo, str):
            key_name, modifiers = _parse_key_combo(combo)
            if not key_name:
                return None, {}
            if modifiers:
                tool = "hotkey"
                args["keys"] = modifiers + [key_name]
            else:
                tool = "press_key"
                args["key"] = key_name
        elif isinstance(combo, list):
            tool = "hotkey"
            args["keys"] = combo
        elif isinstance(args.get("key"), str) and isinstance(args.get("modifiers"), list) and args["modifiers"]:
            tool = "hotkey"
            args["keys"] = [*args.pop("modifiers"), args.pop("key")]
        # Hermes `text` is the driver's `type_text` argument; `type` is not a driver tool name.
        if tool == "type_text" and "text" not in args and "value" in args:
            args["text"] = args.pop("value")
        if tool == "type_text" and "text" not in args and isinstance(args.get("keys"), str):
            args["text"] = args.pop("keys")
        # Hermes `middle_click` is a `click` carrying button=middle.
        if tool == "click" and "button" not in args and raw.get("button") == "middle":
            args["button"] = "middle"
        # Hermes coordinate pair -> driver x/y.
        coord = args.pop("coordinate", None)
        if isinstance(coord, (list, tuple)) and len(coord) == 2 and "x" not in args:
            args["x"], args["y"] = coord[0], coord[1]
        # A step must not carry the batch's own session/pid/window; the driver inherits them.
        for k in ("session", "pid", "window_id"):
            args.pop(k, None)
        return tool, args

    def run_steps(self, steps: list[Any], *, observe: Optional[dict[str, Any]] = None,
                  delay_ms: Optional[int] = None, delivery_mode: Optional[str] = None,
                  foreground_fallback: bool = False) -> ActionResult:
        """Run an ordered batch of actions in one driver call (the driver's native ``run_steps`` tool).

        Older drivers without the tool refuse with ``run_steps_unsupported`` rather than silently
        splitting a batch into individual calls. Capture or focus an exact target first so the driver
        has a default window for steps that use role/name addressing.
        """
        refusal, _ = self._target_args("run_steps", need_window=True)
        if refusal is not None:
            return _refuse("run_steps", "run_steps needs a current window target — call capture() or focus_app() first.",
                           code="run_steps_target_required")
        if not self._session._has_tool("run_steps"):
            return _refuse("run_steps",
                           "This cua-driver does not provide the run_steps batch tool — send the steps one "
                           "action per call and verify the resulting app state after the sequence.",
                           code="run_steps_unsupported")
        if not steps:
            return _refuse("run_steps", "run_steps requires a non-empty `steps` list.")
        if len(steps) > 64:
            return _refuse("run_steps", f"run_steps accepts at most 64 steps (got {len(steps)}).")
        translated: list[dict[str, Any]] = []
        for i, step in enumerate(steps):
            tool, args = self._run_step_args(step)
            if tool is None:
                return _refuse("run_steps",
                               f"Step {i + 1} is not a valid step: use {{'tool': <tool>, 'args': {{...}}}} with tool in "
                               f"{list(self._RUN_STEP_TOOLS)}.")
            entry: dict[str, Any] = {"tool": tool, "args": args}
            # wait_for / expect / timeout_ms are driver-side step modifiers, not tool arguments.
            for k in ("wait_for", "expect", "timeout_ms"):
                if isinstance(step, dict) and k in step:
                    entry[k] = step[k]
            translated.append(entry)
        payload: dict[str, Any] = {"steps": translated}
        # The batch inherits the sticky target: a step that names no window needs a default window, and
        # the driver rejects the whole batch ("observe: `pid` and `window_id` are required (no step
        # names a window to inherit from)") when none can be derived. The sticky target from the last
        # capture/focus_app is that default.
        if self._active_pid is not None:
            payload["pid"] = self._active_pid
        if self._active_window_id is not None:
            payload["window_id"] = self._active_window_id
        if observe is not None:
            payload["observe"] = observe
        if delay_ms is not None:
            payload["delay_ms"] = max(0, min(2000, int(delay_ms)))
        if delivery_mode == "foreground":
            payload["delivery_mode"] = "foreground"
        if foreground_fallback:
            payload["foreground_fallback"] = True
        return self._action("run_steps", payload)
