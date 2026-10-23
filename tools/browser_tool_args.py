"""Argument checks for the browser tools, shared by both browser lanes.

The extension controller clamps or ignores arguments it does not understand (an unknown step field is
dropped, an oversized batch is refused only after the call reaches it), and the built-in backend only
reads the arguments its table declares. Checking here, before a call reaches either lane, turns a bad
argument into one clear error the model can act on instead of a silent no-op, and an approval card is
never raised for a call that could not have run.

``invalid_arguments(tool, args, extension_lane=...)`` is the entry point; tools without a check pass through
untouched. ``extension_lane`` says whether the session was shown the extension form, which takes ``tab_id``: a
refusal never points a session at an argument its own form does not have.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

#: The extension's executor refuses longer batches (``batch_too_large``).
BATCH_MAX_STEPS = 20
#: Step actions the extension's executor accepts inside a batch. Tab management, screenshots and snapshots
#: stay out: a snapshot in the middle would re-mint every ref the later steps rely on.
BATCH_STEP_ACTIONS = (
    "back", "click", "drag", "fill", "hover", "navigate", "press", "scroll", "scroll_to", "select", "type",
)
SCROLL_DIRECTIONS = ("up", "down", "left", "right")
SCROLL_AMOUNT_RANGE = (100, 5000)
#: Canonical modifier names; the extension drops any other name silently.
MODIFIERS = ("Alt", "Control", "Meta", "Shift")

# Per step action: ({argument: expected type}, required arguments). Names match the standalone tools.
_STEP_ARGS: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "back": ({}, ()),
    "click": ({"ref": "str", "x": "number", "y": "number"}, ()),
    "drag": ({"ref": "str", "to_ref": "str"}, ("ref", "to_ref")),
    "fill": ({"ref": "str", "value": "str"}, ("ref", "value")),
    "hover": ({"ref": "str"}, ("ref",)),
    "navigate": ({"url": "str"}, ("url",)),
    "press": ({"key": "str", "modifiers": "list"}, ("key",)),
    "scroll": ({"direction": "str", "amount": "int"}, ("direction",)),
    "scroll_to": ({"ref": "str"}, ("ref",)),
    "select": ({"ref": "str", "value": "str"}, ("ref", "value")),
    "type": ({"ref": "str", "text": "str", "clear": "bool"}, ("ref", "text")),
}
# Tools outside a batch that take arguments worth checking (tab ids must be real ids; a tab opens on a real URL).
_TAB_TOOL_ARGS: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "tab_activate": ({"tab_id": "tab"}, ("tab_id",)),
    "tab_close": ({"tab_id": "tab"}, ("tab_id",)),
    "tab_create": ({"url": "str"}, ("url",)),
    "tab_group": ({"tab_ids": "tabs"}, ("tab_ids",)),
    "tab_ungroup": ({"tab_ids": "tabs"}, ("tab_ids",)),
    "screenshot": ({"ref": "str"}, ()),
}
#: Arguments that may be an empty string (typing "" into a field clears it); every other required string may not.
_MAY_BE_EMPTY = frozenset({"text", "value"})
#: Arguments the batch call itself takes besides ``actions``; ``tab_id`` picks the tab for the whole batch.
_BATCH_ARGS = frozenset({"actions", "stop_on_error", "tab_id"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return _is_number(value) and float(value).is_integer()


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "number": _is_number,
    "int": _is_int,
    "tab": lambda v: _is_int(v) and v >= 1,
    "tabs": lambda v: isinstance(v, list) and bool(v) and all(_is_int(i) and i >= 1 for i in v),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
}
_TYPE_NAMES = {"str": "a string", "number": "a number", "int": "a whole number", "bool": "true or false",
               "list": "a list", "tab": "a tab id (a whole number from 1)",
               "tabs": "a non-empty list of tab ids (whole numbers from 1)"}


def _present(args: dict[str, Any], key: str) -> bool:
    return args.get(key) is not None


def click_error(args: dict[str, Any]) -> Optional[str]:
    """A click needs an element ref, or both viewport coordinates (never one of them alone)."""
    has_x, has_y = _present(args, "x"), _present(args, "y")
    if args.get("ref") or (has_x and has_y):
        return None
    if has_x or has_y:
        return "a coordinate click needs both 'x' and 'y' (CSS pixels from the top-left of the viewport)"
    return "a click needs 'ref' (an element ref from the page snapshot) or both 'x' and 'y'"


def scroll_error(args: dict[str, Any]) -> Optional[str]:
    direction = args.get("direction")
    if direction not in SCROLL_DIRECTIONS:
        return f"'direction' must be one of {', '.join(SCROLL_DIRECTIONS)}, got {direction!r}"
    amount = args.get("amount")
    low, high = SCROLL_AMOUNT_RANGE
    if amount is not None and not (_is_int(amount) and low <= amount <= high):
        return f"'amount' must be a whole number of pixels from {low} to {high}, got {amount!r}"
    return None


def press_error(args: dict[str, Any]) -> Optional[str]:
    modifiers = args.get("modifiers")
    if modifiers is None:
        return None
    if not isinstance(modifiers, list):
        return f"'modifiers' must be a list drawn from {', '.join(MODIFIERS)}"
    known = {name.lower() for name in MODIFIERS}
    for modifier in modifiers:
        if not isinstance(modifier, str) or modifier.lower() not in known:
            return f"unknown modifier {modifier!r}; use {', '.join(MODIFIERS)}"
    return None


def tab_ids_error(args: dict[str, Any]) -> Optional[str]:
    """The tab group tools name their tabs in ``tab_ids`` only, each tab once; they take no ``tab_id``."""
    unknown = sorted(set(args) - {"tab_ids"})
    if unknown:
        return f"does not take {', '.join(unknown)}; it takes tab_ids"
    tab_ids = args.get("tab_ids")
    if isinstance(tab_ids, list) and len(set(tab_ids)) != len(tab_ids):
        return "'tab_ids' must list each tab once"
    return None


_STEP_CHECKS: dict[str, Callable[[dict[str, Any]], Optional[str]]] = {
    "click": click_error, "scroll": scroll_error, "press": press_error,
    "tab_group": tab_ids_error, "tab_ungroup": tab_ids_error,
}


def step_action(step: Any) -> str:
    """The step's action without the optional ``browser_`` prefix (``""`` when it names none)."""
    name = step.get("action") if isinstance(step, dict) else None
    if not isinstance(name, str):
        return ""
    return name.removeprefix("browser_")


def _fields_error(action: str, fields: dict[str, Any], allowed: dict[str, str], required: tuple[str, ...]) -> Optional[str]:
    """Type, presence and range problems in ``fields`` for ``action``, or ``None``. Unknown fields are not judged here."""
    for key, kind in allowed.items():
        if _present(fields, key) and not _TYPE_CHECKS[kind](fields[key]):
            return f"'{key}' must be {_TYPE_NAMES[kind]}, got {fields[key]!r}"
    for key in required:
        if not _present(fields, key) or (fields[key] == "" and key not in _MAY_BE_EMPTY):
            return f"needs '{key}'"
    check = _STEP_CHECKS.get(action)
    return check(fields) if check else None


def step_error(index: int, step: Any, extension_lane: bool = True) -> Optional[str]:
    """Why batch step ``index`` (zero-based) cannot run, or ``None``."""
    label = f"step {index + 1}"
    if not isinstance(step, dict):
        return f"{label} must be an object with an 'action'"
    action = step_action(step)
    if action not in _STEP_ARGS:
        return (f"{label} has action {step.get('action')!r}, which cannot run inside a batch; "
                f"use one of {', '.join(BATCH_STEP_ACTIONS)}, or call the tool on its own")
    allowed, required = _STEP_ARGS[action]
    label = f"{label} ({action})"
    unknown = sorted(set(step) - set(allowed) - {"action"})
    if unknown:
        hint = " (tab_id goes on the batch itself, not on a step)" if "tab_id" in unknown and extension_lane else ""
        return f"{label} does not take {', '.join(unknown)}{hint}; it takes {', '.join(allowed) or 'no arguments'}"
    detail = _fields_error(action, step, allowed, required)
    return f"{label}: {detail}" if detail else None


def batch_error(args: dict[str, Any], extension_lane: bool = True) -> Optional[str]:
    actions = args.get("actions")
    if not isinstance(actions, list) or not actions:
        return "'actions' must be a non-empty list of steps, each an object with an 'action'"
    if len(actions) > BATCH_MAX_STEPS:
        return f"a batch holds at most {BATCH_MAX_STEPS} steps, got {len(actions)}; split it into several calls"
    accepted = _BATCH_ARGS if extension_lane else _BATCH_ARGS - {"tab_id"}
    unknown = sorted(set(args) - accepted)
    if unknown:
        return f"does not take {', '.join(unknown)}; it takes {', '.join(sorted(accepted))}"
    stop = args.get("stop_on_error")
    if stop is not None and not isinstance(stop, bool):
        return f"'stop_on_error' must be true or false, got {stop!r}"
    for index, step in enumerate(actions):
        error = step_error(index, step, extension_lane)
        if error:
            return error
    return None


def _tool_check(action: str) -> Callable[[dict[str, Any]], Optional[str]]:
    allowed, required = _STEP_ARGS.get(action) or _TAB_TOOL_ARGS[action]
    return lambda args: _fields_error(action, args, allowed, required)


#: Tool name -> check for the arguments that tool declares. Tools without an entry (navigate, back, snapshot,
#: tabs and the built-in-only tools) have nothing to check.
_TOOL_CHECKS: dict[str, Callable[[dict[str, Any]], Optional[str]]] = {
    "browser_batch": batch_error,
    **{f"browser_{action}": _tool_check(action)
       for action in (*_STEP_ARGS, *_TAB_TOOL_ARGS) if action not in ("navigate", "back")},
}


def invalid_arguments(tool: str, args: dict[str, Any], *, extension_lane: bool = True) -> Optional[str]:
    """Error text for a call to ``tool`` that cannot run as given, or ``None`` when it may proceed."""
    check = _TOOL_CHECKS.get(tool)
    tab_id = args.get("tab_id")  # any routed tool may name the tab it acts on
    if tab_id is not None and not _TYPE_CHECKS["tab"](tab_id):
        detail = f"'tab_id' must be {_TYPE_NAMES['tab']}, got {tab_id!r}"
    elif tool == "browser_batch":
        detail = batch_error(args, extension_lane)
    else:
        detail = check(args) if check else None
    return f"{tool}: {detail}" if detail else None
