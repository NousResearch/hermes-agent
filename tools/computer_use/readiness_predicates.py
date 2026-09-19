"""Per-input-action ``verify_state`` predicate builders (RFC #112639).

The cua driver understands two predicate families: ``window`` (exists,
bounds) and ``element`` (selector + exists/enabled/selected/value_equals).
These builders translate a computer_use input action into the postcondition
predicates that verify the step landed, so the guarded-run executor can
confirm a step without a capture + model round trip.

Honest scope: only ``type``/``set_value`` have a driver-expressible
postcondition without caller knowledge (the field's value). Clicks, keys,
drags and scrolls fall back to ``window.exists`` unless the caller names the
element they expect to see afterwards. ``exists`` is always ``True`` — the
driver rejects ``false`` instead of returning an unknown predicate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from tools.computer_use.readiness import ReadinessResult, verify_readiness

# The only postcondition that holds for every input action.
FALLBACK_EXPECT: List[Dict[str, Any]] = [{"window": {"exists": True}}]


def _selector(label: Optional[str], role: Optional[str]) -> Dict[str, str]:
    sel: Dict[str, str] = {}
    if label:
        sel["label_contains"] = label
    if role:
        sel["role"] = role
    if not sel:
        raise ValueError("element predicate needs a label and/or role (driver minLength 1)")
    return sel


def _element_expect(
    *,
    label: Optional[str] = None,
    role: Optional[str] = None,
    enabled: Optional[bool] = None,
    selected: Optional[bool] = None,
    value_equals: Optional[str] = None,
) -> List[Dict[str, Any]]:
    element: Dict[str, Any] = {"selector": _selector(label, role), "exists": True}
    if enabled is not None:
        element["enabled"] = enabled
    if selected is not None:
        element["selected"] = selected
    if value_equals is not None:
        element["value_equals"] = value_equals
    return [{"element": element}]


def predicates_for_type(
    text: str, *, field_label: Optional[str] = None, field_role: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Postcondition for a ``type`` action: the field holds the typed text."""
    if not text:
        raise ValueError("type predicate needs the expected text")
    if field_label is None and field_role is None:
        return list(FALLBACK_EXPECT)
    return _element_expect(label=field_label, role=field_role, value_equals=text)


def predicates_for_set_value(
    value: str, *, field_label: Optional[str] = None, field_role: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Postcondition for ``set_value``: the control holds the set value."""
    if not value:
        raise ValueError("set_value predicate needs the expected value")
    if field_label is None and field_role is None:
        return list(FALLBACK_EXPECT)
    return _element_expect(label=field_label, role=field_role, value_equals=value)


def predicates_for_click(
    *,
    expect_label: Optional[str] = None,
    expect_role: Optional[str] = None,
    expect_enabled: Optional[bool] = None,
    expect_selected: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Postcondition for a click: the element expected afterwards exists."""
    if expect_label is None and expect_role is None:
        return list(FALLBACK_EXPECT)
    return _element_expect(
        label=expect_label, role=expect_role, enabled=expect_enabled, selected=expect_selected
    )


def predicates_for_key(
    *, expect_label: Optional[str] = None, expect_role: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Postcondition for a ``key`` press: the element expected afterwards exists."""
    if expect_label is None and expect_role is None:
        return list(FALLBACK_EXPECT)
    return _element_expect(label=expect_label, role=expect_role)


def _fallback_builder(**kwargs: Any) -> List[Dict[str, Any]]:
    return list(FALLBACK_EXPECT)


# Table-driven: no elif ladder on the action name (repo shape rule).
_ACTION_BUILDERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "type": predicates_for_type,
    "set_value": predicates_for_set_value,
    "click": predicates_for_click,
    "double_click": predicates_for_click,
    "right_click": predicates_for_click,
    "middle_click": predicates_for_click,
    "key": predicates_for_key,
    "drag": _fallback_builder,
    "scroll": _fallback_builder,
    "focus_app": _fallback_builder,
    "wait": _fallback_builder,
    "capture": _fallback_builder,
}


def predicates_for_action(action: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Build the expect list for a computer_use action; unknown actions fall back."""
    builder = _ACTION_BUILDERS.get(action, _fallback_builder)
    return builder(*args, **kwargs)


def verify_action(
    backend: Any,
    *,
    action: str,
    pid: int,
    window_id: int,
    timeout_ms: int = 2000,
    stable_samples: int = 1,
    include_screenshot: bool = False,
    **kwargs: Any,
) -> ReadinessResult:
    """Build the action's predicates and run one bounded readiness check.

    The single integration point for the guarded-run executor: one call turns
    "the model just typed X" into a driver-verified postcondition.
    """
    expect = predicates_for_action(action, **kwargs)
    return verify_readiness(
        backend,
        pid=pid,
        window_id=window_id,
        expect=expect,
        timeout_ms=timeout_ms,
        stable_samples=stable_samples,
        include_screenshot=include_screenshot,
    )
