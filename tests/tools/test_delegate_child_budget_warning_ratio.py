"""``delegation.budget_warning_ratio`` overrides the ``agent:`` knob for children; with neither
set, a child still gets a checkpoint notice (0.8 default) instead of none at all — mirroring the
kanban-worker 0.9 default in ``turn_iteration_prep.py``. Regression for #124291.
"""
from types import SimpleNamespace

from tools.delegate_tool import _CHILD_BUDGET_WARNING_RATIO_DEFAULT, _apply_child_budget_warning_ratio


def _child(budget_warning_ratio=None):
    return SimpleNamespace(budget_warning_ratio=budget_warning_ratio)


def test_neither_knob_set_gets_the_shipped_default():
    child = _child()
    _apply_child_budget_warning_ratio(child, {})
    assert child.budget_warning_ratio == _CHILD_BUDGET_WARNING_RATIO_DEFAULT


def test_delegation_ratio_overrides_the_inherited_global_ratio():
    child = _child(budget_warning_ratio=0.5)  # as if inherited from agent.budget_warning_ratio
    _apply_child_budget_warning_ratio(child, {"budget_warning_ratio": 0.2})
    assert child.budget_warning_ratio == 0.2


def test_global_ratio_alone_is_kept_when_delegation_knob_is_unset():
    child = _child(budget_warning_ratio=0.6)
    _apply_child_budget_warning_ratio(child, {})
    assert child.budget_warning_ratio == 0.6


def test_invalid_delegation_ratio_falls_back_to_the_default_not_left_unset():
    for bad in (True, "junk", 0, 1, -0.1, 1.5):
        child = _child()
        _apply_child_budget_warning_ratio(child, {"budget_warning_ratio": bad})
        assert child.budget_warning_ratio == _CHILD_BUDGET_WARNING_RATIO_DEFAULT, bad
