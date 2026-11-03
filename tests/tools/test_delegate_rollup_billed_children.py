"""A provider-billed subagent must not report $0 to its parent.

The child's spend lands in ``session_actual_cost_usd`` when the provider
reported a cost. Both the per-delegation footer and the parent rollup read
the estimated column, so they have to read the billed one too.
"""

from types import SimpleNamespace

from tools.delegate_tool_results import _rollup_children_cost


def _parent(**kw):
    base = dict(
        session_estimated_cost_usd=0.0,
        session_actual_cost_usd=0.0,
        session_cost_status="unknown",
        session_cost_source="none",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_rollup_lands_in_the_estimated_column_for_an_estimating_parent():
    parent = _parent(session_estimated_cost_usd=0.1)
    _rollup_children_cost(parent, 0.75)
    assert parent.session_estimated_cost_usd == 0.85
    assert parent.session_actual_cost_usd == 0.0


def test_rollup_lands_in_the_billed_column_for_a_billed_parent():
    """A parent already carrying provider-billed spend keeps the rollup there."""
    parent = _parent(session_actual_cost_usd=0.0187, session_cost_status="actual")
    _rollup_children_cost(parent, 0.0187)
    assert parent.session_actual_cost_usd == 0.0187 * 2
    assert parent.session_estimated_cost_usd == 0.0, "no double count across columns"


def test_rollup_ignores_a_non_numeric_parent_cost():
    parent = _parent()
    parent.session_actual_cost_usd = "not-a-number"
    _rollup_children_cost(parent, 0.5)
    # Degrades to the estimated column instead of raising inside the rollup.
    assert parent.session_estimated_cost_usd == 0.5


def test_rollup_is_a_noop_for_zero_and_negative_totals():
    for total in (0.0, -1.0):
        parent = _parent()
        _rollup_children_cost(parent, total)
        assert parent.session_estimated_cost_usd == 0.0
        assert parent.session_actual_cost_usd == 0.0
