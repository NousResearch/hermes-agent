"""A delegated child vetoed by a ``pre_llm_call`` hook reports ``blocked``, not budget exhaustion (#154234).

``_build_result_entry`` reads ``completed``/``failed`` from the child's result; after the finalizer
started keeping vetoed turns out of ``completed``, a block would otherwise fall into the
``max_iterations`` branch and report ``truncated: True`` — a policy stop misread as the child
running out of its budget. The dedicated branch keeps the block message as the deliverable.
"""

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry
from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON


class _Child:
    model = "test-model"
    session_estimated_cost_usd = 0.0
    session_cost_status = None
    session_prompt_tokens = 3
    session_completion_tokens = 0


def _entry(result):
    return _build_result_entry(_Child(), result, 0, 1.0, _SchemaOutcome(None, None, [], 0))


def test_vetoed_child_reports_blocked_not_budget_exhaustion():
    result = {
        "final_response": "Blocked by policy: prompt contains a customer ID.",
        "turn_exit_reason": PRE_LLM_BLOCK_EXIT_REASON,
        "completed": False,
        "failed": False,
        "api_calls": 0,
        "messages": [],
    }
    entry = _entry(result)
    assert entry["status"] == "completed"  # the block message is the deliverable
    assert entry["exit_reason"] == "blocked"
    assert entry["truncated"] is False  # a policy stop is not a truncation


def test_budget_exhaustion_still_reports_max_iterations():
    """The control: the same completed=False/failed=False shape without the veto reason keeps its
    genuine budget-exhaustion reading."""
    result = {"final_response": "Ran out of budget.", "completed": False, "failed": False,
              "api_calls": 12, "messages": []}
    entry = _entry(result)
    assert entry["exit_reason"] == "max_iterations" and entry["truncated"] is True


def test_interrupted_child_still_reports_interrupted():
    result = {"final_response": "Operation interrupted.", "interrupted": True, "messages": []}
    entry = _entry(result)
    assert entry["status"] == "interrupted" and entry["exit_reason"] == "interrupted"
