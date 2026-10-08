import json

from tools import code_execution_tool
from tools import delegate_tool_child_run as child_run
from tools import process_registry_notifications as notifications


class _Child:
    model = "test-model"
    session_estimated_cost_usd = 0.0
    session_cost_status = "unknown"
    _delegate_role = "leaf"


def test_execute_code_error_result_preserves_only_bounded_outcomes():
    denied = json.loads(code_execution_tool._error_result("blocked", approval_outcome="denied"))
    assert denied["approval_outcome"] == "denied"

    arbitrary = json.loads(code_execution_tool._error_result("blocked", approval_outcome="secret"))
    assert "approval_outcome" not in arbitrary


def test_child_result_and_async_notice_preserve_approval_outcome():
    schema = child_run._SchemaOutcome(schema=None, valid=None, errors=[], retries=0)
    entry = child_run._build_result_entry(
        _Child(),
        {"failed": True, "error": "approval blocked", "approval_outcome": "cancelled"},
        task_index=0,
        duration=1.0,
        schema=schema,
    )
    assert entry["approval_outcome"] == "cancelled"

    text = notifications._format_async_delegation({
        "delegation_id": "delegation-test",
        "completed_at": 0,
        "goal": "run the task",
        "status": "failed",
        "error": entry["error"],
        "approval_outcome": entry["approval_outcome"],
    })
    assert "approval request was withdrawn" in text
    assert "approval_outcome" not in text
