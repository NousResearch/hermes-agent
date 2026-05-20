import json

from tools.memory_evidence_repair_recovery_execution_preview_tool import (
    memory_evidence_repair_recovery_execution_preview_tool,
)
from tools.memory_evidence_repair_recovery_human_approval_token_tool import (
    memory_evidence_repair_recovery_human_approval_token_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
    _token_report,
)
from tests.tools.test_memory_evidence_repair_recovery_execution_preview_tool import (
    _manual_rollback_decision,
    _preparedness_decision,
)


def _manual_rollback_execution():
    return json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {
                "recovery_decision": _manual_rollback_decision(),
            }
        )
    )


def _preparedness_execution():
    return json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {
                "recovery_decision": _preparedness_decision(),
            }
        )
    )


def test_tool_accepts_explicit_recovery_execution_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {
                "recovery_execution": _manual_rollback_execution(),
                "approver": "han",
                "approval_reason": "manual recovery approval",
                "expires_in_minutes": 20,
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "draft_ready_for_recovery_human_approval"
    assert result["summary"]["token_count"] == 1
    assert result["tokens"][0]["approver"] == "han"
    assert result["tokens"][0]["expires_in_minutes"] == 20
    assert "Memory Evidence Repair Recovery Human Approval Token" in result["markdown"]


def test_tool_returns_no_action_for_non_mutating_preparedness_execution():
    result = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {
                "recovery_execution": _preparedness_execution(),
            }
        )
    )

    assert result["status"] == "no_action_needed"
    assert result["tokens"] == []


def test_tool_generates_recovery_execution_from_failed_audit_path():
    result = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {
                "recovery_decision": _manual_rollback_decision(),
                "approver": "han",
            }
        )
    )

    assert result["status"] == "draft_ready_for_recovery_human_approval"
    assert result["summary"]["recovery_human_approval_token_available"] is True
    assert result["tokens"][0]["operation_ids"] == ["dryrun-op-123"]


def test_tool_blocks_when_generated_execution_is_not_usable():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {
                "approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["tokens"] == []
    assert "provide_exact_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_execution_available():
    result = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["tokens"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {"recovery_execution": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_execution must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_human_approval_token_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["token_count"] == 0
    assert result["tokens"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_human_approval_token")

    assert entry is not None
    assert entry.toolset == "memory"
