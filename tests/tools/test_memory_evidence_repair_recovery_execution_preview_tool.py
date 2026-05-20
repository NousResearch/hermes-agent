import json

from tools.memory_evidence_repair_recovery_decision_gate_tool import (
    memory_evidence_repair_recovery_decision_gate_tool,
)
from tools.memory_evidence_repair_recovery_execution_preview_tool import (
    memory_evidence_repair_recovery_execution_preview_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
    _executor_preview,
    _token_report,
)
from tests.tools.test_memory_evidence_repair_recovery_decision_gate_tool import (
    _failure_drill,
)


def _preparedness_decision():
    return json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {
                "executor_preview": _executor_preview(),
            }
        )
    )


def _manual_rollback_decision():
    return json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {
                "rollback_drill": _failure_drill(),
            }
        )
    )


def test_tool_accepts_explicit_recovery_decision_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {
                "recovery_decision": _preparedness_decision(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_execution_preview_ready"
    assert result["previews"][0]["route"] == "preparedness_review_only"
    assert "Memory Evidence Repair Recovery Execution Preview" in result["markdown"]


def test_tool_accepts_manual_rollback_decision():
    result = json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {
                "recovery_decision": _manual_rollback_decision(),
            }
        )
    )

    assert result["status"] == "recovery_execution_preview_ready"
    assert result["summary"]["manual_rollback_preview_count"] == 1
    assert result["summary"]["future_mutation_step_count"] == 1
    assert result["previews"][0]["future_would_mutate_memory"] is True


def test_tool_generates_recovery_decision_from_token():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "lock_owner": "han",
            }
        )
    )

    assert result["status"] == "recovery_execution_preview_ready"
    assert result["summary"]["recovery_execution_preview_available"] is True


def test_tool_blocks_when_generated_decision_is_not_usable():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "provide_exact_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_decision_available():
    result = json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_execution_preview_tool(
            {"recovery_decision": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_decision must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_execution_preview_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["preview_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_execution_preview")

    assert entry is not None
    assert entry.toolset == "memory"
