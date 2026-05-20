import json

from tools.memory_evidence_repair_recovery_decision_gate_tool import (
    memory_evidence_repair_recovery_decision_gate_tool,
)
from tools.memory_evidence_repair_rollback_drill_preview_tool import (
    memory_evidence_repair_rollback_drill_preview_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
    _executor_preview,
    _token_report,
)
from tests.tools.test_memory_evidence_repair_rollback_drill_preview_tool import (
    _failed_post_commit_audit,
)


def _preparedness_drill():
    return json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
            {
                "executor_preview": _executor_preview(),
            }
        )
    )


def _failure_drill():
    return json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
            {
                "post_commit_audit": _failed_post_commit_audit(),
            }
        )
    )


def test_tool_accepts_explicit_rollback_drill_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {
                "rollback_drill": _preparedness_drill(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_decision_gate_ready"
    assert result["decisions"][0]["route"] == "preparedness_review_only"
    assert "Memory Evidence Repair Recovery Decision Gate" in result["markdown"]


def test_tool_accepts_failed_drill_and_requires_manual_rollback():
    result = json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {
                "rollback_drill": _failure_drill(),
            }
        )
    )

    assert result["status"] == "recovery_decision_gate_ready"
    assert result["summary"]["manual_rollback_required_count"] == 1
    assert result["decisions"][0]["route"] == "manual_rollback_required"


def test_tool_generates_rollback_drill_from_token():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "lock_owner": "han",
            }
        )
    )

    assert result["status"] == "recovery_decision_gate_ready"
    assert result["summary"]["recovery_decision_available"] is True


def test_tool_blocks_when_generated_drill_is_not_usable():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {
                "approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["decisions"] == []
    assert "provide_exact_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_drill_available():
    result = json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["decisions"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_decision_gate_tool(
            {"rollback_drill": "bad"}
        )
    )

    assert result["success"] is False
    assert "rollback_drill must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_decision_gate_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["decision_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_decision_gate")

    assert entry is not None
    assert entry.toolset == "memory"
