import json

from tools.memory_evidence_repair_recovery_approval_token_gate_tool import (
    memory_evidence_repair_recovery_approval_token_gate_tool,
)
from tools.memory_evidence_repair_recovery_human_approval_token_tool import (
    memory_evidence_repair_recovery_human_approval_token_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
)
from tests.tools.test_memory_evidence_repair_recovery_human_approval_token_tool import (
    _manual_rollback_execution,
    _preparedness_execution,
)


def _recovery_token_report():
    payload = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {
                "recovery_execution": _manual_rollback_execution(),
                "approver": "han",
                "approval_reason": "manual recovery approval",
                "expires_in_minutes": 20,
            }
        )
    )
    payload["generated_at"] = "2026-05-11T00:00:00+00:00"
    return payload


def test_tool_accepts_explicit_recovery_token_markdown():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "verified_for_manual_recovery"
    assert result["summary"]["manual_recovery_verified"] is True
    assert "Memory Evidence Repair Recovery Approval Token Gate" in result["markdown"]


def test_tool_blocks_wrong_confirmation():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_blocks_used_token():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "used_token_ids": [token["id"]],
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert "regenerate_recovery_human_approval_token" in result["required_actions"]


def test_tool_returns_no_action_for_non_mutating_recovery_token_report():
    token_report = json.loads(
        memory_evidence_repair_recovery_human_approval_token_tool(
            {"recovery_execution": _preparedness_execution()}
        )
    )

    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {"recovery_approval_token": token_report}
        )
    )

    assert result["status"] == "no_action_needed"
    assert result["checks"] == []


def test_tool_blocks_when_generated_token_is_not_confirmed():
    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {
                "recovery_execution": _manual_rollback_execution(),
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_token_available():
    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {"recovery_approval_token": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_approval_token must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_approval_token_gate_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["check_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_approval_token_gate")

    assert entry is not None
    assert entry.toolset == "memory"
