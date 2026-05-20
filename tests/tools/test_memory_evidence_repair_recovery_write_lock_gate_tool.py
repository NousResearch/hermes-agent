import json

from tools.memory_evidence_repair_recovery_approval_token_gate_tool import (
    memory_evidence_repair_recovery_approval_token_gate_tool,
)
from tools.memory_evidence_repair_recovery_write_lock_gate_tool import (
    memory_evidence_repair_recovery_write_lock_gate_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
)
from tests.tools.test_memory_evidence_repair_recovery_approval_token_gate_tool import (
    _recovery_token_report,
)
from tests.tools.test_memory_evidence_repair_recovery_human_approval_token_tool import (
    _preparedness_execution,
)


def _verified_recovery_token_gate():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]
    return json.loads(
        memory_evidence_repair_recovery_approval_token_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )


def test_tool_accepts_explicit_recovery_token_gate_markdown():
    token_gate = _verified_recovery_token_gate()

    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {
                "recovery_token_gate": token_gate,
                "lock_owner": "han",
                "lock_ttl_minutes": 15,
                "current_time": "2026-05-11T00:10:00+00:00",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_write_lock_ready_for_manual_recovery"
    assert result["summary"]["recovery_executor_preview_allowed"] is True
    assert result["locks"][0]["owner"] == "han"
    assert result["locks"][0]["expires_at"] == "2026-05-11T00:25:00+00:00"
    assert "Memory Evidence Repair Recovery Write Lock Gate" in result["markdown"]


def test_tool_blocks_on_active_recovery_lock_conflict():
    token_gate = _verified_recovery_token_gate()

    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {
                "recovery_token_gate": token_gate,
                "active_locks": [
                    {
                        "id": "recovery-lock-active",
                        "status": "active",
                        "scope": "manual_memory_evidence_repair_recovery",
                        "operation_ids": ["dryrun-op-123"],
                        "expires_at": "2026-05-11T00:20:00+00:00",
                    }
                ],
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["summary"]["active_lock_conflict_count"] == 1
    assert result["active_lock_conflicts"][0]["reason"] == "overlapping_operation_ids"


def test_tool_generates_token_gate_from_recovery_approval_token():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["success"] is True
    assert result["status"] == "recovery_write_lock_ready_for_manual_recovery"
    assert result["summary"]["recovery_write_lock_available"] is True
    assert result["locks"][0]["token_id"] == token["id"]


def test_tool_blocks_when_generated_token_gate_is_not_verified():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_token_gate_available():
    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["blocking_reasons"]


def test_tool_returns_no_action_for_non_mutating_recovery_path():
    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {
                "recovery_execution": _preparedness_execution(),
            }
        )
    )

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["locks"] == []


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {"recovery_token_gate": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_token_gate must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_write_lock_gate_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["lock_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_write_lock_gate")

    assert entry is not None
    assert entry.toolset == "memory"
