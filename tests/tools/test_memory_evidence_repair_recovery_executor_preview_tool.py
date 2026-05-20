import json

from tools.memory_evidence_repair_recovery_executor_preview_tool import (
    memory_evidence_repair_recovery_executor_preview_tool,
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


def _recovery_write_lock_gate():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]
    return json.loads(
        memory_evidence_repair_recovery_write_lock_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "lock_ttl_minutes": 15,
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )


def test_tool_accepts_explicit_recovery_write_lock_gate_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {
                "recovery_write_lock_gate": _recovery_write_lock_gate(),
                "current_time": "2026-05-11T00:12:00+00:00",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_executor_preview_ready_for_manual_recovery"
    assert result["summary"]["manual_recovery_executor_preview_ready"] is True
    assert result["summary"]["future_mutation_step_count"] == 1
    assert result["previews"][0]["steps"][0]["action"] == (
        "verify_recovery_write_lock_token_and_execution"
    )
    assert "Memory Evidence Repair Recovery Executor Preview" in result["markdown"]


def test_tool_blocks_expired_recovery_write_lock_gate():
    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {
                "recovery_write_lock_gate": _recovery_write_lock_gate(),
                "current_time": "2026-05-11T00:26:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "regenerate_recovery_write_lock_gate" in result["required_actions"]


def test_tool_generates_recovery_write_lock_gate_from_token_and_creates_preview():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "recovery_executor_preview_ready_for_manual_recovery"
    assert result["summary"]["recovery_executor_preview_available"] is True
    assert result["previews"][0]["operation_ids"] == ["dryrun-op-123"]


def test_tool_blocks_when_generated_recovery_write_lock_gate_is_not_ready():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_lock_available():
    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert result["blocking_reasons"]


def test_tool_returns_no_action_for_non_mutating_recovery_path():
    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {
                "recovery_execution": _preparedness_execution(),
            }
        )
    )

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["previews"] == []


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {"recovery_write_lock_gate": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_write_lock_gate must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_executor_preview_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["preview_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_executor_preview")

    assert entry is not None
    assert entry.toolset == "memory"
