import json

from tools.memory_evidence_repair_recovery_completion_receipt_tool import (
    memory_evidence_repair_recovery_completion_receipt_tool,
)
from tools.memory_evidence_repair_recovery_executor_preview_tool import (
    memory_evidence_repair_recovery_executor_preview_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
)
from tests.tools.test_memory_evidence_repair_recovery_approval_token_gate_tool import (
    _recovery_token_report,
)
from tests.tools.test_memory_evidence_repair_recovery_executor_preview_tool import (
    _recovery_write_lock_gate,
)


def _recovery_executor_preview():
    return json.loads(
        memory_evidence_repair_recovery_executor_preview_tool(
            {
                "recovery_write_lock_gate": _recovery_write_lock_gate(),
                "current_time": "2026-05-11T00:12:00+00:00",
            }
        )
    )


def test_tool_accepts_explicit_recovery_executor_preview_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_executor_preview": _recovery_executor_preview(),
                "actor": "han",
                "recovery_reason": "manual post-audit recovery complete",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_completion_receipt_ready"
    assert result["summary"]["recovery_completion_receipt_available"] is True
    assert result["receipts"][0]["actor"] == "han"
    assert result["receipts"][0]["operation_ids"] == ["dryrun-op-123"]
    assert "Memory Evidence Repair Recovery Completion Receipt" in result["markdown"]


def test_tool_blocks_when_recovery_token_already_used():
    executor_preview = _recovery_executor_preview()
    token_id = executor_preview["previews"][0]["token_id"]

    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_executor_preview": executor_preview,
                "used_token_ids": [token_id],
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["receipts"] == []
    assert "regenerate_recovery_human_approval_token" in result["required_actions"]


def test_tool_blocks_when_recovery_lock_already_released():
    executor_preview = _recovery_executor_preview()
    lock_id = executor_preview["previews"][0]["lock_id"]

    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_executor_preview": executor_preview,
                "released_lock_ids": [lock_id],
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["receipts"] == []
    assert "regenerate_recovery_write_lock_gate" in result["required_actions"]


def test_tool_generates_recovery_executor_preview_from_token_and_creates_receipt():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
            }
        )
    )

    assert result["status"] == "recovery_completion_receipt_ready"
    assert result["summary"]["would_record_receipt_count"] == 1
    assert result["receipts"][0]["token_id"] == token["id"]


def test_tool_blocks_when_generated_recovery_executor_preview_is_not_ready():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["receipts"] == []
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_executor_available():
    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["receipts"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {"recovery_executor_preview": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_executor_preview must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_completion_receipt_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["receipt_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_completion_receipt")

    assert entry is not None
    assert entry.toolset == "memory"
