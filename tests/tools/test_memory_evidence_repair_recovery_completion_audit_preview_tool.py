import json

from tools.memory_evidence_repair_recovery_completion_audit_preview_tool import (
    memory_evidence_repair_recovery_completion_audit_preview_tool,
)
from tools.memory_evidence_repair_recovery_completion_receipt_tool import (
    memory_evidence_repair_recovery_completion_receipt_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
)
from tests.tools.test_memory_evidence_repair_recovery_approval_token_gate_tool import (
    _recovery_token_report,
)
from tests.tools.test_memory_evidence_repair_recovery_completion_receipt_tool import (
    _recovery_executor_preview,
)


def _recovery_completion_receipt():
    return json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_executor_preview": _recovery_executor_preview(),
                "actor": "han",
                "recovery_reason": "manual post-audit recovery complete",
            }
        )
    )


def _observed_recovery_steps(receipt):
    return {
        step_id: {"status": "completed"}
        for step_id in receipt["receipts"][0]["recovery_step_ids"]
    }


def test_tool_accepts_explicit_recovery_completion_receipt_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
            {
                "recovery_completion_receipt": _recovery_completion_receipt(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_completion_audit_preview_ready"
    assert result["summary"]["planned_audit_step_count"] == 12
    assert "Memory Evidence Repair Recovery Completion Audit Preview" in result["markdown"]


def test_tool_accepts_observed_recovery_completion_state_and_passes():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]

    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
            {
                "recovery_completion_receipt": receipt,
                "observed_recovery_steps": _observed_recovery_steps(receipt),
                "recorded_receipts": [{"id": row["id"]}],
                "used_token_ids": [row["token_id"]],
                "released_lock_ids": [row["lock_id"]],
                "post_recovery_audit_status": {"status": "pass"},
                "contamination_status": {"contaminated": False},
            }
        )
    )

    assert result["status"] == "recovery_completion_audit_preview_ready"
    assert result["summary"]["observed_pass_step_count"] == 12
    assert result["summary"]["observed_fail_step_count"] == 0


def test_tool_blocks_bad_observed_contamination_status():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]

    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
            {
                "recovery_completion_receipt": receipt,
                "observed_recovery_steps": _observed_recovery_steps(receipt),
                "recorded_receipts": [{"id": row["id"]}],
                "used_token_ids": [row["token_id"]],
                "released_lock_ids": [row["lock_id"]],
                "post_recovery_audit_status": {"status": "pass"},
                "contamination_status": {"contaminated": True},
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "investigate_recovery_completion_audit_failures" in result["required_actions"]


def test_tool_generates_recovery_completion_receipt_from_token():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
            }
        )
    )

    assert result["status"] == "recovery_completion_audit_preview_ready"
    assert result["summary"]["recovery_completion_audit_preview_available"] is True


def test_tool_blocks_when_generated_recovery_completion_receipt_is_not_ready():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
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


def test_tool_uses_latest_manager_state_and_blocks_when_no_receipt_available():
    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
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
        memory_evidence_repair_recovery_completion_audit_preview_tool(
            {"recovery_completion_receipt": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_completion_receipt must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool({})
    )

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["preview_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_completion_audit_preview")

    assert entry is not None
    assert entry.toolset == "memory"
