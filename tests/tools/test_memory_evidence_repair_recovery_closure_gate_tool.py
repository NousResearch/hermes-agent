import json

from tools.memory_evidence_repair_recovery_closure_gate_tool import (
    memory_evidence_repair_recovery_closure_gate_tool,
)
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
from tests.tools.test_memory_evidence_repair_recovery_completion_audit_preview_tool import (
    _observed_recovery_steps,
    _recovery_completion_receipt,
)


def _passing_completion_audit():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]
    return json.loads(
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


def test_tool_accepts_explicit_passing_completion_audit_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {
                "recovery_completion_audit": _passing_completion_audit(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_closure_ready"
    assert result["summary"]["recovery_closure_ready"] is True
    assert result["closures"][0]["closure_decision"] == "close_recovery_loop"
    assert "Memory Evidence Repair Recovery Closure Gate" in result["markdown"]


def test_tool_blocks_planned_completion_audit_without_observed_state():
    receipt = _recovery_completion_receipt()
    planned = json.loads(
        memory_evidence_repair_recovery_completion_audit_preview_tool(
            {"recovery_completion_receipt": receipt}
        )
    )

    result = json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {"recovery_completion_audit": planned}
        )
    )

    assert result["status"] == "blocked"
    assert result["closures"] == []
    assert "provide_observed_recovery_completion_state" in result["required_actions"]


def test_tool_generates_completion_audit_from_token_and_creates_closure():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]
    receipt = json.loads(
        memory_evidence_repair_recovery_completion_receipt_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "recovery_reason": "manual post-audit recovery complete",
            }
        )
    )
    row = receipt["receipts"][0]

    result = json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": token["required_confirmation_text"],
                "lock_owner": "han",
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "recovery_reason": "manual post-audit recovery complete",
                "observed_recovery_steps": _observed_recovery_steps(receipt),
                "recorded_receipts": [{"id": row["id"]}],
                "used_token_ids": [row["token_id"]],
                "released_lock_ids": [row["lock_id"]],
                "post_recovery_audit_status": {"status": "pass"},
                "contamination_status": {"contaminated": False},
            }
        )
    )

    assert result["status"] == "recovery_closure_ready"
    assert result["summary"]["recovery_closure_available"] is True
    assert result["closures"][0]["operation_ids"] == ["dryrun-op-123"]


def test_tool_blocks_when_generated_completion_audit_is_not_ready():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["closures"] == []
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_audit_available():
    result = json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["closures"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {"recovery_completion_audit": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_completion_audit must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_closure_gate_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["closure_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_closure_gate")

    assert entry is not None
    assert entry.toolset == "memory"
