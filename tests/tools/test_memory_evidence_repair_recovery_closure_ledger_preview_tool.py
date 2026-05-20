import json

from tools.memory_evidence_repair_recovery_closure_gate_tool import (
    memory_evidence_repair_recovery_closure_gate_tool,
)
from tools.memory_evidence_repair_recovery_closure_ledger_preview_tool import (
    memory_evidence_repair_recovery_closure_ledger_preview_tool,
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
from tests.tools.test_memory_evidence_repair_recovery_closure_gate_tool import (
    _passing_completion_audit,
)
from tests.tools.test_memory_evidence_repair_recovery_completion_audit_preview_tool import (
    _observed_recovery_steps,
)


def _ready_closure_gate():
    return json.loads(
        memory_evidence_repair_recovery_closure_gate_tool(
            {"recovery_completion_audit": _passing_completion_audit()}
        )
    )


def test_tool_accepts_explicit_recovery_closure_gate_markdown():
    result = json.loads(
        memory_evidence_repair_recovery_closure_ledger_preview_tool(
            {
                "recovery_closure_gate": _ready_closure_gate(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "recovery_closure_ledger_preview_ready"
    assert result["summary"]["recovery_closure_ledger_ready"] is True
    assert result["entries"][0]["entry_type"] == (
        "memory_evidence_repair_recovery_closure_ledger_entry"
    )
    assert "Memory Evidence Repair Recovery Closure Ledger Preview" in result["markdown"]


def test_tool_accepts_explicit_recovery_closure_draft():
    closure = _ready_closure_gate()["closures"][0]

    result = json.loads(
        memory_evidence_repair_recovery_closure_ledger_preview_tool(
            {"recovery_closure": closure}
        )
    )

    assert result["status"] == "recovery_closure_ledger_preview_ready"
    assert result["entries"][0]["closure_id"] == closure["id"]
    assert result["entries"][0]["operation_ids"] == ["dryrun-op-123"]


def test_tool_generates_closure_chain_from_token_and_creates_ledger_entry():
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
        memory_evidence_repair_recovery_closure_ledger_preview_tool(
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

    assert result["status"] == "recovery_closure_ledger_preview_ready"
    assert result["summary"]["entry_count"] == 1
    assert result["entries"][0]["operation_ids"] == ["dryrun-op-123"]
    assert result["entries"][0]["would_record_ledger_entry"] is True


def test_tool_blocks_when_generated_closure_gate_is_not_ready():
    token_report = _recovery_token_report()

    result = json.loads(
        memory_evidence_repair_recovery_closure_ledger_preview_tool(
            {
                "recovery_approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["entries"] == []
    assert "provide_exact_recovery_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_closure_available():
    result = json.loads(
        memory_evidence_repair_recovery_closure_ledger_preview_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["entries"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_recovery_closure_ledger_preview_tool(
            {"recovery_closure_gate": "bad"}
        )
    )

    assert result["success"] is False
    assert "recovery_closure_gate must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_recovery_closure_ledger_preview_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["entry_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_recovery_closure_ledger_preview")

    assert entry is not None
    assert entry.toolset == "memory"
