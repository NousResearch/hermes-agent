import json

from tools.memory_evidence_repair_approval_token_gate_tool import (
    memory_evidence_repair_approval_token_gate_tool,
)
from tools.memory_evidence_repair_commit_receipt_tool import (
    memory_evidence_repair_commit_receipt_tool,
)
from tools.memory_evidence_repair_human_approval_token_tool import (
    memory_evidence_repair_human_approval_token_tool,
)
from tools.registry import registry


class FakeMemoryManager:
    def last_memory_policy_gate(self):
        return {
            "summary": {"input_count": 1, "output_count": 0},
            "provider_results": [
                {
                    "provider": "builtin",
                    "action": "require_provenance_before_reuse",
                    "effect": "filtered",
                    "reason": "Recall candidate lacks explicit provenance required by policy.",
                }
            ],
            "read_only": True,
        }

    def last_recall_audit(self):
        return {"entries": [], "read_only": True}

    def last_recall_diagnostics(self):
        return {"issues": [], "read_only": True}

    def last_memory_auto_policy(self):
        return {"decisions": [], "read_only": True}


def _ready_dry_run():
    return {
        "status": "ready_for_manual_commit",
        "summary": {"manual_commit_allowed": True, "operation_count": 1},
        "checks": [{"id": "commit_gate", "status": "pass"}],
        "operations": [
            {
                "id": "dryrun-op-123",
                "ledger_entry_id": "ledger-123",
                "candidate_id": "repair-123",
                "preview_id": "preview-123",
                "provider": "builtin",
                "repair_action": "attach_provenance",
                "patch_digest": "a" * 64,
                "target": {"candidate_id": "repair-123"},
                "evidence_fields": ["source_url"],
            }
        ],
        "blocking_reasons": [],
        "required_actions": [],
    }


def _token_report():
    result = json.loads(
        memory_evidence_repair_human_approval_token_tool(
            {
                "dry_run": _ready_dry_run(),
                "approver": "han",
                "approval_reason": "final manual approval",
                "expires_in_minutes": 30,
            }
        )
    )
    result["generated_at"] = "2026-05-11T00:00:00+00:00"
    return result


def _verified_gate():
    token_report = _token_report()
    return json.loads(
        memory_evidence_repair_approval_token_gate_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )


def _candidate_args():
    return {
        "candidates": [
            {
                "id": "repair-123",
                "provider": "builtin",
                "priority": "high",
                "repair_action": "attach_provenance",
                "required_evidence": [
                    "source_url",
                    "observed_at",
                    "verification_signal",
                ],
                "proposed_evidence_patch": {
                    "source_url": "https://example.com/source",
                    "observed_at": "2026-05-11T00:00:00Z",
                    "verification_signal": "manual_verified",
                },
            }
        ],
        "approved_candidate_ids": ["repair-123"],
        "user_confirmed": True,
        "existing_snapshots": {
            "source_url": "https://old.example.com/source",
            "observed_at": "2026-05-01T00:00:00Z",
            "verification_signal": "previous_manual_verified",
        },
        "approver": "han",
    }


def test_tool_accepts_explicit_verified_gate_markdown():
    result = json.loads(
        memory_evidence_repair_commit_receipt_tool(
            {
                "token_gate": _verified_gate(),
                "actor": "han",
                "commit_reason": "manual memory evidence repair",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "receipt_ready_for_manual_commit"
    assert result["summary"]["commit_receipt_available"] is True
    assert result["receipts"][0]["actor"] == "han"
    assert "Memory Evidence Repair Commit Receipt" in result["markdown"]


def test_tool_blocks_when_token_already_used():
    gate = _verified_gate()

    result = json.loads(
        memory_evidence_repair_commit_receipt_tool(
            {
                "token_gate": gate,
                "used_token_ids": [gate["token_id"]],
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["receipts"] == []
    assert "regenerate_human_approval_token" in result["required_actions"]


def test_tool_generates_gate_from_token_and_creates_receipt():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_commit_receipt_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
            }
        )
    )

    assert result["status"] == "receipt_ready_for_manual_commit"
    assert result["summary"]["would_record_receipt_count"] == 1
    assert result["receipts"][0]["token_id"] == token_report["tokens"][0]["id"]


def test_tool_blocks_when_generated_gate_is_not_verified():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_commit_receipt_tool(
            {
                "approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["receipts"] == []
    assert "provide_exact_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_gate_available():
    result = json.loads(
        memory_evidence_repair_commit_receipt_tool(
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
        memory_evidence_repair_commit_receipt_tool({"token_gate": "bad"})
    )

    assert result["success"] is False
    assert "token_gate must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_commit_receipt_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["receipt_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_commit_receipt")

    assert entry is not None
    assert entry.toolset == "memory"
