import json

from tools.memory_evidence_repair_commit_receipt_tool import (
    memory_evidence_repair_commit_receipt_tool,
)
from tools.memory_evidence_repair_human_approval_token_tool import (
    memory_evidence_repair_human_approval_token_tool,
)
from tools.memory_evidence_repair_write_lock_gate_tool import (
    memory_evidence_repair_write_lock_gate_tool,
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


def _commit_receipt():
    token_report = _token_report()
    return json.loads(
        memory_evidence_repair_commit_receipt_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "commit_reason": "manual memory evidence repair",
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


def test_tool_accepts_explicit_commit_receipt_markdown():
    result = json.loads(
        memory_evidence_repair_write_lock_gate_tool(
            {
                "commit_receipt": _commit_receipt(),
                "lock_owner": "han",
                "lock_ttl_minutes": 15,
                "current_time": "2026-05-11T00:20:00+00:00",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "write_lock_ready_for_manual_commit"
    assert result["summary"]["executor_dry_run_allowed"] is True
    assert result["locks"][0]["owner"] == "han"
    assert "Memory Evidence Repair Write Lock Gate" in result["markdown"]


def test_tool_blocks_on_active_lock_conflict():
    result = json.loads(
        memory_evidence_repair_write_lock_gate_tool(
            {
                "commit_receipt": _commit_receipt(),
                "active_locks": [
                    {
                        "id": "write-lock-existing",
                        "status": "active",
                        "scope": "manual_memory_evidence_repair_commit",
                        "operation_ids": ["dryrun-op-123"],
                        "expires_at": "2026-05-11T00:40:00+00:00",
                    }
                ],
                "current_time": "2026-05-11T00:20:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["locks"] == []
    assert result["summary"]["active_lock_conflict_count"] == 1
    assert "wait_for_write_lock_release" in result["required_actions"]


def test_tool_generates_receipt_from_token_and_creates_write_lock():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_write_lock_gate_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "lock_owner": "han",
            }
        )
    )

    assert result["status"] == "write_lock_ready_for_manual_commit"
    assert result["summary"]["write_lock_available"] is True
    assert result["locks"][0]["token_id"] == token_report["tokens"][0]["id"]


def test_tool_blocks_when_generated_receipt_is_not_ready():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_write_lock_gate_tool(
            {
                "approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["locks"] == []
    assert "provide_exact_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_receipt_available():
    result = json.loads(
        memory_evidence_repair_write_lock_gate_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["locks"] == []
    assert result["blocking_reasons"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_write_lock_gate_tool({"commit_receipt": "bad"})
    )

    assert result["success"] is False
    assert "commit_receipt must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_write_lock_gate_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["lock_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_write_lock_gate")

    assert entry is not None
    assert entry.toolset == "memory"
