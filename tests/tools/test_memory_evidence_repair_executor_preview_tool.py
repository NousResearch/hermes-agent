import json

from tools.memory_evidence_repair_executor_preview_tool import (
    memory_evidence_repair_executor_preview_tool,
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


def _write_lock_gate():
    token_report = _token_report()
    return json.loads(
        memory_evidence_repair_write_lock_gate_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "commit_reason": "manual memory evidence repair",
                "lock_owner": "han",
                "lock_ttl_minutes": 15,
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


def test_tool_accepts_explicit_write_lock_gate_markdown():
    result = json.loads(
        memory_evidence_repair_executor_preview_tool(
            {
                "write_lock_gate": _write_lock_gate(),
                "current_time": "2026-05-11T00:12:00+00:00",
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "executor_preview_ready_for_manual_commit"
    assert result["summary"]["manual_executor_preview_ready"] is True
    assert result["summary"]["future_write_step_count"] == 1
    assert result["previews"][0]["steps"][1]["action"] == "apply_evidence_metadata_patch"
    assert "Memory Evidence Repair Executor Preview" in result["markdown"]


def test_tool_blocks_expired_write_lock_gate():
    result = json.loads(
        memory_evidence_repair_executor_preview_tool(
            {
                "write_lock_gate": _write_lock_gate(),
                "current_time": "2026-05-11T00:26:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "regenerate_write_lock_gate" in result["required_actions"]


def test_tool_generates_write_lock_gate_from_token_and_creates_preview():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_executor_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "lock_owner": "han",
            }
        )
    )

    assert result["status"] == "executor_preview_ready_for_manual_commit"
    assert result["summary"]["executor_preview_available"] is True
    assert result["previews"][0]["operation_ids"] == ["dryrun-op-123"]


def test_tool_blocks_when_generated_write_lock_gate_is_not_ready():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_executor_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": "wrong confirmation",
                "current_time": "2026-05-11T00:10:00+00:00",
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "provide_exact_confirmation_text" in result["required_actions"]


def test_tool_uses_latest_manager_state_and_blocks_when_no_lock_available():
    result = json.loads(
        memory_evidence_repair_executor_preview_tool(
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
        memory_evidence_repair_executor_preview_tool({"write_lock_gate": "bad"})
    )

    assert result["success"] is False
    assert "write_lock_gate must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_executor_preview_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["preview_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_executor_preview")

    assert entry is not None
    assert entry.toolset == "memory"
