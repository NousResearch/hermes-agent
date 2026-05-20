import json

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


def test_tool_accepts_explicit_ready_dry_run_markdown():
    result = json.loads(
        memory_evidence_repair_human_approval_token_tool(
            {
                "dry_run": _ready_dry_run(),
                "approver": "han",
                "approval_reason": "final manual approval",
                "expires_in_minutes": 45,
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "draft_ready_for_human_approval"
    assert result["summary"]["token_count"] == 1
    assert result["tokens"][0]["approver"] == "han"
    assert result["tokens"][0]["expires_in_minutes"] == 45
    assert "Memory Evidence Repair Human Approval Token" in result["markdown"]


def test_tool_uses_latest_manager_state_and_blocks_when_dry_run_blocked():
    result = json.loads(
        memory_evidence_repair_human_approval_token_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["summary"]["token_count"] == 0
    assert result["tokens"] == []
    assert result["blocking_reasons"]


def test_tool_accepts_candidates_and_existing_snapshots_for_token():
    result = json.loads(
        memory_evidence_repair_human_approval_token_tool(
            {
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
        )
    )

    assert result["status"] == "draft_ready_for_human_approval"
    assert result["summary"]["human_approval_token_available"] is True
    assert result["tokens"][0]["approver"] == "han"
    assert result["tokens"][0]["operation_ids"]


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_human_approval_token_tool({"dry_run": "bad"})
    )

    assert result["success"] is False
    assert "dry_run must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_human_approval_token_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["token_count"] == 0
    assert result["tokens"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_human_approval_token")

    assert entry is not None
    assert entry.toolset == "memory"
