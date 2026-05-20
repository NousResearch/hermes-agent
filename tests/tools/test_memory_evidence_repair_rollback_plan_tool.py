import json

from tools.memory_evidence_repair_rollback_plan_tool import (
    memory_evidence_repair_rollback_plan_tool,
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


def _allowed_entry():
    return {
        "id": "ledger-123",
        "sequence": 1,
        "outcome": "allowed_for_manual_commit",
        "decision_id": "gate-preview-123",
        "preview_id": "preview-123",
        "candidate_id": "repair-123",
        "provider": "builtin",
        "priority": "high",
        "repair_action": "attach_provenance",
        "evidence_fields": ["source_url", "observed_at", "verification_signal"],
        "patch_digest": "a" * 64,
        "manual_commit_allowed": True,
        "blocked": False,
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
    }


def test_tool_uses_latest_manager_state_read_only_markdown():
    result = json.loads(
        memory_evidence_repair_rollback_plan_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["summary"]["rollback_step_count"] == 1
    assert result["summary"]["no_action_count"] == 1
    assert "Memory Evidence Repair Rollback Plan" in result["markdown"]


def test_tool_accepts_explicit_entries_and_snapshots():
    result = json.loads(
        memory_evidence_repair_rollback_plan_tool(
            {
                "entries": [_allowed_entry()],
                "pre_commit_snapshots": {
                    "ledger-123": {
                        "evidence": {
                            "source_url": "https://old.example.com/source",
                            "observed_at": "2026-05-01T00:00:00Z",
                            "verification_signal": "previous_manual_verified",
                        }
                    }
                },
            }
        )
    )

    assert result["summary"]["ready_count"] == 1
    assert result["steps"][0]["status"] == "ready_for_manual_rollback"
    assert result["steps"][0]["inverse_patch_preview"]["set"]["evidence"] == {
        "source_url": "https://old.example.com/source",
        "observed_at": "2026-05-01T00:00:00Z",
        "verification_signal": "previous_manual_verified",
    }


def test_tool_accepts_candidates_and_builds_full_pipeline():
    result = json.loads(
        memory_evidence_repair_rollback_plan_tool(
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
            }
        )
    )

    assert result["summary"]["snapshot_required_count"] == 1
    assert result["steps"][0]["status"] == "snapshot_required"


def test_tool_accepts_ledger_and_limits_steps():
    result = json.loads(
        memory_evidence_repair_rollback_plan_tool(
            {
                "limit": 1,
                "ledger": {"entries": [_allowed_entry(), _allowed_entry()]},
            }
        )
    )

    assert result["limited_to"] == 1
    assert result["summary"]["rollback_step_count"] == 1
    assert len(result["steps"]) == 1


def test_tool_rejects_invalid_input_shape():
    result = json.loads(memory_evidence_repair_rollback_plan_tool({"ledger": "bad"}))

    assert result["success"] is False
    assert "ledger must be an object" in result["error"]


def test_tool_without_manager_returns_empty_plan():
    result = json.loads(memory_evidence_repair_rollback_plan_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["rollback_step_count"] == 0
    assert result["steps"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_rollback_plan")

    assert entry is not None
    assert entry.toolset == "memory"
