import json

from tools.memory_evidence_repair_manual_commit_dry_run_tool import (
    memory_evidence_repair_manual_commit_dry_run_tool,
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


def test_tool_uses_latest_manager_state_read_only_markdown():
    result = json.loads(
        memory_evidence_repair_manual_commit_dry_run_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "blocked"
    assert result["summary"]["has_blocks"] is True
    assert "Memory Evidence Repair Manual Commit Dry-Run" in result["markdown"]


def test_tool_accepts_candidates_and_existing_snapshots_for_ready_dry_run():
    result = json.loads(
        memory_evidence_repair_manual_commit_dry_run_tool(
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
            }
        )
    )

    assert result["status"] == "ready_for_manual_commit"
    assert result["summary"]["manual_commit_allowed"] is True
    assert result["summary"]["operation_count"] == 1
    assert result["blocking_reasons"] == []


def test_tool_blocks_when_snapshot_plan_requires_capture():
    result = json.loads(
        memory_evidence_repair_manual_commit_dry_run_tool(
            {
                "snapshot_plan": {
                    "summary": {
                        "snapshot_request_count": 1,
                        "capture_required_count": 1,
                        "blocking_count": 1,
                    },
                    "requests": [{"status": "capture_required"}],
                },
                "commit_gate": {
                    "summary": {
                        "decision_count": 1,
                        "allow_count": 1,
                        "blocked_count": 0,
                        "needs_confirmation_count": 0,
                    },
                    "decisions": [{"decision": "allow_manual_commit"}],
                },
                "ledger": {
                    "summary": {
                        "entry_count": 1,
                        "allow_count": 1,
                        "blocked_count": 0,
                        "followup_count": 0,
                    },
                    "entries": [
                        {
                            "id": "ledger-123",
                            "manual_commit_allowed": True,
                            "patch_digest": "a" * 64,
                            "evidence_fields": ["source_url"],
                        }
                    ],
                },
                "rollback_plan": {
                    "summary": {
                        "rollback_step_count": 1,
                        "ready_count": 1,
                        "snapshot_required_count": 0,
                    },
                    "steps": [{"status": "ready_for_manual_rollback"}],
                },
            }
        )
    )

    assert result["status"] == "blocked"
    assert "capture_pre_commit_snapshots" in result["required_actions"]


def test_tool_accepts_limit_for_operations():
    result = json.loads(
        memory_evidence_repair_manual_commit_dry_run_tool(
            {
                "limit": 1,
                "snapshot_plan": {
                    "summary": {
                        "snapshot_request_count": 1,
                        "capture_required_count": 0,
                        "blocking_count": 0,
                    },
                    "requests": [{"status": "already_available"}],
                },
                "commit_gate": {
                    "summary": {
                        "decision_count": 1,
                        "allow_count": 1,
                        "blocked_count": 0,
                        "needs_confirmation_count": 0,
                    },
                    "decisions": [{"decision": "allow_manual_commit"}],
                },
                "ledger": {
                    "summary": {
                        "entry_count": 2,
                        "allow_count": 2,
                        "blocked_count": 0,
                        "followup_count": 0,
                    },
                    "entries": [
                        {"id": "ledger-1", "manual_commit_allowed": True},
                        {"id": "ledger-2", "manual_commit_allowed": True},
                    ],
                },
                "rollback_plan": {
                    "summary": {
                        "rollback_step_count": 1,
                        "ready_count": 1,
                        "snapshot_required_count": 0,
                    },
                    "steps": [{"status": "ready_for_manual_rollback"}],
                },
            }
        )
    )

    assert result["limited_to"] == 1
    assert result["summary"]["operation_count"] == 1
    assert len(result["operations"]) == 1


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_manual_commit_dry_run_tool({"snapshot_plan": "bad"})
    )

    assert result["success"] is False
    assert "snapshot_plan must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_manual_commit_dry_run_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["operation_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_manual_commit_dry_run")

    assert entry is not None
    assert entry.toolset == "memory"
