import json

from tools.memory_evidence_repair_snapshot_planner_tool import (
    memory_evidence_repair_snapshot_planner_tool,
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


def _snapshot_required_step():
    return {
        "id": "rollback-123",
        "ledger_entry_id": "ledger-123",
        "sequence": 1,
        "status": "snapshot_required",
        "rollback_action": "restore_evidence_metadata",
        "provider": "builtin",
        "repair_action": "attach_provenance",
        "candidate_id": "repair-123",
        "preview_id": "preview-123",
        "patch_digest": "a" * 64,
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_fields": ["source_url", "observed_at", "verification_signal"],
    }


def test_tool_uses_latest_manager_state_read_only_markdown():
    result = json.loads(
        memory_evidence_repair_snapshot_planner_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["summary"]["snapshot_request_count"] == 1
    assert result["summary"]["no_action_count"] == 1
    assert "Memory Evidence Repair Snapshot Planner" in result["markdown"]


def test_tool_accepts_explicit_steps_and_limits_requests():
    result = json.loads(
        memory_evidence_repair_snapshot_planner_tool(
            {
                "limit": 1,
                "steps": [_snapshot_required_step(), _snapshot_required_step()],
            }
        )
    )

    assert result["limited_to"] == 1
    assert result["summary"]["snapshot_request_count"] == 1
    assert result["requests"][0]["status"] == "capture_required"
    assert result["requests"][0]["blocks_manual_commit"] is True


def test_tool_accepts_entries_and_existing_snapshots():
    result = json.loads(
        memory_evidence_repair_snapshot_planner_tool(
            {
                "entries": [
                    {
                        "id": "ledger-123",
                        "outcome": "allowed_for_manual_commit",
                        "manual_commit_allowed": True,
                        "provider": "builtin",
                        "repair_action": "attach_provenance",
                        "candidate_id": "repair-123",
                        "preview_id": "preview-123",
                        "patch_digest": "a" * 64,
                        "target": {"candidate_id": "repair-123", "provider": "builtin"},
                        "evidence_fields": ["source_url"],
                    }
                ],
                "existing_snapshots": {
                    "ledger-123": {"evidence": {"source_url": "https://old.example.com"}}
                },
            }
        )
    )

    assert result["summary"]["already_available_count"] == 1
    assert result["requests"][0]["status"] == "already_available"


def test_tool_accepts_candidates_and_builds_full_pipeline():
    result = json.loads(
        memory_evidence_repair_snapshot_planner_tool(
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

    assert result["summary"]["capture_required_count"] == 1
    assert result["summary"]["blocks_manual_commit"] is True
    assert result["requests"][0]["status"] == "capture_required"


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_snapshot_planner_tool({"rollback_plan": "bad"})
    )

    assert result["success"] is False
    assert "rollback_plan must be an object" in result["error"]


def test_tool_without_manager_returns_empty_plan():
    result = json.loads(memory_evidence_repair_snapshot_planner_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["snapshot_request_count"] == 0
    assert result["requests"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_snapshot_planner")

    assert entry is not None
    assert entry.toolset == "memory"
