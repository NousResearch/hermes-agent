import json

from tools.memory_evidence_repair_commit_gate_tool import (
    memory_evidence_repair_commit_gate_tool,
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


def _ready_preview():
    return {
        "id": "preview-123",
        "candidate_id": "repair-123",
        "status": "ready_for_manual_apply",
        "provider": "builtin",
        "priority": "high",
        "repair_action": "attach_provenance",
        "operation": "merge_evidence_metadata",
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_patch": {
            "source_url": "https://example.com/source",
            "observed_at": "2026-05-11T00:00:00Z",
            "verification_signal": "manual_verified",
        },
        "memory_patch_preview": {
            "operation": "merge_evidence_metadata",
            "target": {"candidate_id": "repair-123", "provider": "builtin"},
            "set": {"evidence": {}},
        },
        "missing_evidence": [],
    }


def test_tool_uses_latest_manager_state_read_only_markdown():
    result = json.loads(
        memory_evidence_repair_commit_gate_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["summary"]["decision_count"] == 1
    assert result["summary"]["blocked_count"] == 1
    assert "Memory Evidence Repair Commit Gate" in result["markdown"]


def test_tool_accepts_explicit_preview_and_confirmation():
    result = json.loads(
        memory_evidence_repair_commit_gate_tool(
            {
                "preview": _ready_preview(),
                "confirmed_preview_ids": ["preview-123"],
            }
        )
    )

    assert result["summary"]["allow_count"] == 1
    assert result["decisions"][0]["decision"] == "allow_manual_commit"
    assert result["decisions"][0]["reasons"] == []


def test_tool_accepts_candidates_and_builds_gate():
    result = json.loads(
        memory_evidence_repair_commit_gate_tool(
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

    assert result["summary"]["allow_count"] == 1
    assert result["decisions"][0]["decision"] == "allow_manual_commit"


def test_tool_accepts_previews_and_limits_decisions():
    result = json.loads(
        memory_evidence_repair_commit_gate_tool(
            {
                "limit": 1,
                "previews": [_ready_preview(), _ready_preview()],
                "confirmed_preview_ids": ["preview-123"],
            }
        )
    )

    assert result["limited_to"] == 1
    assert result["summary"]["decision_count"] == 1
    assert len(result["decisions"]) == 1


def test_tool_rejects_invalid_input_shape():
    result = json.loads(memory_evidence_repair_commit_gate_tool({"preview": "bad"}))

    assert result["success"] is False
    assert "preview must be an object" in result["error"]


def test_tool_without_manager_returns_empty_gate():
    result = json.loads(memory_evidence_repair_commit_gate_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["decision_count"] == 0
    assert result["decisions"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_commit_gate")

    assert entry is not None
    assert entry.toolset == "memory"
