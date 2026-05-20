import json

from tools.memory_evidence_repair_apply_preview_tool import (
    memory_evidence_repair_apply_preview_tool,
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
        memory_evidence_repair_apply_preview_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["summary"]["preview_count"] == 1
    assert result["summary"]["missing_evidence_count"] == 1
    assert "Memory Evidence Repair Apply Preview" in result["markdown"]


def test_tool_accepts_explicit_candidates_and_approved_ids():
    result = json.loads(
        memory_evidence_repair_apply_preview_tool(
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
                            "source_url": "<pending>",
                            "observed_at": "<pending>",
                            "verification_signal": "<pending>",
                        },
                    }
                ],
                "approved_candidate_ids": ["repair-123"],
                "proposed_evidence": {
                    "source_url": "https://example.com/source",
                    "observed_at": "2026-05-11T00:00:00Z",
                    "verification_signal": "manual_verified",
                },
            }
        )
    )

    assert result["summary"]["ready_count"] == 1
    assert result["previews"][0]["status"] == "ready_for_manual_apply"
    assert result["previews"][0]["evidence_patch"] == {
        "source_url": "https://example.com/source",
        "observed_at": "2026-05-11T00:00:00Z",
        "verification_signal": "manual_verified",
    }


def test_tool_accepts_repairs_and_limits_previews():
    result = json.loads(
        memory_evidence_repair_apply_preview_tool(
            {
                "limit": 1,
                "repairs": [
                    {
                        "provider": "builtin",
                        "priority": "high",
                        "action": "attach_provenance",
                        "source": "policy_gate",
                        "reason": "Missing provenance.",
                    },
                    {
                        "provider": "graph",
                        "priority": "medium",
                        "action": "refresh_observation",
                        "source": "diagnostics",
                        "reason": "Stale recall.",
                    },
                ],
            }
        )
    )

    assert result["limited_to"] == 1
    assert result["summary"]["preview_count"] == 1
    assert len(result["previews"]) == 1


def test_tool_rejects_invalid_input_shape():
    result = json.loads(memory_evidence_repair_apply_preview_tool({"approval": "bad"}))

    assert result["success"] is False
    assert "approval must be an object" in result["error"]


def test_tool_without_manager_returns_empty_preview():
    result = json.loads(memory_evidence_repair_apply_preview_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["preview_count"] == 0
    assert result["previews"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_apply_preview")

    assert entry is not None
    assert entry.toolset == "memory"
