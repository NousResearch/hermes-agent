import json

from tools.memory_evidence_repair_approval_candidates_tool import (
    memory_evidence_repair_approval_candidates_tool,
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
        memory_evidence_repair_approval_candidates_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["requires_user_confirmation"] is True
    assert "Memory Evidence Repair Approval Candidates" in result["markdown"]


def test_tool_accepts_explicit_repairs_and_limits_candidates():
    result = json.loads(
        memory_evidence_repair_approval_candidates_tool(
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
    assert result["summary"]["candidate_count"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["priority"] == "high"


def test_tool_accepts_explicit_plan_and_proposed_evidence():
    result = json.loads(
        memory_evidence_repair_approval_candidates_tool(
            {
                "plan": {
                    "repairs": [
                        {
                            "provider": "graph",
                            "priority": "medium",
                            "action": "refresh_observation",
                            "source": "diagnostics",
                            "reason": "Recall may be stale.",
                            "required_evidence": ["observed_at", "freshness_check"],
                        }
                    ]
                },
                "proposed_evidence": {
                    "observed_at": "2026-05-11T00:00:00Z",
                    "freshness_check": "confirmed_current",
                },
            }
        )
    )

    assert result["summary"]["candidate_count"] == 1
    assert result["candidates"][0]["proposed_evidence_patch"] == {
        "observed_at": "2026-05-11T00:00:00Z",
        "freshness_check": "confirmed_current",
    }


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_approval_candidates_tool({"plan": "bad"})
    )

    assert result["success"] is False
    assert "plan must be an object" in result["error"]


def test_tool_without_manager_returns_empty_candidates():
    result = json.loads(memory_evidence_repair_approval_candidates_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["candidate_count"] == 0
    assert result["candidates"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_approval_candidates")

    assert entry is not None
    assert entry.toolset == "memory"
