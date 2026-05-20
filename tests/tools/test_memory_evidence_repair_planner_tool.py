import json

from tools.memory_evidence_repair_planner_tool import (
    memory_evidence_repair_planner_tool,
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
        memory_evidence_repair_planner_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["summary"]["repair_count"] == 1
    assert result["summary"]["blocked_count"] == 1
    assert "Memory Evidence Repair Planner" in result["markdown"]


def test_tool_accepts_explicit_payloads_and_limits_repairs():
    result = json.loads(
        memory_evidence_repair_planner_tool(
            {
                "limit": 1,
                "gate": {
                    "provider_results": [
                        {
                            "provider": "builtin",
                            "action": "require_provenance_before_reuse",
                            "effect": "filtered",
                            "reason": "Missing provenance.",
                        }
                    ]
                },
                "diagnostics": {
                    "issues": [
                        {
                            "provider": "graph",
                            "code": "refresh_stale_recall",
                            "reason": "Recall may be stale.",
                        }
                    ]
                },
            }
        )
    )

    assert result["limited_to"] == 1
    assert len(result["repairs"]) == 1
    assert result["repairs"][0]["priority"] == "high"


def test_tool_rejects_invalid_input_shape():
    result = json.loads(memory_evidence_repair_planner_tool({"gate": "bad"}))

    assert result["success"] is False
    assert "gate must be an object" in result["error"]


def test_tool_without_manager_returns_empty_plan():
    result = json.loads(memory_evidence_repair_planner_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["repair_count"] == 0
    assert result["repairs"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_planner")

    assert entry is not None
    assert entry.toolset == "memory"
