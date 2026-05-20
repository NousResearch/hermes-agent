import json

from tools.memory_evidence_repair_commit_ledger_tool import (
    memory_evidence_repair_commit_ledger_tool,
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


def _allow_decision():
    return {
        "id": "gate-preview-123",
        "preview_id": "preview-123",
        "candidate_id": "repair-123",
        "decision": "allow_manual_commit",
        "provider": "builtin",
        "priority": "high",
        "repair_action": "attach_provenance",
        "reasons": [],
        "required_actions": [],
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_patch": {
            "source_url": "https://example.com/source",
            "observed_at": "2026-05-11T00:00:00Z",
            "verification_signal": "manual_verified",
        },
        "memory_patch_preview": {
            "operation": "merge_evidence_metadata",
            "set": {"evidence": {}},
        },
        "conflict_risk": "low",
    }


def test_tool_uses_latest_manager_state_read_only_markdown():
    result = json.loads(
        memory_evidence_repair_commit_ledger_tool(
            {"format": "markdown"},
            memory_manager=FakeMemoryManager(),
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["summary"]["entry_count"] == 1
    assert result["summary"]["blocked_count"] == 1
    assert "Memory Evidence Repair Commit Ledger" in result["markdown"]


def test_tool_accepts_explicit_decisions_and_ledger_metadata():
    result = json.loads(
        memory_evidence_repair_commit_ledger_tool(
            {
                "decisions": [_allow_decision()],
                "ledger_actor": "tester",
                "ledger_reason": "audit before manual memory write",
            }
        )
    )

    assert result["summary"]["allow_count"] == 1
    assert result["entries"][0]["outcome"] == "allowed_for_manual_commit"
    assert result["entries"][0]["metadata"]["ledger_actor"] == "tester"
    assert result["entries"][0]["metadata"]["ledger_reason"] == (
        "audit before manual memory write"
    )


def test_tool_accepts_candidates_and_builds_full_pipeline():
    result = json.loads(
        memory_evidence_repair_commit_ledger_tool(
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
    assert result["entries"][0]["manual_commit_allowed"] is True


def test_tool_accepts_commit_gate_and_limits_entries():
    result = json.loads(
        memory_evidence_repair_commit_ledger_tool(
            {
                "limit": 1,
                "commit_gate": {
                    "decisions": [_allow_decision(), _allow_decision()],
                    "read_only": True,
                },
            }
        )
    )

    assert result["limited_to"] == 1
    assert result["summary"]["entry_count"] == 1
    assert len(result["entries"]) == 1


def test_tool_rejects_invalid_input_shape():
    result = json.loads(
        memory_evidence_repair_commit_ledger_tool({"commit_gate": "bad"})
    )

    assert result["success"] is False
    assert "commit_gate must be an object" in result["error"]


def test_tool_without_manager_returns_empty_ledger():
    result = json.loads(memory_evidence_repair_commit_ledger_tool({}))

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["summary"]["entry_count"] == 0
    assert result["entries"] == []


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_commit_ledger")

    assert entry is not None
    assert entry.toolset == "memory"
