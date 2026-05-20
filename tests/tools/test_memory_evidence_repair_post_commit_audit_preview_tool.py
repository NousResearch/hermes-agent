import json

from tools.memory_evidence_repair_executor_preview_tool import (
    memory_evidence_repair_executor_preview_tool,
)
from tools.memory_evidence_repair_human_approval_token_tool import (
    memory_evidence_repair_human_approval_token_tool,
)
from tools.memory_evidence_repair_post_commit_audit_preview_tool import (
    memory_evidence_repair_post_commit_audit_preview_tool,
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


def _executor_preview():
    token_report = _token_report()
    return json.loads(
        memory_evidence_repair_executor_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "commit_reason": "manual memory evidence repair",
                "lock_owner": "han",
            }
        )
    )


def test_tool_accepts_explicit_executor_preview_markdown():
    result = json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
            {
                "executor_preview": _executor_preview(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "post_commit_audit_preview_ready"
    assert result["summary"]["planned_audit_step_count"] == 5
    assert "Memory Evidence Repair Post-Commit Audit Preview" in result["markdown"]


def test_tool_accepts_observed_state_and_passes():
    executor = _executor_preview()
    preview = executor["previews"][0]
    result = json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
            {
                "executor_preview": executor,
                "observed_memory_patches": {
                    "dryrun-op-123": {"applied": True, "patch_digest": "a" * 64}
                },
                "recorded_receipts": [{"id": preview["receipt_id"]}],
                "used_token_ids": [preview["token_id"]],
                "released_lock_ids": [preview["lock_id"]],
                "rollback_status": {"status": "snapshot_available"},
            }
        )
    )

    assert result["status"] == "post_commit_audit_preview_ready"
    assert result["summary"]["observed_pass_step_count"] == 5
    assert result["summary"]["observed_fail_step_count"] == 0


def test_tool_blocks_bad_observed_patch_digest():
    executor = _executor_preview()
    preview = executor["previews"][0]
    result = json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
            {
                "executor_preview": executor,
                "observed_memory_patches": {
                    "dryrun-op-123": {"applied": True, "patch_digest": "b" * 64}
                },
                "recorded_receipts": [{"id": preview["receipt_id"]}],
                "used_token_ids": [preview["token_id"]],
                "released_lock_ids": [preview["lock_id"]],
                "rollback_status": {"status": "snapshot_available"},
            }
        )
    )

    assert result["status"] == "blocked"
    assert result["previews"] == []
    assert "investigate_post_commit_audit_failures" in result["required_actions"]


def test_tool_generates_executor_preview_from_token():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "lock_owner": "han",
            }
        )
    )

    assert result["status"] == "post_commit_audit_preview_ready"
    assert result["summary"]["post_commit_audit_preview_available"] is True


def test_tool_blocks_when_generated_executor_preview_is_not_ready():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
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


def test_tool_uses_latest_manager_state_and_blocks_when_no_executor_available():
    result = json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
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
        memory_evidence_repair_post_commit_audit_preview_tool(
            {"executor_preview": "bad"}
        )
    )

    assert result["success"] is False
    assert "executor_preview must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_post_commit_audit_preview_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["preview_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_post_commit_audit_preview")

    assert entry is not None
    assert entry.toolset == "memory"
