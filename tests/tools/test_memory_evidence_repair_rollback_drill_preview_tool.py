import json

from tools.memory_evidence_repair_post_commit_audit_preview_tool import (
    memory_evidence_repair_post_commit_audit_preview_tool,
)
from tools.memory_evidence_repair_rollback_drill_preview_tool import (
    memory_evidence_repair_rollback_drill_preview_tool,
)
from tools.registry import registry
from tests.tools.test_memory_evidence_repair_post_commit_audit_preview_tool import (
    FakeMemoryManager,
    _executor_preview,
    _token_report,
)


def _planned_post_commit_audit():
    return json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
            {"executor_preview": _executor_preview()}
        )
    )


def _failed_post_commit_audit():
    executor = _executor_preview()
    preview = executor["previews"][0]
    return json.loads(
        memory_evidence_repair_post_commit_audit_preview_tool(
            {
                "executor_preview": executor,
                "observed_memory_patches": {
                    "dryrun-op-123": {"applied": True, "patch_digest": "a" * 64}
                },
                "recorded_receipts": [{"id": preview["receipt_id"]}],
                "used_token_ids": [],
                "released_lock_ids": [preview["lock_id"]],
                "rollback_status": {"status": "snapshot_available"},
            }
        )
    )


def test_tool_accepts_explicit_post_commit_audit_markdown():
    result = json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
            {
                "post_commit_audit": _planned_post_commit_audit(),
                "format": "markdown",
            }
        )
    )

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["status"] == "rollback_drill_preview_ready"
    assert result["previews"][0]["trigger"] == "post_commit_audit_preparedness"
    assert "Memory Evidence Repair Rollback Drill Preview" in result["markdown"]


def test_tool_accepts_failed_audit_and_creates_failure_drill():
    result = json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
            {
                "post_commit_audit": _failed_post_commit_audit(),
                "rollback_plan": {
                    "steps": [
                        {
                            "operation_id": "dryrun-op-123",
                            "patch_digest": "a" * 64,
                        }
                    ]
                },
            }
        )
    )

    assert result["status"] == "rollback_drill_preview_ready"
    assert result["summary"]["failed_audit_step_count"] >= 1
    assert result["summary"]["future_restore_step_count"] == 1
    assert result["previews"][0]["trigger"] == "post_commit_audit_failure"


def test_tool_generates_post_commit_audit_from_token():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
            {
                "approval_token": token_report,
                "confirmation_text": token_report["tokens"][0]["required_confirmation_text"],
                "current_time": "2026-05-11T00:10:00+00:00",
                "actor": "han",
                "lock_owner": "han",
            }
        )
    )

    assert result["status"] == "rollback_drill_preview_ready"
    assert result["summary"]["rollback_drill_preview_available"] is True


def test_tool_blocks_when_generated_post_commit_audit_is_not_usable():
    token_report = _token_report()
    result = json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
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


def test_tool_uses_latest_manager_state_and_blocks_when_no_audit_available():
    result = json.loads(
        memory_evidence_repair_rollback_drill_preview_tool(
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
        memory_evidence_repair_rollback_drill_preview_tool(
            {"post_commit_audit": "bad"}
        )
    )

    assert result["success"] is False
    assert "post_commit_audit must be an object" in result["error"]


def test_tool_without_manager_returns_no_action():
    result = json.loads(memory_evidence_repair_rollback_drill_preview_tool({}))

    assert result["success"] is True
    assert result["status"] == "no_action_needed"
    assert result["summary"]["preview_count"] == 0


def test_tool_is_registered_under_memory_toolset():
    entry = registry.get_entry("memory_evidence_repair_rollback_drill_preview")

    assert entry is not None
    assert entry.toolset == "memory"
