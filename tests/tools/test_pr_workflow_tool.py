import json

import pytest

from tools.pr_workflow_tool import pr_workflow_tool
from agent.pr_workflow import GhCliCommandError


def test_tool_formats_external_comment_with_ai_signature():
    result = json.loads(pr_workflow_tool(action="format_comment", body="LGTM"))

    assert result["ok"] is True
    assert "🤖 This was written by an AI coding agent" in result["body"]


def test_tool_refuses_merge_and_auto_merge_actions():
    for action in ("merge", "auto_merge", "delete_branch"):
        result = json.loads(pr_workflow_tool(action=action, repo="acme/widgets", pr_number=123))

        assert result["ok"] is False
        assert result["error"] == "unsupported_pr_workflow_action"
        assert "does not merge" in result["message"]


def test_tool_returns_scheduler_ready_remediation_request_for_task_creation():
    result = json.loads(
        pr_workflow_tool(
            action="create_remediation_task",
            repo="acme/widgets",
            pr_number=123,
            reason="ci_failure",
            title="Fix failing pytest",
            summary="pytest failed",
        )
    )

    assert result["ok"] is True
    assert result["mode"] == "scheduler_ready_remediation_request"
    assert result["remediation_request"]["reason"] == "ci_failure"
    assert result["remediation_request"]["task_contract"]["context"]["scheduler_ready"] is True
    assert result["remediation_request"]["task_contract"]["context"]["task_tool_available"] is True
    assert result["remediation_request"]["task_contract"]["context"]["repo"] == "acme/widgets"


def test_tool_requires_repo_and_pr_for_polling_before_any_gh_call():
    result = json.loads(pr_workflow_tool(action="poll_pr", repo="", pr_number=None))

    assert result["ok"] is False
    assert result["error"] == "repo_and_pr_number_required"


def test_tool_never_mentions_token_in_unsupported_action_error():
    result = json.loads(
        pr_workflow_tool(action="merge", repo="acme/widgets", pr_number=123, body="token ghp_secret")
    )

    assert "ghp_secret" not in json.dumps(result)


def test_tool_redacts_polling_adapter_errors(monkeypatch):
    class FakeAdapter:
        def poll_pull_request(self, repo, pr_number, **kwargs):
            raise GhCliCommandError("gh failed with ghp_secretTOKEN1234567890")

    monkeypatch.setattr("tools.pr_workflow_tool.GhPrPollingAdapter", FakeAdapter)

    result = json.loads(pr_workflow_tool(action="poll_pr", repo="acme/widgets", pr_number=123))
    dumped = json.dumps(result)

    assert result["ok"] is False
    assert "[REDACTED]" in dumped
    assert "ghp_secret" not in dumped
