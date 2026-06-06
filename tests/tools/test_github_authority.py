"""Tests for GitHub authority allowlist/preflight classification."""

from unittest.mock import Mock, patch

from tools.approval import check_all_command_guards
from tools.github_authority import classify_github_command


def _allow_tirith():
    return {"action": "allow", "findings": [], "summary": ""}


def test_green_gh_pr_view_is_classified_read_only():
    decision = classify_github_command("gh pr view 25 --repo yunuenmf/hermes-maintenance")

    assert decision.tier == "green"
    assert decision.requires_preflight is False
    assert decision.command_family == "gh pr"


def test_yellow_gh_issue_comment_is_classified_scoped_write():
    decision = classify_github_command("gh issue comment 36 --repo yunuenmf/hermes-maintenance --body-file /tmp/body.md")

    assert decision.tier == "yellow"
    assert decision.requires_preflight is True
    assert decision.command_family == "gh issue"


def test_compound_command_is_not_green_allowlisted():
    decision = classify_github_command("gh pr view 25 && gh pr merge 25")

    assert decision.tier == "unknown"


def test_red_repo_delete_is_classified_red():
    decision = classify_github_command("gh repo delete yunuenmf/hermes-maintenance --yes")

    assert decision.tier == "red"
    assert decision.requires_preflight is True
    assert "approval" in decision.reason.lower()


def test_yellow_pr_merge_requires_preflight():
    decision = classify_github_command("gh pr merge 25 --squash")

    assert decision.tier == "yellow"
    assert decision.requires_preflight is True
    assert "preflight" in decision.reason.lower()


def test_green_github_command_does_not_prompt_for_approval(monkeypatch):
    callback = Mock(return_value="deny")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

    with patch("tools.tirith_security.check_command_security", return_value=_allow_tirith()), \
         patch("tools.approval._get_approval_mode", return_value="manual"):
        result = check_all_command_guards(
            "gh pr view 25 --repo yunuenmf/hermes-maintenance",
            "local",
            approval_callback=callback,
        )

    assert result["approved"] is True
    callback.assert_not_called()


def test_red_github_command_requires_manual_approval(monkeypatch):
    callback = Mock(return_value="deny")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

    with patch("tools.tirith_security.check_command_security", return_value=_allow_tirith()), \
         patch("tools.approval._get_approval_mode", return_value="manual"):
        result = check_all_command_guards(
            "gh repo delete yunuenmf/hermes-maintenance --yes",
            "local",
            approval_callback=callback,
        )

    assert result["approved"] is False
    assert "denied" in result["message"].lower()
    callback.assert_called_once()
    assert "red" in callback.call_args.args[1].lower()
    assert "yunuen" in callback.call_args.args[1].lower()


def test_yellow_github_command_requires_preflight_evidence_not_blanket_allow(monkeypatch):
    callback = Mock(return_value="deny")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

    with patch("tools.tirith_security.check_command_security", return_value=_allow_tirith()), \
         patch("tools.approval._get_approval_mode", return_value="smart"), \
         patch("tools.approval._smart_approve") as smart_approve:
        result = check_all_command_guards(
            "gh pr merge 25 --squash",
            "local",
            approval_callback=callback,
        )

    assert result["approved"] is False
    callback.assert_called_once()
    smart_approve.assert_not_called()
    assert "yellow" in callback.call_args.args[1].lower()
    assert "preflight" in callback.call_args.args[1].lower()
