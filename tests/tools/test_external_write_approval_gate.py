"""Approval gates for external repository writes and live provider actions.

All command fixtures are inert strings.  These tests exercise classification and
approval decisions only; they must never execute the commands they mention.
"""

import pytest

import tools.approval as approval_module
from tools.approval import (
    approve_session,
    check_all_command_guards,
    classify_command_risk,
    reset_current_session_key,
    set_current_session_key,
)


@pytest.fixture(autouse=True)
def _approval_session(monkeypatch):
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    approval_module._permanent_approved.clear()
    approval_module.clear_session("external-write-test")
    token = set_current_session_key("external-write-test")
    try:
        yield
    finally:
        reset_current_session_key(token)
        approval_module.clear_session("external-write-test")
        approval_module._permanent_approved.clear()


@pytest.mark.parametrize(
    ("command", "risk_class"),
    [
        ("git push", "external_repo_write"),
        ("git push -u origin task-10", "external_repo_write"),
        ("gh pr create --fill", "external_repo_write"),
        ("gh pr merge 123 --squash", "external_repo_write"),
        ("git push origin --delete stale-branch", "external_repo_write"),
        ("git push origin :stale-branch", "external_repo_write"),
        ("gh release create v1.2.3 dist/hermes.zip", "live_deploy_provider_action"),
        ("vercel --prod", "live_deploy_provider_action"),
        ("npm run hostinger:deploy", "live_deploy_provider_action"),
        ("hostinger deploy --site example.invalid", "live_deploy_provider_action"),
        ("cloudflare dns record create example.invalid A 192.0.2.1", "live_deploy_provider_action"),
        ("stripe products create --name example", "live_deploy_provider_action"),
        ("sendgrid mail send --template-id tmpl_123", "live_deploy_provider_action"),
        ("gh pr checks --watch", "indefinite_watcher"),
    ],
)
def test_command_risk_classifier_flags_external_writes_live_actions_and_watchers(command, risk_class):
    risk = classify_command_risk(command)

    assert risk["risk_class"] == risk_class
    assert risk["requires_explicit_approval"] is True


@pytest.mark.parametrize(
    ("command", "risk_class"),
    [
        ("git commit -m 'docs: update runbook'", "local_write"),
        ("python scripts/build_docs.py --out docs/report.md", "local_write"),
    ],
)
def test_command_risk_classifier_separates_local_writes_from_external_actions(command, risk_class):
    risk = classify_command_risk(command)

    assert risk["risk_class"] == risk_class
    assert risk["requires_explicit_approval"] is False


@pytest.mark.parametrize(
    ("command", "risk_class"),
    [
        ("git push", "external_repo_write"),
        ("gh pr create --fill", "external_repo_write"),
        ("gh pr merge 123 --squash", "external_repo_write"),
        ("git push origin --delete stale-branch", "external_repo_write"),
        ("gh release create v1.2.3 dist/hermes.zip", "live_deploy_provider_action"),
        ("vercel --prod", "live_deploy_provider_action"),
        ("npm run hostinger:deploy", "live_deploy_provider_action"),
        ("cloudflare dns record create example.invalid A 192.0.2.1", "live_deploy_provider_action"),
        ("stripe products create --name example", "live_deploy_provider_action"),
        ("sendgrid mail send --template-id tmpl_123", "live_deploy_provider_action"),
        ("gh pr checks --watch", "indefinite_watcher"),
    ],
)
def test_external_writes_live_actions_and_indefinite_watchers_block_without_explicit_approval(command, risk_class):
    result = check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert result["risk_class"] == risk_class
    assert result["outcome"] == "external_action_approval_required"


def test_docs_only_local_commit_is_not_blocked_by_external_write_gate():
    result = check_all_command_guards("git commit -m 'docs: update runbook'", "local")

    assert result["approved"] is True


def test_external_repo_approval_does_not_approve_live_provider_actions():
    approve_session("external-write-test", "external_repo_write")

    pushed = check_all_command_guards("git push", "local")
    deployed = check_all_command_guards("vercel --prod", "local")

    assert pushed["approved"] is True
    assert deployed["approved"] is False
    assert deployed["risk_class"] == "live_deploy_provider_action"


def test_live_provider_approval_does_not_approve_external_repo_writes():
    approve_session("external-write-test", "live_deploy_provider_action")

    deployed = check_all_command_guards("vercel --prod", "local")
    pushed = check_all_command_guards("git push", "local")

    assert deployed["approved"] is True
    assert pushed["approved"] is False
    assert pushed["risk_class"] == "external_repo_write"
