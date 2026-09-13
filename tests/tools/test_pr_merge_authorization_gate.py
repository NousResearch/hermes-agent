"""Fail-closed authorization contract for GitHub pull-request merges."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

import tools.approval as approval
import tools.terminal_tool as terminal_tool


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    approval.clear_session("pr-merge-test")
    token = approval.set_current_session_key("pr-merge-test")
    obs_tokens = approval.set_current_observability_context(
        session_id="session-123", turn_id="turn-456", tool_call_id="tool-789"
    )
    yield
    approval.reset_current_observability_context(obs_tokens)
    approval.reset_current_session_key(token)
    approval.clear_session("pr-merge-test")


def _target(head="a" * 40):
    return {
        "repository": "batumilove/hermes-agent",
        "pull_request": 71,
        "head_sha": head,
        "merge_method": "squash",
        "source": "gh-pr-merge",
    }


@pytest.mark.parametrize(
    "command, expected",
    [
        (
            f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
            {"repository": "batumilove/hermes-agent", "pull_request": 71, "merge_method": "squash", "source": "gh-pr-merge", "expected_head_sha": "a" * 40},
        ),
        (
            f"gh api --method PUT repos/batumilove/hermes-agent/pulls/71/merge -f merge_method=squash -f sha={'a' * 40}",
            {"repository": "batumilove/hermes-agent", "pull_request": 71, "merge_method": "squash", "source": "gh-api-merge", "expected_head_sha": "a" * 40},
        ),
    ],
)
def test_detects_exact_pr_merge_target(command, expected):
    assert approval.detect_pr_merge_target(command, cwd="/tmp/repo") == expected


@pytest.mark.parametrize(
    "command",
    [
        "echo 'gh pr merge 71 --repo batumilove/hermes-agent'",
        "gh pr view 71 --repo batumilove/hermes-agent",
        "gh api repos/batumilove/hermes-agent/pulls/71/merge",
        "gh api --method GET repos/batumilove/hermes-agent/pulls/71/merge",
        "gh pr merge $PR --repo batumilove/hermes-agent --squash",
        "gh pr merge 71 --repo batumilove/hermes-agent --squash",
        f"gh api --method PUT repos/batumilove/hermes-agent/pulls/71/merge -f sha={'a' * 40} --input payload.json",
    ],
)
def test_non_merge_or_unresolved_forms_do_not_become_approvable_targets(command):
    assert approval.detect_pr_merge_target(command, cwd="/tmp/repo") is None


def test_exact_target_requires_once_only_human_approval_even_with_yolo(monkeypatch):
    seen = {}

    def callback(command, description, **kwargs):
        seen.update(command=command, description=description, kwargs=kwargs)
        return "once"

    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setattr(approval, "_resolve_pr_merge_target", Mock(side_effect=[_target(), _target()]))

    result = approval.check_all_command_guards(
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        "local",
        cwd="/tmp/repo",
        approval_callback=callback,
    )

    assert result["approved"] is True
    assert result["user_approved"] is True
    assert result["pr_merge_authorized"] is True
    assert result["authorization_receipt_id"]
    assert "batumilove/hermes-agent" in seen["description"]
    assert "71" in seen["description"]
    assert "a" * 40 in seen["description"]
    assert seen["kwargs"]["allow_session"] is False
    assert seen["kwargs"]["allow_permanent"] is False
    receipt_path = (
        Path(os.environ["HERMES_HOME"])
        / "approval-receipts"
        / "pr-merges"
        / f"{result['authorization_receipt_id']}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert receipt["state"] == "consumed_before_execution"
    assert receipt["repository"] == "batumilove/hermes-agent"
    assert receipt["pull_request"] == 71
    assert receipt["head_sha"] == "a" * 40
    assert receipt["merge_method"] == "squash"
    assert receipt["session_id"] == "session-123"
    assert receipt["turn_id"] == "turn-456"
    assert receipt["tool_call_id"] == "tool-789"
    assert receipt["approval_request_id"]


def test_stale_head_after_prompt_is_blocked(monkeypatch):
    callback = Mock(return_value="once")
    monkeypatch.setattr(
        approval,
        "_resolve_pr_merge_target",
        Mock(side_effect=[_target("a" * 40), _target("b" * 40)]),
    )

    result = approval.check_all_command_guards(
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        "local",
        cwd="/tmp/repo",
        approval_callback=callback,
    )

    assert result["approved"] is False
    assert result["outcome"] == "target_changed"
    assert "changed after approval" in result["message"].lower()


@pytest.mark.parametrize("choice", ["session", "always"])
def test_session_and_permanent_choices_cannot_authorize_merge(monkeypatch, choice):
    monkeypatch.setattr(approval, "_resolve_pr_merge_target", Mock(return_value=_target()))

    result = approval.check_all_command_guards(
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        "local",
        cwd="/tmp/repo",
        approval_callback=Mock(return_value=choice),
    )

    assert result["approved"] is False
    assert result["outcome"] == "invalid_approval_scope"


def test_cron_merge_is_blocked_even_when_cron_mode_approves(monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "approve")
    monkeypatch.setattr(approval, "_resolve_pr_merge_target", Mock(return_value=_target()))

    result = approval.check_all_command_guards(
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        "local",
        cwd="/tmp/repo",
    )

    assert result["approved"] is False
    assert result["outcome"] == "interactive_user_required"


def test_unresolved_exact_target_fails_closed(monkeypatch):
    monkeypatch.setattr(approval, "_resolve_pr_merge_target", Mock(return_value=None))

    result = approval.check_all_command_guards(
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        "local",
        cwd="/tmp/repo",
        approval_callback=Mock(return_value="once"),
    )

    assert result["approved"] is False
    assert result["outcome"] == "target_unresolved"


@pytest.mark.parametrize(
    "command",
    [
        f"curl -X PUT https://api.github.com/repos/batumilove/hermes-agent/pulls/71/merge -d '{{\"sha\":\"{'a' * 40}\",\"merge_method\":\"squash\"}}'",
        "gh api graphql -f 'query=mutation { mergePullRequest(input: {}) { pullRequest { id } } }'",
    ],
)
def test_direct_rest_and_graphql_merge_routes_fail_closed(monkeypatch, command):
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    result = approval.check_all_command_guards(command, "local", cwd="/tmp/repo")

    assert result["approved"] is False
    assert result["outcome"] == "target_unresolved"


def test_approve_all_does_not_resolve_protected_merge_gate():
    session = "protected-queue"
    approval.clear_session(session)
    entry = approval._ApprovalEntry(
        {
            "command": f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
            "protected_once": True,
        }
    )
    approval._gateway_queues[session] = [entry]

    assert approval.resolve_gateway_approval(session, "once", resolve_all=True) == 0
    assert entry.event.is_set() is False
    assert approval.list_gateway_approvals(session)
    approval.clear_session(session)


def test_terminal_force_replay_cannot_bypass_protected_gate(monkeypatch, tmp_path):
    guard = Mock(
        return_value={
            "approved": False,
            "message": "BLOCKED: exact approval required",
            "description": "protected merge",
        }
    )
    monkeypatch.setattr(terminal_tool, "_check_all_guards", guard)
    monkeypatch.setattr(
        terminal_tool,
        "_check_live_state_db_guard",
        lambda **_kwargs: (False, None),
    )
    command = (
        "gh pr merge 71 --repo batumilove/hermes-agent --squash "
        f"--match-head-commit {'a' * 40}"
    )

    result = terminal_tool.json.loads(
        terminal_tool.terminal_tool(command, force=True, workdir=str(tmp_path))
    )

    assert result["status"] == "blocked"
    guard.assert_called_once()


def _canonical_merge(pr=71, repository="batumilove/hermes-agent", head="a" * 40):
    return (
        f"gh pr merge {pr} --repo {repository} --squash "
        f"--match-head-commit {head}"
    )


@pytest.mark.parametrize(
    "command",
    [
        f"{_canonical_merge()} ; {_canonical_merge(pr=72, head='b' * 40)}",
        f"bash -c {_canonical_merge()!r}",
        f"sudo {_canonical_merge()}",
        f"sudo -u root {_canonical_merge()}",
        f"env GH_HOST=ghe.example {_canonical_merge()}",
        f"command {_canonical_merge()}",
        f"exec {_canonical_merge()}",
        f"nohup {_canonical_merge()}",
        f"timeout 30 {_canonical_merge()}",
        f"timeout 30 gh api repos/batumilove/hermes-agent/pulls/71/merge -X PUT -f sha={'a' * 40} -f merge_method=squash",
        "sudo curl -X POST https://api.github.com/graphql --data @payload.json",
        "curl -X POST https://api.github.com/graphql "
        "-d '{\"query\":\"mutation { mergePullRequest(input: {}) { pullRequest { id } } }\"}'",
        "curl -X POST https://api.github.com/graphql --data @payload.json",
        "gh api graphql --input payload.json",
        "curl -X PUT 'https://api.github.com/repos/batumilove/hermes-agent/"
        "pulls/71/merge?apiVersion=2022-11-28'",
        "curl -X PUT 'https://API.GITHUB.COM:443/repos/batumilove/hermes-agent/pulls/71/merge'",
        "curl -X PUT '$API/repos/batumilove/hermes-agent/pulls/71/merge'",
        "curl -X POST '$API/graphql' --data @payload.json",
        "curl -X PUT 'https://api.github.com/repos/batumilove/hermes-agent/pulls/71/{merge,noop}'",
        "curl -X PUT 'https://api.github.com/repos/batumilove/hermes-agent/pulls/71/{m,noop}erge'",
        "curl -X PUT 'https://api.github.com/repos/batumilove/hermes-agent/pulls/71/%6Derge'",
        "curl -X PUT 'https://api.github.com/repos/batumilove/hermes-agent/pulls/71/./merge'",
        f"gh api --method PUT repos/batumilove/hermes-agent/pulls/71/%6Derge -f merge_method=squash -f sha={'a' * 40}",
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40} # '",
        f"g\\\nh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        f"gh $'pr' merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        f"gh p$'r' $'m\\x65rge' 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        f"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}; : $'\\x27'",
        f"python -c 'import os; os.execvp(\"gh\", [\"gh\", \"pr\", \"merge\", \"71\", \"--repo\", \"batumilove/hermes-agent\", \"--squash\", \"--match-head-commit\", \"{'a' * 40}\"])'",
        f"python3.11 -c 'import os; os.system(\"gh pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}\")'",
        f"gh --hostname ghe.example api --method PUT repos/batumilove/hermes-agent/pulls/71/merge -f merge_method=squash -f sha={'a' * 40}",
        f"gh pr --repo batumilove/hermes-agent merge 71 --squash --match-head-commit {'a' * 40}",
        f"gh.exe pr merge 71 --repo batumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        f"gh api --method PUT repos/batumilove/hermes-agent/pulls/71/merge -f merge_method=squash -f sha={'a' * 40} -fmerge_method=merge",
    ],
)
def test_noncanonical_merge_carriers_are_detected_but_never_approvable(command):
    assert approval.is_protected_pr_merge_command(command) is True
    assert approval.detect_pr_merge_target(command, cwd="/tmp/repo") is None


@pytest.mark.parametrize(
    "command",
    [
        f"{_canonical_merge()} --repo attacker/other",
        f"gh pr merge 71 -Rbatumilove/hermes-agent --squash --match-head-commit {'a' * 40}",
        f"gh pr merge 71 --squash --match-head-commit {'a' * 40}",
        f"{_canonical_merge()} --match-head-commit {'b' * 40}",
        f"gh api repos/batumilove/hermes-agent/pulls/71/merge -X PUT --input=payload.json -f sha={'a' * 40} -f merge_method=squash",
    ],
)
def test_ambiguous_or_cwd_inferred_gh_targets_are_not_approvable(command):
    assert approval.is_protected_pr_merge_command(command) is True
    assert approval.detect_pr_merge_target(command, cwd="/tmp/repo") is None


def test_missing_correlation_identity_blocks_before_prompt(monkeypatch):
    empty_tokens = approval.set_current_observability_context(
        session_id="", turn_id="", tool_call_id=""
    )
    callback = Mock(return_value="once")
    try:
        monkeypatch.setattr(
            approval, "_resolve_pr_merge_target", Mock(return_value=_target())
        )
        result = approval.check_all_command_guards(
            _canonical_merge(),
            "local",
            cwd="/tmp/repo",
            approval_callback=callback,
        )
    finally:
        approval.reset_current_observability_context(empty_tokens)

    assert result["approved"] is False
    assert result["outcome"] == "correlation_unavailable"
    callback.assert_not_called()


def test_protected_gateway_approval_requires_exact_request_id():
    session = "protected-exact-request"
    approval.clear_session(session)
    entry = approval._ApprovalEntry(
        {
            "command": _canonical_merge(),
            "protected_once": True,
            "request_id": "request-exact",
        }
    )
    approval._gateway_queues[session] = [entry]

    assert approval.resolve_gateway_approval(session, "once") == 0
    assert entry.event.is_set() is False
    assert approval.resolve_gateway_approval(
        session, "once", request_id="request-exact"
    ) == 1
    assert entry.event.is_set() is True
    approval.clear_session(session)


def _terminal_config(tmp_path):
    return {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "local_persistent": False,
    }


def test_approved_protected_merge_is_never_retried(monkeypatch, tmp_path):
    env = MagicMock()
    env.execute.side_effect = RuntimeError("transient backend failure")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _terminal_config(tmp_path))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        Mock(
            return_value={
                "approved": True,
                "user_approved": True,
                "pr_merge_authorized": True,
                "description": "protected merge",
            }
        ),
    )
    monkeypatch.setattr(
        terminal_tool, "_check_live_state_db_guard", lambda **_kwargs: (False, None)
    )

    with patch("tools.process_registry._is_supervised_gateway_process", return_value=False), patch(
        "tools.terminal_tool.time.sleep"
    ):
        result = json.loads(
            terminal_tool.terminal_tool(_canonical_merge(), workdir=str(tmp_path))
        )

    assert result["exit_code"] == -1
    assert env.execute.call_count == 1


def test_protected_merge_cannot_start_in_background(monkeypatch, tmp_path):
    env = MagicMock()
    guard = Mock()
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _terminal_config(tmp_path))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(terminal_tool, "_check_all_guards", guard)
    monkeypatch.setattr(
        terminal_tool, "_check_live_state_db_guard", lambda **_kwargs: (False, None)
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            _canonical_merge(), workdir=str(tmp_path), background=True, pty=True
        )
    )

    assert result["status"] == "blocked"
    assert "foreground" in result["error"]
    guard.assert_not_called()
    env.execute.assert_not_called()


def test_receipt_fsyncs_containing_directory(monkeypatch):
    directory_synced = False
    real_fsync = os.fsync

    def tracking_fsync(fd):
        nonlocal directory_synced
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_synced = True
        return real_fsync(fd)

    monkeypatch.setattr(approval.os, "fsync", tracking_fsync)
    monkeypatch.setattr(
        approval, "_resolve_pr_merge_target", Mock(side_effect=[_target(), _target()])
    )
    result = approval.check_all_command_guards(
        _canonical_merge(),
        "local",
        cwd="/tmp/repo",
        approval_callback=Mock(return_value="once"),
    )

    assert result["approved"] is True
    assert directory_synced is True


def test_receipt_directory_fsync_failure_blocks_authorization(monkeypatch):
    real_fsync = os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(approval.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(
        approval, "_resolve_pr_merge_target", Mock(side_effect=[_target(), _target()])
    )
    result = approval.check_all_command_guards(
        _canonical_merge(),
        "local",
        cwd="/tmp/repo",
        approval_callback=Mock(return_value="once"),
    )

    assert result["approved"] is False
    assert result["outcome"] == "receipt_failed"
