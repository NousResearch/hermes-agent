"""Kanban workers cannot spend hosted Actions budget through approval bypasses (#98238)."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_cli.config as config
from tools import approval, terminal_tool
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.registry import registry


MUTATIONS = (
    "gh run rerun 123456",
    "gh run cancel 123456",
    "gh workflow run ci.yml --ref stable",
    "gh api -X POST repos/acme/widgets/actions/runs/123456/rerun",
    "curl -X POST https://api.github.com/repos/acme/widgets/actions/workflows/ci.yml/dispatches",
    "bash -lc 'gh run rerun 123456'",
    "gh api -X POST repos/acme/widgets/dispatches",
)
ENTRIES = ("dangerous", "combined", "terminal", "terminal-forced")


@pytest.fixture(params=(
    ("local", "off", False, "cli"),
    ("local", "manual", True, "telegram"),
    ("docker", "manual", False, "cron"),
    ("modal", "smart", True, "cli"),
))
def guard_context(request, tmp_path, monkeypatch):
    env_type, mode, yolo, platform = request.param
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"approvals:\n  mode: {mode}\nterminal:\n  env: {env_type}\n"
        f"  cwd: {tmp_path}\n  timeout: 10\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", env_type)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "budget-guard-task")
    config._LOAD_CONFIG_CACHE.clear()
    session_key = f"kanban-actions-{tmp_path.name}"
    token = set_current_session_key(session_key)
    if yolo:
        approval.enable_session_yolo(session_key)
    execute = Mock(return_value={"output": "headless result", "returncode": 0})
    env = SimpleNamespace(cwd=str(tmp_path), host_cwd=None, execute=execute, cleanup=Mock())
    # The backend boundary never creates a container or executes a shell command.
    monkeypatch.setattr(terminal_tool, "_create_configured_env", lambda *_a, **_kw: env)
    yield env_type, session_key, execute
    terminal_tool.cleanup_all_environments()
    approval.clear_session(session_key)
    reset_current_session_key(token)
    config._LOAD_CONFIG_CACHE.clear()


def _invoke(entry, command, guard_context):
    env_type, session_key, execute = guard_context
    if entry == "terminal-forced":
        return json.loads(terminal_tool.terminal_tool(command, task_id=session_key, force=True))
    if entry == "terminal":
        return json.loads(registry.get_entry("terminal").handler({"command": command}, task_id=session_key))
    check = {"dangerous": approval.check_dangerous_command, "combined": approval.check_all_command_guards}[entry]
    return check(command, env_type)


@pytest.mark.parametrize("command", MUTATIONS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_worker_mutations_stop_before_approval_or_terminal_execution(command, entry, guard_context):
    result = _invoke(entry, command, guard_context)
    if entry.startswith("terminal"):
        assert result.get("status") == "blocked", result
        assert "operator-owned Actions budget" in result["error"]
    else:
        assert result["approved"] is False, result
        assert result["kanban_policy"] == "github_actions_mutation"
    guard_context[2].assert_not_called()


@pytest.mark.parametrize("command,worker", (
    ("gh run list --limit 5", True),
    ("gh run view 123456 --json status,conclusion", True),
    ("gh pr checks 123", True),
    ("gh workflow run ci.yml --ref stable", False),
))
@pytest.mark.parametrize("entry", ENTRIES)
def test_readonly_worker_and_ordinary_operator_paths_keep_their_authority(
        command, worker, entry, guard_context, monkeypatch):
    if not worker:
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    result = _invoke(entry, command, guard_context)
    if entry.startswith("terminal"):
        assert result.get("exit_code") == 0, result
        guard_context[2].assert_called_once()
        assert guard_context[2].call_args.args[0] == command
    else:
        assert result["approved"] is True, result
        guard_context[2].assert_not_called()
