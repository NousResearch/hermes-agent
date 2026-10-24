"""Regression tests for #140395: a subagent child's terminal cwd must stay on its
own task_id record even when the approval contextvar — the parent's session key —
is what the executing daemon worker sees (``DaemonThreadPoolExecutor.submit``
copies the caller's context, so ``get_current_session_key`` returns the parent's
key inside child tool calls). A child ``cd`` must not rewrite the parent record,
and the child's commands must resolve against the child record first."""

import json

from tools import terminal_tool
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.terminal_tool import get_session_cwd, record_session_cwd


def _minimal_terminal_config(cwd="/default"):
    return {"env_type": "local", "cwd": cwd, "timeout": 60, "lifetime_seconds": 3600}


class _ObservingEnv:
    """Fake env capturing the cwd each command ran in; reports *observed* as the
    directory the command ended in (what ``cwd_observed`` markers do for real)."""

    env: dict = {}

    def __init__(self, observed=None):
        self.executed_cwds = []
        self.observed = observed

    def execute(self, command, **kwargs):
        self.executed_cwds.append(kwargs.get("cwd"))
        observed = self.observed if self.observed is not None else kwargs.get("cwd")
        return {"output": "ok", "returncode": 0, "cwd": observed, "cwd_observed": True}


def _install(monkeypatch, env):
    monkeypatch.setattr(terminal_tool, "_acquire_env", lambda plan, task_id: env)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_container_aliases", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool, "_get_env_config", lambda: _minimal_terminal_config()
    )


def test_child_cd_does_not_rewrite_parent_record(monkeypatch):
    """A child's observed cwd lands on its own task_id record, not the parent's.
    The container alias is what seed_workspace registers for every real child
    (child→parent), so an alias-registered task_id is never treated as a
    top-level turn whose cd should mirror onto the caller's session key."""
    env = _ObservingEnv(observed="/child/dir")
    child, parent = "child-task-1", "parent-session-1"
    _install(monkeypatch, env)
    record_session_cwd(parent, "/parent/start")
    terminal_tool.register_container_alias(child, parent)

    token = set_current_session_key(parent)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(command="cd /child/dir && pwd", task_id=child)
        )
    finally:
        reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert get_session_cwd(child) == "/child/dir"
    assert get_session_cwd(parent) == "/parent/start"


def test_child_command_resolves_own_record_over_parent_context(monkeypatch):
    """With both records present, the child's command runs in the child's directory."""
    env = _ObservingEnv()
    child, parent = "child-task-2", "parent-session-2"
    _install(monkeypatch, env)
    record_session_cwd(parent, "/parent/dir")
    record_session_cwd(child, "/child/dir")

    token = set_current_session_key(parent)
    try:
        result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=child))
    finally:
        reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert env.executed_cwds == ["/child/dir"]


def test_top_level_call_without_task_id_keeps_session_key_anchor(monkeypatch):
    """No task_id (top-level): the cwd record still lands on the session key."""
    env = _ObservingEnv(observed="/top/dir")
    _install(monkeypatch, env)

    token = set_current_session_key("session-anchor-3")
    try:
        result = json.loads(terminal_tool.terminal_tool(command="cd /top/dir && pwd"))
    finally:
        reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert get_session_cwd("session-anchor-3") == "/top/dir"


def test_top_level_cd_mirrors_onto_bound_gateway_session_key(monkeypatch):
    """A top-level turn under a bound gateway/TUI session (contextvar session key
    differs from the turn task_id) still tracks its cd on BOTH keys: the task_id
    record (what seed_workspace and the turn-workspace binding read) and the
    session-key record (what session-keyed readers — the /init ladder
    ``_resolve_session_cwd``, gateway ``run_inbound`` / TUI ``methods_tools``
    dispatch — read via HERMES_SESSION_KEY). Without the mirror, /init would
    stop following a top-level cd once the anchor moved to the task_id."""
    env = _ObservingEnv(observed="/moved/dir")
    task, gateway_key = "turn-task-4", "agent:main:conv-4"
    _install(monkeypatch, env)
    record_session_cwd(gateway_key, "/original/start")

    token = set_current_session_key(gateway_key)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(command="cd /moved/dir && pwd", task_id=task)
        )
    finally:
        reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert get_session_cwd(task) == "/moved/dir"
    assert get_session_cwd(gateway_key) == "/moved/dir"


def test_child_background_command_resolves_own_cwd_record(monkeypatch):
    """A child's background spawn resolves its start directory from the child's
    own record, not the parent session-keyed one the contextvar carries."""
    from types import SimpleNamespace

    import tools.process_registry as process_registry_mod

    class FakeRegistry:
        def __init__(self):
            self.calls = []

        def spawn_local(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(id="proc_test", pid=1234)

    registry = FakeRegistry()
    child, parent = "child-task-5", "parent-session-5"
    _install(monkeypatch, _ObservingEnv())
    record_session_cwd(parent, "/parent/dir")
    record_session_cwd(child, "/child/dir")
    terminal_tool.register_container_alias(child, parent)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_mod, "process_registry", registry)

    token = set_current_session_key(parent)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="sleep 1", task_id=child, background=True
            )
        )
    finally:
        reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert len(registry.calls) == 1
    assert registry.calls[0]["cwd"] == "/child/dir"
