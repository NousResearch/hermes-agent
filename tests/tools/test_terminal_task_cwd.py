"""Regression tests for task/session cwd propagation in terminal_tool."""

import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool


def _minimal_terminal_config(cwd="/default"):
    return {
        "env_type": "local",
        "cwd": cwd,
        "timeout": 60,
        "lifetime_seconds": 3600,
    }


def test_foreground_command_uses_registered_task_cwd_for_existing_environment(monkeypatch):
    """ACP can update task cwd after the local env exists; foreground must honor it."""
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/acp"})]


def test_explicit_workdir_still_wins_over_registered_task_cwd(monkeypatch):
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append(kwargs)
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="pwd",
            task_id=task_id,
            workdir="/explicit/workdir",
        )
    )

    assert result["exit_code"] == 0
    assert calls == [{"timeout": 60, "cwd": "/explicit/workdir"}]


def test_foreground_command_prefers_live_env_cwd_over_init_time_cwd(monkeypatch):
    """A prior `cd` updates env.cwd; terminal_tool must honor that live cwd."""
    calls = []

    class FakeEnv:
        env = {}
        cwd = "/workspace/live"

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    task_id = "session-live-cwd"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/init"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd="/workspace/init"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/live"})]


def test_terminal_endpoint_blocks_relative_write_from_live_cwd(tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    live_cwd = tmp_path / "outside"
    safe_root.mkdir()
    live_cwd.mkdir()
    calls = []

    class FakeEnv:
        env = {}
        cwd = str(live_cwd)

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "unexpected", "returncode": 0}

    task_id = "session-live-cwd-write"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": str(live_cwd)}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd=str(safe_root)))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(
        terminal_tool.terminal_tool(command="echo data > relative.txt", task_id=task_id)
    )

    assert result["status"] == "blocked"
    assert "Blocked" in result["error"]
    assert calls == []


def test_background_command_prefers_live_env_cwd_over_init_time_cwd(monkeypatch):
    """Background process launches must also use the live session cwd."""

    class FakeEnv:
        env = {}
        cwd = "/workspace/live"

    class FakeRegistry:
        def __init__(self):
            self.calls = []
            self.pending_watchers = []

        def spawn_local(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(id="proc_test", pid=1234)

    import tools.process_registry as process_registry_mod

    registry = FakeRegistry()
    task_id = "session-live-cwd-bg"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/init"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd="/workspace/init"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_mod, "process_registry", registry)

    result = json.loads(
        terminal_tool.terminal_tool(
            command="sleep 1",
            task_id=task_id,
            background=True,
        )
    )

    assert result["exit_code"] == 0
    # session_key falls back to the raw task_id when no gateway contextvar is set
    # (it doesn't propagate to tool-worker threads), so process.kill / stop can
    # still find and terminate this background process.
    assert registry.calls == [{
        "command": "sleep 1",
        "cwd": "/workspace/live",
        "task_id": task_id,
        "session_key": task_id,
        "env_vars": {},
        "use_pty": False,
    }]


def test_registering_cwd_override_updates_live_env_cwd(monkeypatch):
    """An ACP ``update_cwd`` (re-)registered mid-session must win over a
    previously ``cd``-ed live ``env.cwd``.

    Preferring live ``env.cwd`` (so session-local ``cd`` survives) means a
    freshly registered ``cwd`` override would otherwise sit *below* the
    already-set ``env.cwd`` and be silently ignored. ``register_task_env_overrides``
    syncs the new cwd onto the live cached env so an explicit ACP project-root
    change takes effect, as the editor client expects.
    """

    class FakeEnv:
        env = {}
        cwd = "/workspace/old"

    task_id = "acp-session-update"
    fake_env = FakeEnv()
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: fake_env})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    terminal_tool.register_task_env_overrides(task_id, {"cwd": "/workspace/new"})

    # The live env now reflects the editor's new project root.
    assert fake_env.cwd == "/workspace/new"

    # A subsequent command resolves to the new cwd (env.cwd precedence).
    assert terminal_tool._resolve_command_cwd(
        workdir=None, env=fake_env, default_cwd="/workspace/config"
    ) == "/workspace/new"


def test_registering_cwd_override_noop_when_no_live_env(monkeypatch):
    """Registering an override before the env exists must not crash; the cwd
    is applied at env creation time instead."""
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    # Should not raise even though no env is cached yet.
    terminal_tool.register_task_env_overrides("acp-session-pending", {"cwd": "/workspace/new"})

    assert terminal_tool._task_env_overrides["acp-session-pending"] == {"cwd": "/workspace/new"}


def test_registering_non_cwd_override_leaves_live_env_cwd_untouched(monkeypatch):
    """A non-cwd override (e.g. a per-task Modal image) must not disturb the
    live env's cwd."""

    class FakeEnv:
        env = {}
        cwd = "/workspace/keep"

    task_id = "rl-rollout-1"
    fake_env = FakeEnv()
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: fake_env})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    terminal_tool.register_task_env_overrides(task_id, {"modal_image": "custom:latest"})

    assert fake_env.cwd == "/workspace/keep"


def test_safe_getcwd_returns_real_cwd(monkeypatch):
    monkeypatch.setattr(terminal_tool.os, "getcwd", lambda: "/home/user/project")
    assert terminal_tool._safe_getcwd() == "/home/user/project"


def test_safe_getcwd_falls_back_to_terminal_cwd_when_cwd_deleted(monkeypatch):
    def _boom():
        raise FileNotFoundError("[Errno 2] No such file or directory")

    monkeypatch.setattr(terminal_tool.os, "getcwd", _boom)
    monkeypatch.setenv("TERMINAL_CWD", "/srv/work")
    assert terminal_tool._safe_getcwd() == "/srv/work"


def test_safe_getcwd_falls_back_to_home_when_no_terminal_cwd(monkeypatch):
    def _boom():
        raise FileNotFoundError()

    monkeypatch.setattr(terminal_tool.os, "getcwd", _boom)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool.os.path, "expanduser", lambda p: "/home/me")
    assert terminal_tool._safe_getcwd() == "/home/me"
