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
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/acp", "bounded_capture": True})]


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
    assert calls == [{"timeout": 60, "cwd": "/explicit/workdir", "bounded_capture": True}]


def test_explicit_workdir_does_not_persist_into_session_cwd(monkeypatch):
    """A per-command ``workdir`` must not hijack the durable session cwd.

    Regression: the post-command dual-write recorded ``env.cwd`` (stamped to
    the transient ``workdir``) into the session-cwd store, so every later
    command that omitted ``workdir`` inherited the one-off directory.
    """
    recorded = []

    class FakeEnv:
        env = {}
        cwd = "/workspace/acp"

        def execute(self, command, **kwargs):
            # Marker parse stamps env.cwd to where the command ran.
            self.cwd = kwargs.get("cwd", self.cwd)
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-2"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        terminal_tool,
        "record_session_cwd",
        lambda session_key, cwd: recorded.append((session_key, cwd)),
    )

    terminal_tool.terminal_tool(command="pwd", task_id=task_id, workdir="/one/off/dir")

    # The transient workdir must NOT have been recorded as the session cwd.
    assert all(cwd != "/one/off/dir" for _, cwd in recorded), recorded


def test_background_command_prefers_recorded_session_cwd_over_init_time_cwd(monkeypatch):
    """Background process launches must also use the recorded session cwd."""

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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    terminal_tool.record_session_cwd(task_id, "/workspace/live")

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
        "owner_task_id": task_id,
        "session_key": task_id,
        "env_vars": {},
        "use_pty": False,
    }]


def test_host_local_background_command_bypasses_configured_backend(tmp_path, monkeypatch):
    """Hermes control-plane children stay on the host when tools use Docker."""
    calls = []

    class FakeEnv:
        env = {}
        cwd = str(tmp_path)

    class FakeRegistry:
        pending_watchers = []

        def spawn_local(self, **kwargs):
            calls.append(("local", kwargs))
            return SimpleNamespace(id="proc_host", pid=1234)

        def spawn_via_env(self, **kwargs):
            calls.append(("configured", kwargs))
            raise AssertionError("host-local command reached configured backend")

    import tools.process_registry as process_registry_mod
    import tools.self_repo_guard as self_repo_guard

    task_id = "bot-delivery"
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "docker_image": "python:3.11",
            "cwd": str(tmp_path),
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {f"host-local-{task_id}": FakeEnv()},
    )
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value)
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda config: False)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": env_type == "local"},
    )
    monkeypatch.setattr(self_repo_guard, "guard_active", lambda: False)
    monkeypatch.setattr(process_registry_mod, "process_registry", FakeRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            command="host-runner",
            task_id=task_id,
            workdir=str(tmp_path),
            background=True,
            _host_local=True,
        )
    )

    assert result["session_id"] == "proc_host"
    assert calls[0][0] == "local"
    assert calls[0][1]["task_id"] == f"host-local-{task_id}"


def test_concurrent_commands_record_their_own_observed_cwd(monkeypatch, tmp_path):
    """A shared env's mutable cwd must not cross-write concurrent sessions."""
    import threading

    cwd_a = tmp_path / "a"
    cwd_b = tmp_path / "b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    a_set = threading.Event()
    b_set = threading.Event()

    class SharedEnv:
        env = {}
        cwd = str(tmp_path)

        def execute(self, command, **kwargs):
            observed = str(cwd_a if command == "session-a" else cwd_b)
            if command == "session-a":
                self.cwd = observed
                a_set.set()
                assert b_set.wait(timeout=5)
            else:
                assert a_set.wait(timeout=5)
                self.cwd = observed
                b_set.set()
            return {
                "output": "ok",
                "returncode": 0,
                "cwd_observed": True,
                "cwd": observed,
            }

    shared = SharedEnv()
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": shared})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _value: "default")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    terminal_tool.record_session_cwd("task-a", str(cwd_a))
    terminal_tool.record_session_cwd("task-b", str(cwd_b))

    results = {}

    def run(name, task_id):
        results[task_id] = json.loads(
            terminal_tool.terminal_tool(command=name, task_id=task_id)
        )

    threads = [
        threading.Thread(target=run, args=("session-a", "task-a")),
        threading.Thread(target=run, args=("session-b", "task-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert results["task-a"]["exit_code"] == 0
    assert results["task-b"]["exit_code"] == 0
    assert terminal_tool.get_session_cwd("task-a") == str(cwd_a)
    assert terminal_tool.get_session_cwd("task-b") == str(cwd_b)


def test_safe_getcwd_falls_back_to_home_when_no_terminal_cwd(monkeypatch):
    def _boom():
        raise FileNotFoundError()

    monkeypatch.setattr(terminal_tool.os, "getcwd", _boom)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool.os.path, "expanduser", lambda p: "/home/me")
    assert terminal_tool._safe_getcwd() == "/home/me"
