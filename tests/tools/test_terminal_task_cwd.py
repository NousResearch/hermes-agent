"""Regression tests for task/session cwd propagation in terminal_tool."""

import json
from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
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
    guard_calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append(kwargs)
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    def approve(command, env_type, **kwargs):
        guard_calls.append((command, env_type, kwargs))
        return {"approved": True}

    monkeypatch.setattr(terminal_tool, "_check_all_guards", approve)

    result = json.loads(
        terminal_tool.terminal_tool(
            command="pwd",
            task_id=task_id,
            workdir="/explicit/workdir",
        )
    )

    assert result["exit_code"] == 0
    assert calls == [{"timeout": 60, "cwd": "/explicit/workdir", "bounded_capture": True}]
    assert len(guard_calls) == 1
    command, env_type, guard_kwargs = guard_calls[0]
    assert command == "pwd" and env_type == "local"
    assert guard_kwargs["has_host_access"] is False
    assert guard_kwargs["workdir"] == "/explicit/workdir"
    assert guard_kwargs["execution_context"]["background"] is False
    assert guard_kwargs["execution_context"]["pty"] is False
    assert guard_kwargs["execution_context"]["timeout"] == 60


def test_foreground_command_prefers_recorded_session_cwd_over_init_time_cwd(monkeypatch):
    """A prior `cd` records the session cwd; terminal_tool must honor it."""
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
    # The prior command's completed `cd` recorded the session cwd.
    terminal_tool.record_session_cwd(task_id, "/workspace/live")

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/live", "bounded_capture": True})]


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
        "session_key": task_id,
        "env_vars": {},
        "use_pty": False,
    }]


def test_registering_cwd_override_updates_session_record(monkeypatch):
    """An ACP ``update_cwd`` (re-)registered mid-session must win over a
    previously ``cd``-ed session cwd.

    Registration writes the session record directly, so an explicit ACP
    project-root change takes effect on the next command, as the editor
    client expects.
    """

    class FakeEnv:
        env = {}
        cwd = "/workspace/old"

    task_id = "acp-session-update"
    fake_env = FakeEnv()
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: fake_env})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    # The session had cd'd somewhere before the editor switched project roots.
    terminal_tool.record_session_cwd(task_id, "/workspace/old")

    terminal_tool.register_task_env_overrides(task_id, {"cwd": "/workspace/new"})

    # The live env mirror still updates (legacy env seeding) …
    assert fake_env.cwd == "/workspace/new"
    # … and the session record — what commands actually resolve against — too.
    assert terminal_tool.get_session_cwd(task_id) == "/workspace/new"
    assert terminal_tool._resolve_command_cwd(
        workdir=None, default_cwd="/workspace/config", session_key=task_id
    ) == "/workspace/new"


def test_registering_cwd_override_noop_when_no_live_env(monkeypatch):
    """Registering an override before the env exists must not crash; the cwd
    is applied at env creation time instead."""
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})

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
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})

    terminal_tool.register_task_env_overrides(task_id, {"modal_image": "custom:latest"})

    assert fake_env.cwd == "/workspace/keep"


def test_stale_env_cwd_from_different_session_is_ignored(monkeypatch):
    """A different session's ``cd`` left the shared env in its checkout.

    Session cwd records are keyed by raw session id, so session B must ignore
    the shared environment's mutable ``cwd`` and fall through to its own
    config/override cwd.
    """
    calls = []

    class FakeEnv:
        env = {}
        cwd = "/home/user/src/hermes-desktop-tipc/apps/desktop"

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    task_id = "session-B"
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd="/home/user/src/hermes-agent"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    # The command must run in the config cwd (hermes-agent), NOT the stale
    # env.cwd left by session A (hermes-desktop-tipc).
    assert calls == [("pwd", {"timeout": 60, "cwd": "/home/user/src/hermes-agent", "bounded_capture": True})]


def test_same_session_recorded_cwd_survives_across_commands(monkeypatch):
    """In-session `cd` state survives: the record written by one command is
    used by the next command in the same session."""
    calls = []

    class FakeEnv:
        env = {}
        cwd = "/workspace/deep"

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    env = FakeEnv()
    task_id = "session-X"
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd="/workspace/config"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    # First command runs in the config cwd (no record yet) and afterwards
    # mirrors the env's post-command cwd into the session record.
    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))
    assert result["exit_code"] == 0
    assert calls[0] == ("pwd", {"timeout": 60, "cwd": "/workspace/config", "bounded_capture": True})
    assert terminal_tool.get_session_cwd(task_id) == "/workspace/deep"

    # Second command in the same session trusts the record.
    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))
    assert result["exit_code"] == 0
    assert calls[1] == ("pwd", {"timeout": 60, "cwd": "/workspace/deep", "bounded_capture": True})


def test_shared_docker_cwd_overrides_reuse_stable_runtime(monkeypatch):
    """Two cwd-only sessions must not replace the shared default container."""
    created = []
    executed = []
    approved_workdirs = []

    config = {
        "env_type": "docker",
        "docker_image": "python:3.11",
        "cwd": "/root",
        "host_cwd": None,
        "timeout": 60,
        "lifetime_seconds": 3600,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        "docker_env": {},
        "docker_forward_env": [],
        "docker_extra_args": [],
        "docker_mount_cwd_to_workspace": False,
        "docker_run_as_host_user": False,
        "docker_network": True,
        "docker_persist_across_processes": True,
    }

    class FakeEnv:
        runtime_fingerprint = "stable-default-runtime"
        init_env_digest = "sha256:test"
        has_host_access = False

        def __init__(self, cwd):
            self.cwd = cwd

        def execute(self, command, **kwargs):
            executed.append((command, kwargs["cwd"]))
            return {"output": "ok", "returncode": 0}

    def create_environment(*, cwd, **kwargs):
        created.append(cwd)
        return FakeEnv(cwd)

    def approve(command, env_type, **kwargs):
        approved_workdirs.append(kwargs["workdir"])
        return {"approved": True}

    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {
            "session-a": {"cwd": "/workspace/a"},
            "session-b": {"cwd": "/workspace/b"},
        },
    )
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", approve)
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_docker_runtime_identity",
        lambda **kwargs: (
            {},
            "sha256:test",
            "stable-default-runtime",
            [],
        ),
    )

    first = json.loads(
        terminal_tool.terminal_tool(command="pwd", task_id="session-a")
    )
    second = json.loads(
        terminal_tool.terminal_tool(command="pwd", task_id="session-b")
    )

    assert first["exit_code"] == second["exit_code"] == 0
    assert created == ["/root"]
    assert approved_workdirs == ["/workspace/a", "/workspace/b"]
    assert executed == [
        ("pwd", "/workspace/a"),
        ("pwd", "/workspace/b"),
    ]


@pytest.mark.parametrize("backend", ["singularity", "modal", "daytona"])
def test_recorded_cwd_does_not_replace_shared_container_runtime(
    monkeypatch,
    backend,
):
    """A session `cd` is command state, not immutable sandbox identity."""
    task_id = f"{backend}-session"
    image = f"example/{backend}:latest"
    config = {
        "env_type": backend,
        "cwd": "/root",
        "timeout": 60,
        "lifetime_seconds": 3600,
        "singularity_image": image,
        "modal_image": image,
        "daytona_image": image,
        "modal_mode": "direct",
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
    }
    executed = []
    approved_workdirs = []
    monkeypatch.setattr(
        terminal_tool,
        "_get_modal_backend_state",
        lambda _mode: {"selected_backend": "direct"},
    )

    class FakeContainer:
        env = {}
        cwd = "/root"

        def execute(self, command, **kwargs):
            executed.append((command, kwargs))
            self.cwd = kwargs["cwd"]
            return {"output": "ok", "returncode": 0}

    env_class = type(f"{backend.title()}Environment", (FakeContainer,), {})
    env = env_class()
    env._hermes_runtime_identity = (
        terminal_tool._requested_environment_runtime_identity(
            config=config,
            image=image,
            cwd="/root",
            task_id="default",
        )
    )

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {task_id: {"cwd": "/workspace/declared"}},
    )
    monkeypatch.setattr(terminal_tool, "_retired_environments", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_create_environment",
        lambda **_kwargs: pytest.fail("recorded cwd replaced the cached runtime"),
    )

    def approve(_command, _env_type, **kwargs):
        approved_workdirs.append(kwargs["workdir"])
        return {"approved": True}

    monkeypatch.setattr(terminal_tool, "_check_all_guards", approve)
    terminal_tool.record_session_cwd(task_id, "/workspace/after-cd")

    result = json.loads(
        terminal_tool.terminal_tool(command="pwd", task_id=task_id)
    )

    assert result["exit_code"] == 0
    assert terminal_tool._active_environments["default"] is env
    assert approved_workdirs == ["/workspace/after-cd"]
    assert executed == [
        (
            "pwd",
            {
                "timeout": 60,
                "cwd": "/workspace/after-cd",
                "bounded_capture": True,
            },
        )
    ]


def test_cached_environment_must_match_backend_and_docker_fingerprint():
    class DockerEnvironment:
        runtime_fingerprint = "v1-current"

    env = DockerEnvironment()

    assert terminal_tool._environment_matches_runtime(
        env, "docker", "v1-current"
    )
    assert not terminal_tool._environment_matches_runtime(
        env, "docker", "v1-changed"
    )
    assert not terminal_tool._environment_matches_runtime(env, "local")


def test_cached_ssh_target_change_recreates_before_approval_and_execution(
    monkeypatch,
):
    config = {
        "env_type": "ssh",
        "cwd": "~",
        "timeout": 60,
        "lifetime_seconds": 3600,
        "ssh_host": "host-b.example",
        "ssh_user": "deploy-b",
        "ssh_port": 2202,
        "ssh_key": "/keys/b",
        "ssh_persistent": True,
        "local_persistent": False,
    }
    old_calls = []
    new_calls = []
    approvals = []
    created = []

    class SSHEnvironment:
        def __init__(self, host, user, port, key_path, calls):
            self.host = host
            self.user = user
            self.port = port
            self.key_path = key_path
            self.cwd = "~"
            self._persistent = True
            self.calls = calls

        def execute(self, command, **kwargs):
            self.calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

        def cleanup(self):
            pass

    old_env = SSHEnvironment(
        "host-a.example",
        "deploy-a",
        22,
        "/keys/a",
        old_calls,
    )

    def create_environment(**kwargs):
        created.append(kwargs)
        ssh = kwargs["ssh_config"]
        return SSHEnvironment(
            ssh["host"],
            ssh["user"],
            ssh["port"],
            ssh["key"],
            new_calls,
        )

    def approve(_command, _env_type, **kwargs):
        approvals.append(kwargs["execution_context"])
        return {"approved": True}

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": old_env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 1.0})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_retired_environments", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", approve)

    result = json.loads(terminal_tool.terminal_tool(command="pwd"))

    assert result["exit_code"] == 0
    assert old_calls == []
    assert len(new_calls) == 1
    assert created[0]["ssh_config"] == {
        "host": "host-b.example",
        "user": "deploy-b",
        "port": 2202,
        "key": "/keys/b",
        "persistent": True,
    }
    assert approvals[0]["target"] == {
        "host": "host-b.example",
        "user": "deploy-b",
        "port": 2202,
        "key_path": "/keys/b",
        "persistent": True,
    }
    assert any(item[1] is old_env for item in terminal_tool._retired_environments.values())


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
