"""Tests for the WorkerRuntime protocol + builtin runtimes (D1 Task 1)."""
from __future__ import annotations

import os
import signal
import time
from unittest.mock import MagicMock

import pytest

from tools.environments.kanban_spawn import (
    LocalRuntime,
    WorkerRuntime,
    load_runtime,
    register_runtime,
)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_load_runtime_default_returns_local():
    rt = load_runtime("local", {})
    assert isinstance(rt, LocalRuntime)
    assert rt.name == "local"


def test_load_runtime_unknown_raises():
    with pytest.raises(ValueError, match="unknown worker_runtime"):
        load_runtime("does-not-exist", {})


def test_load_runtime_passes_cfg():
    """Config dict reaches the runtime constructor."""
    rt = load_runtime("local", {"future_key": "future_value"})
    # LocalRuntime stashes config for future use; just verify no crash
    assert rt._cfg == {"future_key": "future_value"}


def test_load_runtime_none_cfg_is_safe():
    rt = load_runtime("local", None)
    assert rt._cfg == {}


def test_register_runtime_blocks_duplicates():
    """Re-registering a name is a bug; raise loudly."""
    class _Stub:
        name = "stub-test-dup"
        def __init__(self, cfg): pass
        def spawn(self, *a, **kw): return None
        def is_alive(self, h): return False
        def terminate(self, h, reason=""): pass

    register_runtime("__test_register_dup__", _Stub)
    with pytest.raises(ValueError, match="already registered"):
        register_runtime("__test_register_dup__", _Stub)
    # Cleanup
    from tools.environments.kanban_spawn import _RUNTIMES
    _RUNTIMES.pop("__test_register_dup__", None)


# ---------------------------------------------------------------------------
# LocalRuntime delegation
# ---------------------------------------------------------------------------

def test_local_runtime_delegates_to_default_spawn(monkeypatch):
    """LocalRuntime.spawn must call kanban_db._default_spawn unchanged.

    This is the regression-safety guarantee — `worker_runtime: local` is
    byte-identical to pre-D1 behavior.
    """
    rt = LocalRuntime({})

    fake_pid = 4242
    captured = {}

    def fake_default_spawn(task, workspace, *, board=None):
        captured["task"] = task
        captured["workspace"] = workspace
        captured["board"] = board
        return fake_pid

    monkeypatch.setattr(
        "hermes_cli.kanban_db._default_spawn", fake_default_spawn
    )

    fake_task = MagicMock()
    fake_task.id = "t_test001"
    pid = rt.spawn(fake_task, workspace="/tmp/ws", board="main")

    assert pid == fake_pid
    assert captured["task"] is fake_task
    assert captured["workspace"] == "/tmp/ws"
    assert captured["board"] == "main"


def test_local_runtime_passes_through_none_board(monkeypatch):
    """When board=None is passed, default_spawn receives board=None."""
    rt = LocalRuntime({})
    captured = {}

    def fake_default_spawn(task, workspace, *, board=None):
        captured["board"] = board
        return 1

    monkeypatch.setattr(
        "hermes_cli.kanban_db._default_spawn", fake_default_spawn
    )
    rt.spawn(MagicMock(id="t_x"), workspace="/tmp")
    assert captured["board"] is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_worker_runtime_protocol_methods(monkeypatch):
    """LocalRuntime exposes the WorkerRuntime contract."""
    rt = LocalRuntime({})
    assert hasattr(rt, "name")
    assert hasattr(rt, "spawn")
    assert hasattr(rt, "is_alive")
    assert hasattr(rt, "terminate")
    # is_alive on an obviously-dead PID returns False without raising.
    # Mock os.kill to avoid the test-suite's live-system guard on
    # out-of-subtree PIDs.
    monkeypatch.setattr(os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    assert rt.is_alive(handle=999999999) is False


def test_local_runtime_is_alive_handles_bad_input():
    """is_alive is robust against non-integer / negative input."""
    rt = LocalRuntime({})
    assert rt.is_alive(0) is False
    assert rt.is_alive(-1) is False
    assert rt.is_alive("not-a-pid") is False
    assert rt.is_alive(None) is False  # type: ignore[arg-type]


def test_local_runtime_is_alive_returns_true_for_self():
    """The current Python process is alive."""
    rt = LocalRuntime({})
    assert rt.is_alive(os.getpid()) is True


def test_local_runtime_terminate_handles_dead_pid_silently(monkeypatch):
    """terminate() on an already-dead PID does not raise.

    Mocks os.kill to raise ProcessLookupError, which terminate() must swallow.
    Test-suite's live-system guard would otherwise block os.kill on out-of-subtree PIDs.
    """
    rt = LocalRuntime({})

    def _raise_dead(pid, sig):
        raise ProcessLookupError()
    monkeypatch.setattr(os, "kill", _raise_dead)
    rt.terminate(999999999, reason="test")
    # No exception raised → pass.


def test_local_runtime_terminate_handles_bad_input():
    """terminate() on non-integer / negative input is a no-op."""
    rt = LocalRuntime({})
    rt.terminate(0, reason="zero")
    rt.terminate(-1, reason="neg")
    rt.terminate("not-a-pid", reason="str")
    rt.terminate(None, reason="none")  # type: ignore[arg-type]
    # No exceptions → pass.


def test_local_runtime_terminate_actually_signals(monkeypatch):
    """terminate() calls os.kill(pid, SIGTERM) for a valid PID."""
    rt = LocalRuntime({})
    captured = {}

    def fake_kill(pid, sig):
        captured["pid"] = pid
        captured["sig"] = sig

    monkeypatch.setattr(os, "kill", fake_kill)
    rt.terminate(12345, reason="test")
    assert captured["pid"] == 12345
    assert captured["sig"] == signal.SIGTERM


# ---------------------------------------------------------------------------
# DockerRuntime tests (D1 Task 3)
# ---------------------------------------------------------------------------

def _make_fake_task(task_id="t_d1", assignee="researcher", skills=None):
    fake = MagicMock()
    fake.id = task_id
    fake.assignee = assignee
    fake.tenant = None
    fake.current_run_id = 5
    fake.claim_lock = "lock-xyz"
    fake.skills = skills
    return fake


def _stub_kanban_db_helpers(monkeypatch):
    """Stub kanban_db helpers so tests don't touch the real DB or PATH."""
    import hermes_cli.kanban_db as kb
    import pathlib
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: "/db")
    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: "/ws")
    monkeypatch.setattr(kb, "get_current_board", lambda: "main")
    monkeypatch.setattr(kb, "_normalize_board_slug", lambda s: s)
    monkeypatch.setattr(kb, "_resolve_assignee_to_profile", lambda a: "worker")


def _stub_docker_present(monkeypatch):
    """Make shutil.which('docker') return a path so DockerRuntime() boots."""
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda n: "/usr/bin/docker" if n == "docker" else None)


def test_docker_runtime_requires_default_image(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    with pytest.raises(ValueError, match="image_per_profile.default"):
        DockerRuntime({"image_per_profile": {}})


def test_docker_runtime_requires_docker_cli(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="docker.*CLI not found"):
        DockerRuntime({"image_per_profile": {"default": "img:latest"}})


def test_docker_runtime_resolve_image_falls_back_to_default(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "hermes-worker:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })
    assert rt._resolve_image("never-defined") == "hermes-worker:latest"


def test_docker_runtime_resolve_image_uses_per_profile_when_set(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {
            "default": "hermes-worker:latest",
            "ops": "hermes-ops:latest",
        },
        "bind_mounts": [], "env_passthrough": [],
    })
    assert rt._resolve_image("ops") == "hermes-ops:latest"
    assert rt._resolve_image("anything-else") == "hermes-worker:latest"


def test_docker_runtime_expand_bind_substitutes_hermes_home(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "hermes-worker:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })
    expanded = rt._expand_bind(
        "{hermes_home}/kanban.db:/hermes/kanban.db",
        "/home/u/.hermes",
    )
    assert expanded == "/home/u/.hermes/kanban.db:/hermes/kanban.db"


def test_docker_runtime_spawn_builds_correct_command(monkeypatch):
    """Full happy-path: spawn() produces the expected `docker run` argv."""
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    _stub_kanban_db_helpers(monkeypatch)
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-token")

    rt = DockerRuntime({
        "image_per_profile": {
            "worker": "hermes-worker:latest",
            "default": "hermes-worker:latest",
        },
        "bind_mounts": [
            "/home/u/.hermes/kanban.db:/hermes/kanban.db",
            "/home/u/.hermes/.env:/hermes/.env:ro",
        ],
        "env_passthrough": ["OPENROUTER_API_KEY"],
        "mem_limit": "4g",
        "cpus": "2.0",
        "network": "hermes-net",
        "auto_remove": True,
    })

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        out = MagicMock()
        out.stdout = b"abc123def456789\n"
        out.returncode = 0
        out.stderr = b""
        return out

    monkeypatch.setattr("subprocess.run", fake_run)

    handle = rt.spawn(
        _make_fake_task("t_d1", "researcher"),
        workspace="/tmp/work",
        board="main",
    )

    cmd = captured["cmd"]
    assert cmd[0] == "docker"
    assert cmd[1] == "run"
    assert "-d" in cmd
    assert "--rm" in cmd
    assert "--network" in cmd and "hermes-net" in cmd
    assert "--memory" in cmd and "4g" in cmd
    assert "--cpus" in cmd and "2.0" in cmd
    # Bind mounts present
    assert cmd.count("-v") == 2
    # Env passthrough captured
    env_idxs = [i for i, x in enumerate(cmd) if x == "-e"]
    env_pairs = [cmd[i + 1] for i in env_idxs]
    assert any(e.startswith("OPENROUTER_API_KEY=sk-test-token") for e in env_pairs)
    assert any(e.startswith("HERMES_KANBAN_TASK=t_d1") for e in env_pairs)
    # Image present
    assert "hermes-worker:latest" in cmd
    # In-container hermes invocation
    h_idx = cmd.index("hermes-worker:latest")
    in_cmd = cmd[h_idx + 1:]
    assert in_cmd[:2] == ["hermes", "-p"]
    # Per swarm-as-persona, assignee 'researcher' resolved to 'worker'
    assert in_cmd[2] == "worker"
    assert "chat" in in_cmd and "-q" in in_cmd
    # Handle returned is the truncated container id (12 chars)
    assert handle == "abc123def456"


def test_docker_runtime_spawn_raises_on_docker_failure(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    _stub_kanban_db_helpers(monkeypatch)
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)

    rt = DockerRuntime({
        "image_per_profile": {"default": "hermes-worker:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })

    def fake_run(cmd, **kw):
        out = MagicMock()
        out.returncode = 125
        out.stdout = b""
        out.stderr = b"docker: image not found"
        return out

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="docker run failed.*image not found"):
        rt.spawn(_make_fake_task(), workspace="/tmp", board="main")


def test_docker_runtime_skills_appended_to_command(monkeypatch):
    """Per-task skills are emitted as additional --skills X pairs."""
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    _stub_kanban_db_helpers(monkeypatch)
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)

    rt = DockerRuntime({
        "image_per_profile": {"default": "hermes-worker:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        out = MagicMock()
        out.stdout = b"cid123\n"
        out.returncode = 0
        return out

    monkeypatch.setattr("subprocess.run", fake_run)

    rt.spawn(
        _make_fake_task(skills=["custom-skill-a", "kanban-worker", "custom-skill-b"]),
        workspace="/tmp",
        board="main",
    )

    cmd = captured["cmd"]
    skills_idxs = [i for i, x in enumerate(cmd) if x == "--skills"]
    skill_values = [cmd[i + 1] for i in skills_idxs]
    # Built-in kanban-worker is always there
    assert "kanban-worker" in skill_values
    # Per-task skills also included
    assert "custom-skill-a" in skill_values
    assert "custom-skill-b" in skill_values
    # kanban-worker appears EXACTLY once (de-duped against the built-in)
    assert skill_values.count("kanban-worker") == 1


def test_docker_runtime_is_alive_returns_true_when_status_running(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "img:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })

    def fake_run(cmd, **kw):
        out = MagicMock()
        out.stdout = b"running\n"
        out.returncode = 0
        return out

    monkeypatch.setattr("subprocess.run", fake_run)
    assert rt.is_alive("abc123") is True


def test_docker_runtime_is_alive_returns_false_when_exited(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "img:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })

    def fake_run(cmd, **kw):
        out = MagicMock()
        out.stdout = b"exited\n"
        out.returncode = 0
        return out

    monkeypatch.setattr("subprocess.run", fake_run)
    assert rt.is_alive("abc123") is False


def test_docker_runtime_is_alive_handles_unknown_container(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "img:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })

    def fake_run(cmd, **kw):
        out = MagicMock()
        out.returncode = 1   # docker inspect returns non-zero on unknown container
        out.stdout = b""
        return out

    monkeypatch.setattr("subprocess.run", fake_run)
    assert rt.is_alive("nonexistent") is False


def test_docker_runtime_is_alive_handles_empty_handle(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "img:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })
    assert rt.is_alive("") is False
    assert rt.is_alive(None) is False


def test_docker_runtime_terminate_calls_docker_kill(monkeypatch):
    from tools.environments.kanban_spawn import DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = DockerRuntime({
        "image_per_profile": {"default": "img:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })

    captured = {}
    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        out = MagicMock()
        out.returncode = 0
        return out

    monkeypatch.setattr("subprocess.run", fake_run)
    rt.terminate("abc123", reason="test")
    assert captured["cmd"][:2] == ["docker", "kill"]
    assert "abc123" in captured["cmd"]


def test_load_runtime_docker_now_in_registry(monkeypatch):
    """After D1 Task 3, 'docker' is a registered runtime."""
    from tools.environments.kanban_spawn import load_runtime, DockerRuntime
    _stub_docker_present(monkeypatch)
    rt = load_runtime("docker", {
        "image_per_profile": {"default": "hermes-worker:latest"},
        "bind_mounts": [], "env_passthrough": [],
    })
    assert isinstance(rt, DockerRuntime)
    assert rt.name == "docker"
