from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.kanban_runtime import (
    KANBAN_TERMINAL_RUNTIME_ENV,
    build_kanban_terminal_runtime,
    encode_kanban_terminal_runtime,
    physical_task_key,
)

HERMES_KANBAN_TASK_ENV = "HERMES_KANBAN_TASK"


def _pin_kanban_worker(monkeypatch, workspace: Path, task_id: str = "t_runtime"):
    runtime = build_kanban_terminal_runtime(
        task_id=task_id,
        workspace_kind="dir",
        workspace=workspace,
        authorized_roots=[workspace.parent],
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace.resolve()))
    monkeypatch.setenv(
        KANBAN_TERMINAL_RUNTIME_ENV,
        encode_kanban_terminal_runtime(runtime),
    )
    # P1-B: terminal_tool requires positive dispatcher authority. Grant it the
    # same way the worker bootstrap does: embed the one-shot marker, then
    # consume it via bootstrap_dispatcher_authority.
    from agent.delegation_context import (
        DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV,
        _dispatcher_ownership_proof,
        bootstrap_dispatcher_authority,
    )

    proof, nonce = _dispatcher_ownership_proof(task_id)
    monkeypatch.setenv(
        DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV, f"{proof}.{nonce}"
    )
    bootstrap_dispatcher_authority(
        task_id=task_id,
        workspace=str(workspace.resolve()),
    )
    return runtime


@pytest.fixture(autouse=True)
def _isolated_dispatcher_authority():
    """Keep positive dispatcher authority test-local (ContextVar hygiene)."""
    from agent import delegation_context as dc

    token = dc._DISPATCHER_AUTHORITY.set(False)
    veto_token = dc._NON_DISPATCHER_VETO.set(False)
    yield
    dc._DISPATCHER_AUTHORITY.reset(token)
    dc._NON_DISPATCHER_VETO.reset(veto_token)


def _grant_worker_authority(monkeypatch, task_id: str, workspace: Path) -> None:
    """Embed + consume the dispatcher ownership marker for *task_id* (P1-B)."""
    from agent.delegation_context import (
        DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV,
        _dispatcher_ownership_proof,
        bootstrap_dispatcher_authority,
    )

    proof, nonce = _dispatcher_ownership_proof(task_id)
    monkeypatch.setenv(
        DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV, f"{proof}.{nonce}"
    )
    bootstrap_dispatcher_authority(
        task_id=task_id,
        workspace=str(workspace),
    )


def test_runtime_mount_overrides_profile_docker_volumes(monkeypatch, tmp_path):
    from tools import terminal_tool

    ws = tmp_path / "task"
    ws.mkdir()
    _pin_kanban_worker(monkeypatch, ws)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps(["/:/host-root"]))
    monkeypatch.setenv("TERMINAL_DOCKER_HOST_PATH_MAP", "[]")

    cfg = terminal_tool._get_env_config()
    assert cfg["cwd"] == "/workspace"
    assert cfg["host_cwd"] is None
    # The dispatcher-owned task contract is the complete host-bind authority.
    assert cfg["docker_volumes"] == []
    assert cfg["docker_runtime_mounts"] == [
        {
            "source": str(ws.resolve()),
            "target": "/workspace",
            "read_only": False,
            "purpose": "workspace",
        }
    ]
    assert terminal_tool._docker_has_host_access(cfg) is True


def test_preterminal_file_tool_relative_path_uses_runtime_workspace(
    monkeypatch, tmp_path
):
    from tools import file_tools
    from tools import file_tools_paths

    ws = tmp_path / "project" / "task"
    ws.mkdir(parents=True)
    _pin_kanban_worker(monkeypatch, ws, task_id="t_file_path")
    # This is the host-form value the dispatcher exports. Before the first
    # terminal call it must not win over the runtime's container cwd.
    monkeypatch.setenv("TERMINAL_CWD", str(ws.resolve()))
    monkeypatch.setattr(
        file_tools_paths, "_terminal_env_type_for_task", lambda _task: "docker"
    )

    resolved = file_tools._resolve_path_for_task("canary.txt", "default")
    assert str(resolved) == "/workspace/canary.txt"


def test_preterminal_local_worker_keeps_host_workspace_path(monkeypatch, tmp_path):
    from tools import file_tools
    from tools import file_tools_paths

    ws = tmp_path / "project" / "task"
    ws.mkdir(parents=True)
    _pin_kanban_worker(monkeypatch, ws, task_id="t_local_file_path")
    monkeypatch.setenv("TERMINAL_CWD", str(ws.resolve()))
    monkeypatch.setattr(
        file_tools_paths, "_terminal_env_type_for_task", lambda _task: "local"
    )

    resolved = file_tools._resolve_path_for_task("canary.txt", "default")
    assert resolved == (ws / "canary.txt").resolve()


def test_runtime_gets_unique_container_key(monkeypatch, tmp_path):
    from tools import terminal_tool

    ws = tmp_path / "task"
    ws.mkdir()
    _pin_kanban_worker(monkeypatch, ws, task_id="t_unique")
    assert terminal_tool._resolve_container_task_id(None) == physical_task_key("t_unique")
    assert terminal_tool._resolve_container_task_id("arbitrary-session") == physical_task_key("t_unique")
    # P1-D: the physical key is portable-safe even for logical ids containing ':'.
    unsafe = "task:with:colons"
    runtime_unsafe = build_kanban_terminal_runtime(
        task_id=unsafe,
        workspace_kind="dir",
        workspace=ws,
        authorized_roots=[ws.parent],
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", unsafe)
    monkeypatch.setenv(
        KANBAN_TERMINAL_RUNTIME_ENV,
        encode_kanban_terminal_runtime(runtime_unsafe),
    )
    key = terminal_tool._resolve_container_task_id(None)
    assert ":" not in key
    assert key == physical_task_key(unsafe)


def test_runtime_container_never_cross_process_reuses(monkeypatch, tmp_path):
    from tools import terminal_tool
    from tools import terminal_tool_backends

    ws = tmp_path / "task"
    ws.mkdir()
    _pin_kanban_worker(monkeypatch, ws)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_HOST_PATH_MAP", "[]")

    cfg = terminal_tool._get_env_config()
    captured = {}

    class FakeDockerEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        terminal_tool_backends, "_DockerEnvironment", FakeDockerEnv
    )

    task_key = terminal_tool._resolve_container_task_id(None)
    env = terminal_tool_backends._create_environment(
        env_type="docker",
        image=cfg["docker_image"],
        cwd=cfg["cwd"],
        timeout=60,
        container_config=terminal_tool_backends._container_config_from_config(cfg),
        task_id=task_key,
        host_cwd=cfg["host_cwd"],
    )
    assert captured["task_id"] == physical_task_key("t_runtime")
    assert captured["runtime_mounts"] == cfg["docker_runtime_mounts"]
    assert captured["persist_across_processes"] is False
    assert getattr(env, "_session_scoped") is True


def test_remote_worker_translates_runtime_mount(monkeypatch, tmp_path):
    from tools import terminal_tool

    root = tmp_path / "projects"
    ws = root / "task"
    ws.mkdir(parents=True)
    _pin_kanban_worker(monkeypatch, ws)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("DOCKER_HOST", "ssh://docker@example.com")
    monkeypatch.setenv(
        "TERMINAL_DOCKER_HOST_PATH_MAP",
        json.dumps([{"local_root": str(root), "host_root": "/mnt/projects"}]),
    )

    cfg = terminal_tool._get_env_config()
    assert cfg["docker_runtime_mounts"][0]["source"] == "/mnt/projects/task"



@pytest.mark.parametrize("outside_name", ["opt-data", ".hermes", "sibling-project"])
def test_agreed_runtime_workspace_outside_authority_fails_closed(
    monkeypatch, tmp_path, outside_name
):
    from tools import terminal_tool

    project = tmp_path / "project"
    outside = tmp_path / outside_name
    project.mkdir()
    outside.mkdir()
    runtime = build_kanban_terminal_runtime(
        task_id="t_outside",
        workspace_kind="dir",
        workspace=outside,
        authorized_roots=[project],
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_outside")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(outside.resolve()))
    monkeypatch.setenv(
        KANBAN_TERMINAL_RUNTIME_ENV,
        encode_kanban_terminal_runtime(runtime),
    )
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_HOST_PATH_MAP", "[]")
    _grant_worker_authority(monkeypatch, "t_outside", outside.resolve())

    with pytest.raises(RuntimeError, match="outside authorized workspace roots"):
        terminal_tool._get_env_config()


def test_profile_docker_extra_mounts_reach_constructor_and_fail_before_docker(
    monkeypatch, tmp_path
):
    from tools import terminal_tool
    from tools import terminal_tool_backends
    from tools.environments import docker as docker_env

    ws = tmp_path / "project" / "task"
    ws.mkdir(parents=True)
    _pin_kanban_worker(monkeypatch, ws, task_id="t_extra_mounts")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    attacks = [
        "--mount",
        "type=bind,src=/opt/data,dst=/host-data",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
    ]
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", json.dumps(attacks))
    monkeypatch.setenv("TERMINAL_DOCKER_HOST_PATH_MAP", "[]")

    cfg = terminal_tool._get_env_config()
    # Preserve the operator/profile surface; enforcement belongs at physical
    # Docker construction where runtime_mounts are known to be authoritative.
    assert cfg["docker_extra_args"] == attacks

    def must_not_reach_docker(*_args, **_kwargs):
        pytest.fail("Docker availability/probe must not run before mount-arg rejection")

    monkeypatch.setattr(
        docker_env, "_ensure_docker_available", must_not_reach_docker
    )

    with pytest.raises(ValueError, match="mount-capable docker_extra_args"):
        terminal_tool_backends._create_environment(
            env_type="docker",
            image=cfg["docker_image"],
            cwd=cfg["cwd"],
            timeout=60,
            container_config=terminal_tool_backends._container_config_from_config(cfg),
            task_id=terminal_tool._resolve_container_task_id(None),
            host_cwd=cfg["host_cwd"],
        )
