from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.kanban_runtime import (
    KANBAN_TERMINAL_RUNTIME_ENV,
    build_kanban_terminal_runtime,
    encode_kanban_terminal_runtime,
)


def _pin_kanban_worker(monkeypatch, workspace: Path, task_id: str = "t_runtime"):
    runtime = build_kanban_terminal_runtime(
        task_id=task_id, workspace_kind="dir", workspace=workspace
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace.resolve()))
    monkeypatch.setenv(
        KANBAN_TERMINAL_RUNTIME_ENV,
        encode_kanban_terminal_runtime(runtime),
    )
    return runtime


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


def test_runtime_gets_unique_container_key(monkeypatch, tmp_path):
    from tools import terminal_tool

    ws = tmp_path / "task"
    ws.mkdir()
    _pin_kanban_worker(monkeypatch, ws, task_id="t_unique")
    assert terminal_tool._resolve_container_task_id(None) == "kanban:t_unique"
    assert terminal_tool._resolve_container_task_id("arbitrary-session") == "kanban:t_unique"


def test_runtime_container_never_cross_process_reuses(monkeypatch, tmp_path):
    from tools import terminal_tool

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

    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", FakeDockerEnv)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)

    task_key = terminal_tool._resolve_container_task_id(None)
    env = terminal_tool._create_environment(
        env_type="docker",
        image=cfg["docker_image"],
        cwd=cfg["cwd"],
        timeout=60,
        container_config=terminal_tool._container_config_from_config(cfg),
        task_id=task_key,
        host_cwd=cfg["host_cwd"],
    )
    assert captured["task_id"] == "kanban:t_runtime"
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
