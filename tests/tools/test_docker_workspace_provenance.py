"""Regression coverage for provenance-aware Docker workspace mapping."""

from __future__ import annotations

from pathlib import PureWindowsPath

import pytest

from tools import code_execution_tool, file_tools, terminal_tool


@pytest.fixture(autouse=True)
def _isolated_tool_state(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})


def _config(**overrides):
    config = {
        "env_type": "docker",
        "docker_image": "test:latest",
        "singularity_image": "docker://test:latest",
        "modal_image": "test:latest",
        "daytona_image": "test:latest",
        "cwd": "/root",
        "host_cwd": None,
        "timeout": 60,
        "container_persistent": False,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "docker_mount_cwd_to_workspace": True,
        "docker_volumes": [],
        "docker_forward_env": [],
        "docker_env": {},
        "docker_extra_args": [],
        "docker_run_as_host_user": False,
        "docker_network": True,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "host_root",
    [
        "/tmp/project",
        "/root/project",
        "/workspace/project",
        "/workspace-old/project",
    ],
)
def test_session_provenance_overrides_host_path_heuristics(monkeypatch, host_root):
    monkeypatch.setattr(terminal_tool.os.path, "isdir", lambda path: path == host_root)
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": host_root, "cwd_source": "session"}
    )

    resolved = terminal_tool._resolve_task_workspace(_config(), "session")

    assert resolved.host_cwd == host_root
    assert resolved.container_cwd == "/workspace"


def test_live_host_cwd_keeps_relative_suffix_below_workspace(tmp_path):
    host_root = tmp_path / "project"
    live_cwd = host_root / "src" / "pkg"
    live_cwd.mkdir(parents=True)
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(host_root), "cwd_source": "session"}
    )
    terminal_tool.record_session_cwd("session", str(live_cwd))

    resolved = terminal_tool._resolve_task_workspace(_config(), "session")

    assert resolved.host_cwd == str(host_root)
    assert resolved.container_cwd == "/workspace/src/pkg"


def test_windows_suffix_is_joined_with_container_path_semantics(monkeypatch):
    monkeypatch.setattr(
        terminal_tool,
        "_relative_host_suffix",
        lambda _cwd, _host_root: r"src\pkg",
    )
    monkeypatch.setattr(terminal_tool, "_HOST_PURE_PATH", PureWindowsPath)

    assert (
        terminal_tool._container_cwd_for_host_path(
            r"C:\project\src\pkg", r"C:\project"
        )
        == "/workspace/src/pkg"
    )


@pytest.mark.windows_only
def test_native_windows_host_path_maps_to_posix_container_cwd(tmp_path):
    host_root = tmp_path / "project"
    live_cwd = host_root / "src" / "pkg"
    live_cwd.mkdir(parents=True)

    assert (
        terminal_tool._container_cwd_for_host_path(
            str(live_cwd), str(host_root)
        )
        == "/workspace/src/pkg"
    )


def test_recorded_host_cwd_is_not_remapped_without_an_active_mount():
    host_root = "/Users/example/project"
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": host_root, "cwd_source": "session"}
    )
    workspace = terminal_tool._resolve_task_workspace(
        _config(docker_mount_cwd_to_workspace=False), "session"
    )

    resolved = terminal_tool._resolve_command_cwd(
        workdir=None,
        default_cwd=workspace.container_cwd,
        session_key="session",
        env_type="docker",
        host_workspace_root=workspace.host_cwd,
    )

    assert workspace.host_cwd is None
    assert workspace.container_cwd == "/root"
    assert resolved == "/root"


@pytest.mark.parametrize("cwd_source", [None, "container"])
def test_non_session_cwd_never_becomes_host_mount_by_coincidence(tmp_path, cwd_source):
    container_cwd = tmp_path / "benchmark"
    container_cwd.mkdir()
    overrides = {"cwd": str(container_cwd), "docker_image": "bench:latest"}
    if cwd_source is not None:
        overrides["cwd_source"] = cwd_source
    terminal_tool.register_task_env_overrides("benchmark", overrides)

    resolved = terminal_tool._resolve_task_workspace(_config(), "benchmark")

    assert resolved.host_cwd is None
    assert resolved.container_cwd == str(container_cwd)


def test_terminal_file_and_execute_code_share_workspace_mapping(monkeypatch, tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    config = _config()
    captured = {}

    class _FakeEnv:
        cwd = "/workspace"

        def execute(self, *args, **kwargs):
            return {"output": "", "exit_code": 0}

    def _capture_environment(**kwargs):
        captured[kwargs["task_id"]] = kwargs
        return _FakeEnv()

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_create_environment", _capture_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards", lambda *args, **kwargs: {"approved": True}
    )

    for task_id in ("terminal-session", "file-session", "code-session"):
        terminal_tool.register_task_env_overrides(
            task_id, {"cwd": str(workspace), "cwd_source": "session"}
        )

    terminal_tool.terminal_tool(command="pwd", task_id="terminal-session")
    file_tools._get_file_ops("file-session")
    code_execution_tool._get_or_create_env("code-session")

    for task_id in ("terminal-session", "file-session", "code-session"):
        assert captured[task_id]["host_cwd"] == str(workspace)
        assert captured[task_id]["cwd"] == "/workspace"
    assert (
        captured["code-session"]["container_config"]["docker_mount_cwd_to_workspace"]
        is True
    )


def test_explicit_workspace_reset_replaces_preserved_mount_root(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(first), "cwd_source": "session"}
    )
    terminal_tool.register_task_env_overrides(
        "session",
        {
            "cwd": str(second),
            "cwd_source": "session",
            "workspace_reset": True,
        },
    )

    resolved = terminal_tool._resolve_task_workspace(_config(), "session")

    assert resolved.host_cwd == str(second)
    assert resolved.container_cwd == "/workspace"


def test_workspace_reset_recreates_live_environment_and_file_cache(
    monkeypatch, tmp_path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _config())
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(first), "cwd_source": "session"}
    )

    cleanup_calls = []

    class _OldEnv:
        cwd = "/workspace"

        def cleanup(self, *, force_remove=False):
            cleanup_calls.append(("cleanup", force_remove))

        def wait_for_cleanup(self, timeout=30.0):
            cleanup_calls.append(("wait", timeout))
            return True

    terminal_tool._active_environments["session"] = _OldEnv()
    file_tools._file_ops_cache["session"] = object()

    terminal_tool.register_task_env_overrides(
        "session",
        {
            "cwd": str(second),
            "cwd_source": "session",
            "workspace_reset": True,
        },
    )

    captured = {}

    class _NewEnv:
        cwd = "/workspace"

    def _capture_environment(**kwargs):
        captured.update(kwargs)
        return _NewEnv()

    monkeypatch.setattr(terminal_tool, "_create_environment", _capture_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    recreated = terminal_tool.ensure_task_env("session")

    assert cleanup_calls == [("cleanup", True), ("wait", 60.0)]
    assert "session" not in file_tools._file_ops_cache
    assert recreated is not None
    assert captured["host_cwd"] == str(second)
    assert captured["cwd"] == "/workspace"


def test_workspace_reset_without_mount_keeps_live_environment(monkeypatch):
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(docker_mount_cwd_to_workspace=False),
    )
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": "/Users/example/first", "cwd_source": "session"}
    )

    class _ExistingEnv:
        cwd = "/workspace"

        def cleanup(self, *, force_remove=False):
            raise AssertionError("an unmounted workspace must not retire the sandbox")

    existing = _ExistingEnv()
    cached = object()
    terminal_tool._active_environments["session"] = existing
    file_tools._file_ops_cache["session"] = cached

    terminal_tool.register_task_env_overrides(
        "session",
        {
            "cwd": "/Users/example/second",
            "cwd_source": "session",
            "workspace_reset": True,
        },
    )

    assert terminal_tool._active_environments["session"] is existing
    assert existing.cwd == "/root"
    assert file_tools._file_ops_cache["session"] is cached


def test_workspace_root_read_modify_write_holds_override_lock(monkeypatch):
    class ObservedLock:
        held = False

        def __enter__(self):
            assert not self.held
            self.held = True

        def __exit__(self, *_args):
            self.held = False

    lock = ObservedLock()

    class GuardedOverrides(dict):
        def get(self, *args, **kwargs):
            assert lock.held
            return super().get(*args, **kwargs)

        def __setitem__(self, key, value):
            assert lock.held
            return super().__setitem__(key, value)

    monkeypatch.setattr(terminal_tool, "_task_env_overrides_lock", lock)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", GuardedOverrides())

    terminal_tool.register_task_env_overrides(
        "session", {"cwd": "/first", "cwd_source": "session"}
    )
    terminal_tool.register_task_env_overrides(
        "session",
        {"cwd": "/second", "cwd_source": "session", "workspace_reset": True},
    )

    assert terminal_tool._task_env_overrides["session"]["host_workspace_root"] == (
        "/second"
    )
