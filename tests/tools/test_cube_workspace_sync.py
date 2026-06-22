"""Tests for CubeSandboxEnvironment workspace sync hooks."""

from __future__ import annotations

import tarfile
import types
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clean_workspace_sync_state():
    from tools.workspace_sync import clear_touched

    clear_touched()
    yield
    clear_touched()


def _patch_cube_sdk(monkeypatch, *, mock_sandbox: MagicMock) -> None:
    monkeypatch.setattr("tools.environments.cube_sandbox._ensure_cube_sdk", lambda: None)

    e2b_mod = types.ModuleType("e2b_code_interpreter")
    e2b_mod.Sandbox = MagicMock()
    e2b_mod.Sandbox.create = MagicMock(return_value=mock_sandbox)
    monkeypatch.setitem(__import__("sys").modules, "e2b_code_interpreter", e2b_mod)

    cmd_mod = types.ModuleType("e2b.sandbox.commands.command_handle")

    class CommandExitException(Exception):
        def __init__(
            self,
            stdout: str = "",
            stderr: str = "",
            exit_code: int = 1,
            error: object = None,
        ) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.exit_code = exit_code
            self.error = error
            super().__init__()

    cmd_mod.CommandExitException = CommandExitException
    monkeypatch.setitem(
        __import__("sys").modules,
        "e2b.sandbox.commands.command_handle",
        cmd_mod,
    )

    fs_mod = types.ModuleType("e2b.sandbox.filesystem.filesystem")

    class WriteEntry:
        def __init__(self, path, data):
            self.path = path
            self.data = data

    fs_mod.WriteEntry = WriteEntry
    monkeypatch.setitem(
        __import__("sys").modules,
        "e2b.sandbox.filesystem.filesystem",
        fs_mod,
    )


@pytest.fixture()
def cube_sync_env(monkeypatch, tmp_path):
    ws = Path("/tmp") / f"hermes-cube-sync-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)
    sample = ws / "data.txt"
    sample.write_text("pod-content", encoding="utf-8")

    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    monkeypatch.setenv("CUBE_API_URL", "http://cube.test:3000")
    monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl-test")
    monkeypatch.setenv("CUBE_API_KEY", "test-key")

    from tools.workspace_sync import register_touched, get_touched_paths

    register_touched("task-sync", str(sample))

    uploaded: list[tuple[str, bytes]] = []
    tar_payload = BytesIO()

    mock_sb = MagicMock()
    mock_sb.sandbox_id = "sb-sync"
    mock_sb.commands.run = MagicMock(
        return_value=SimpleNamespace(stdout="", stderr="", exit_code=0, error=None)
    )

    def _write(path, data):
        uploaded.append((path, data))

    def _write_files(entries):
        for entry in entries:
            uploaded.append((entry.path, entry.data))

    def _read(path, format="text", **kwargs):
        assert format == "bytes"
        return tar_payload.getvalue()

    mock_sb.files.write = _write
    mock_sb.files.write_files = _write_files
    mock_sb.files.read = _read

    _patch_cube_sdk(monkeypatch, mock_sandbox=mock_sb)
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)

    from tools.environments.cube_sandbox import CubeSandboxEnvironment

    env = CubeSandboxEnvironment(task_id="task-sync", persistent_filesystem=False)
    return env, mock_sb, uploaded, tar_payload, ws, sample


def test_before_execute_uploads_touched_files(cube_sync_env):
    env, _mock_sb, uploaded, _tar_payload, _ws, sample = cube_sync_env

    env._before_execute()

    remote_paths = [path for path, _data in uploaded]
    assert "/home/user/workspace/data.txt" in remote_paths
    assert any(data == sample.read_bytes() for _path, data in uploaded)


def test_before_execute_fail_closed_on_oversized_file(cube_sync_env, monkeypatch):
    env, _mock_sb, _uploaded, _tar_payload, ws, _sample = cube_sync_env
    from tools.workspace_sync import WorkspaceSyncError, register_touched

    big = ws / "too-big.bin"
    big.write_bytes(b"x" * 2048)
    register_touched("task-sync", str(big))
    monkeypatch.setattr(
        "tools.environments.cube_sandbox.workspace_sync_max_bytes",
        lambda: 512,
    )

    with pytest.raises(WorkspaceSyncError, match="exceeds sync cap"):
        env._before_execute()


def test_cleanup_sync_back_applies_remote_changes(cube_sync_env, monkeypatch):
    env, mock_sb, uploaded, tar_payload, ws, sample = cube_sync_env
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_BACK", "cleanup_only")

    env._before_execute()
    assert uploaded, "expected sync in before sync_back test"

    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        changed = b"vm-updated"
        info = tarfile.TarInfo(name="home/user/workspace/data.txt")
        info.size = len(changed)
        tar.addfile(info, BytesIO(changed))
    tar_payload.write(buf.getvalue())

    env.cleanup()

    assert sample.read_bytes() == b"vm-updated"
    mock_sb.create_snapshot.assert_not_called()


def test_cleanup_skips_sync_back_when_after_terminal(cube_sync_env, monkeypatch):
    env, _mock_sb, uploaded, _tar_payload, _ws, _sample = cube_sync_env
    sync_back_calls: list[int] = []

    def _capture_sync_back(hermes_home=None):
        sync_back_calls.append(1)
        return True

    env._before_execute()
    assert uploaded
    env._workspace_sync.sync_back = _capture_sync_back  # type: ignore[method-assign]
    env._post_terminal_sync_back_failed = False

    env.cleanup()
    assert sync_back_calls == []


def test_cleanup_retries_sync_back_after_post_terminal_failure(cube_sync_env):
    env, _mock_sb, uploaded, _tar_payload, _ws, _sample = cube_sync_env
    sync_back_calls: list[int] = []

    def _capture_sync_back(hermes_home=None):
        sync_back_calls.append(1)
        return True

    env._before_execute()
    assert uploaded
    env._workspace_sync.sync_back = _capture_sync_back  # type: ignore[method-assign]
    env._post_terminal_sync_back_failed = True

    env.cleanup()
    assert len(sync_back_calls) == 1
    assert env._post_terminal_sync_back_failed is False


def test_after_execute_sync_back_on_success(cube_sync_env):
    env, _mock_sb, uploaded, _tar_payload, _ws, _sample = cube_sync_env
    sync_back_calls: list[object] = []

    def _capture_sync_back(hermes_home=None):
        sync_back_calls.append(hermes_home)
        return True

    env._before_execute()
    assert uploaded
    env._workspace_sync.sync_back = _capture_sync_back  # type: ignore[method-assign]

    result = env.execute("echo ok")
    assert result["returncode"] == 0
    assert len(sync_back_calls) == 1
    assert env._post_terminal_sync_back_failed is False


def test_after_execute_skips_sync_back_on_failure(cube_sync_env):
    env, mock_sb, uploaded, _tar_payload, _ws, _sample = cube_sync_env
    sync_back_calls: list[int] = []

    def _capture_sync_back(hermes_home=None):
        sync_back_calls.append(1)

    env._before_execute()
    assert uploaded
    env._workspace_sync.sync_back = _capture_sync_back  # type: ignore[method-assign]

    cmd_mod = __import__("sys").modules["e2b.sandbox.commands.command_handle"]
    mock_sb.commands.run = MagicMock(
        side_effect=cmd_mod.CommandExitException(
            stdout="",
            stderr="fail",
            exit_code=1,
            error=None,
        )
    )

    result = env.execute("false")
    assert result["returncode"] == 1
    assert sync_back_calls == []


def test_after_execute_sync_back_without_touched_paths(monkeypatch, tmp_path):
    """Workspace-scope egress: terminal-only VM writes sync back without register_touched."""
    ws = Path("/tmp") / f"hermes-cube-no-touch-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    monkeypatch.setenv("CUBE_API_URL", "http://cube.test:3000")
    monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl-test")
    monkeypatch.setenv("CUBE_API_KEY", "test-key")

    tar_payload = BytesIO()
    uploaded: list[tuple[str, bytes]] = []

    mock_sb = MagicMock()
    mock_sb.sandbox_id = "sb-no-touch"
    mock_sb.commands.run = MagicMock(
        return_value=SimpleNamespace(stdout="ok", stderr="", exit_code=0, error=None)
    )

    def _write(path, data):
        uploaded.append((path, data))

    def _write_files(entries):
        for entry in entries:
            uploaded.append((entry.path, entry.data))

    def _read(path, format="text", **kwargs):
        assert format == "bytes"
        return tar_payload.getvalue()

    mock_sb.files.write = _write
    mock_sb.files.write_files = _write_files
    mock_sb.files.read = _read

    _patch_cube_sdk(monkeypatch, mock_sandbox=mock_sb)
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)

    from tools.environments.cube_sandbox import CubeSandboxEnvironment
    from tools.workspace_sync import get_touched_paths

    env = CubeSandboxEnvironment(task_id="no-touch", persistent_filesystem=False)
    assert get_touched_paths("no-touch") == set()

    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        payload = b"created-in-vm"
        info = tarfile.TarInfo(name="home/user/workspace/new.txt")
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    tar_payload.write(buf.getvalue())

    result = env.execute("echo created > new.txt")
    assert result["returncode"] == 0
    assert (ws / "new.txt").read_bytes() == b"created-in-vm"
