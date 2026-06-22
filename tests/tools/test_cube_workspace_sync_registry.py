"""Tests for Cube Pod↔VM workspace sync registry and path mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import file_tools
from tools.cube_split import remote_workspace_root
from tools.workspace_sync import (
    WorkspaceSyncError,
    check_workspace_sync_ready,
    clear_touched,
    get_touched_paths,
    host_path_from_remote_workspace,
    host_to_remote,
    iter_workspace_sync_files,
    register_touched,
    remap_pod_path_to_vm,
    rewrite_terminal_command_for_workspace_sync,
    validate_file_tool_path,
    workspace_sync_back_after_terminal,
    workspace_sync_back_scope,
    workspace_sync_enabled,
    workspace_sync_max_bytes,
    SYNC_BACK_SCOPE_WORKSPACE,
)


def _pod_workspace(tmp_path: Path) -> Path:
    """Workspace under /tmp — macOS pytest tmp_path lives under /private/var."""
    ws = Path("/tmp") / f"hermes-ws-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture(autouse=True)
def _clean_state():
    clear_touched()
    file_tools.clear_file_ops_cache()
    yield
    clear_touched()
    file_tools.clear_file_ops_cache()


def test_host_to_remote_mapping(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    host = ws / "data" / "input.csv"
    host.parent.mkdir()
    host.write_text("a,b", encoding="utf-8")
    monkeypatch.setenv("CUBE_REMOTE_WORKSPACE_MOUNT", "/home/user/workspace")
    assert host_to_remote(host, ws) == "/home/user/workspace/data/input.csv"


def test_iter_workspace_sync_skips_missing_and_blocked(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "ok.txt").write_text("ok", encoding="utf-8")
    git_dir = ws / ".git" / "config"
    git_dir.parent.mkdir()
    git_dir.write_text("git", encoding="utf-8")

    files = iter_workspace_sync_files(
        ws,
        paths={".git/config", "ok.txt", "missing.txt"},
        remote_root="/home/user/workspace",
        max_bytes=1024,
    )
    assert files == [(str(ws / "ok.txt"), "/home/user/workspace/ok.txt")]


def test_register_and_enumerate_touched(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")

    target = ws / "script.py"
    target.write_text("print(1)", encoding="utf-8")
    register_touched("task-a", str(target))
    assert get_touched_paths("task-a") == {"script.py"}


def test_write_file_registers_touched_path(monkeypatch, tmp_path):
    ws = _pod_workspace(tmp_path)
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")

    result = json.loads(file_tools.write_file_tool("sync-me.txt", "hello", task_id="t1"))
    assert "error" not in result, result
    assert get_touched_paths("default") == {"sync-me.txt"}


def test_workspace_sync_disabled_outside_split(monkeypatch):
    monkeypatch.delenv("SANDBOX_TYPE", raising=False)
    assert not workspace_sync_enabled()


def test_remote_workspace_root_default():
    assert remote_workspace_root() == "/home/user/workspace"


def test_workspace_sync_max_bytes_default():
    assert workspace_sync_max_bytes() == 100 * 1024 * 1024


def test_check_workspace_sync_ready_raises_on_missing(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(WorkspaceSyncError, match="missing.txt"):
        check_workspace_sync_ready(
            {"missing.txt"},
            pod_workspace=ws,
            remote_root="/home/user/workspace",
        )


def test_check_workspace_sync_ready_raises_on_oversized(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    big = ws / "big.bin"
    big.write_bytes(b"x" * 2048)
    with pytest.raises(WorkspaceSyncError, match="exceeds sync cap"):
        check_workspace_sync_ready(
            {"big.bin"},
            pod_workspace=ws,
            remote_root="/home/user/workspace",
            max_bytes=1024,
        )


def test_clear_touched_before_sync_back_empties_file_mapping(monkeypatch, tmp_path):
    """sync_back resolves host paths via get_touched_paths — must not clear early."""
    ws = _pod_workspace(tmp_path)
    target = ws / "data.txt"
    target.write_text("pod", encoding="utf-8")
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")

    register_touched("default", str(target))
    assert get_touched_paths("default") == {"data.txt"}

    clear_touched("default")
    files = iter_workspace_sync_files(
        ws,
        paths=get_touched_paths("default"),
        remote_root="/home/user/workspace",
    )
    assert files == []


def test_validate_file_tool_path_rejects_outside_workspace(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    outside = tmp_path / "tangshi.txt"
    err = validate_file_tool_path(str(outside))
    assert err is not None
    assert "outside the Pod workspace" in err


def test_validate_file_tool_path_rejects_vm_path(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_REMOTE_WORKSPACE_MOUNT", "/vm/workspace")
    err = validate_file_tool_path("/vm/workspace/tangshi.txt")
    assert err is not None
    assert "microVM path" in err


def test_validate_file_tool_path_allows_relative(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    assert validate_file_tool_path("tangshi.txt") is None


def test_rewrite_terminal_command_pod_to_vm(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    monkeypatch.setenv("CUBE_REMOTE_WORKSPACE_MOUNT", "/vm/workspace")
    cmd = f"cat >> {ws}/tangshi.txt"
    assert rewrite_terminal_command_for_workspace_sync(cmd) == "cat >> /vm/workspace/tangshi.txt"


def test_remap_pod_path_to_vm_exact_prefix(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    monkeypatch.setenv("CUBE_REMOTE_WORKSPACE_MOUNT", "/vm/workspace")
    assert remap_pod_path_to_vm(str(ws)) == "/vm/workspace"
    assert remap_pod_path_to_vm(str(ws / "a.txt")) == "/vm/workspace/a.txt"


def test_workspace_sync_back_scope_default(monkeypatch):
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    assert workspace_sync_back_scope() == SYNC_BACK_SCOPE_WORKSPACE


def test_host_path_from_remote_workspace(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    assert host_path_from_remote_workspace(
        "/home/user/workspace/out.txt",
        pod_workspace=ws,
        remote_root="/home/user/workspace",
    ) == str(ws / "out.txt")


def test_host_path_from_remote_workspace_rejects_traversal(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    assert host_path_from_remote_workspace(
        "/home/user/workspace/../../../etc/passwd",
        pod_workspace=ws,
        remote_root="/home/user/workspace",
    ) is None
    assert host_path_from_remote_workspace(
        "/home/user/workspace/sub/../../outside.txt",
        pod_workspace=ws,
        remote_root="/home/user/workspace",
    ) is None


def test_workspace_sync_back_after_terminal_default(monkeypatch):
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    assert workspace_sync_back_after_terminal() is True


def test_workspace_sync_back_cleanup_only(monkeypatch):
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_BACK", "cleanup_only")
    assert workspace_sync_back_after_terminal() is False


def test_write_file_rejects_path_outside_workspace(monkeypatch, tmp_path):
    ws = _pod_workspace(tmp_path)
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("CUBE_WORKSPACE_SYNC_ENABLED", "1")
    outside = Path("/tmp") / f"outside-{tmp_path.name}.txt"
    result = json.loads(file_tools.write_file_tool(str(outside), "hello", task_id="t1"))
    assert "error" in result
    assert "outside the Pod workspace" in result["error"]
