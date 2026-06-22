"""Tests for Cube high/low risk split (P1-a/b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.session_context import set_current_session_id
from tools import file_tools, terminal_tool
from tools.cube_split import is_cube_split_enabled, workspace_root


@pytest.fixture(autouse=True)
def _clean_file_ops_cache():
    file_tools.clear_file_ops_cache()
    yield
    file_tools.clear_file_ops_cache()


def test_is_cube_split_enabled(monkeypatch):
    monkeypatch.delenv("SANDBOX_TYPE", raising=False)
    assert not is_cube_split_enabled()
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    assert is_cube_split_enabled()


def test_workspace_root_from_env(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    assert workspace_root() == ws.resolve()


def test_write_file_stays_on_pod_without_cube(monkeypatch, tmp_path):
    ws = Path("/tmp") / f"hermes-split-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))

    result_raw = file_tools.write_file_tool("split-marker.txt", "pod-workspace")
    result = json.loads(result_raw)
    assert "error" not in result, result
    assert (ws / "split-marker.txt").read_text(encoding="utf-8") == "pod-workspace"


def test_file_ops_does_not_create_cube_env(monkeypatch, tmp_path):
    ws = Path("/tmp") / f"hermes-split2-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))

    called = {"cube": False}

    def _boom(*args, **kwargs):
        if kwargs.get("env_type") == "cube_sandbox" or args[0] == "cube_sandbox":
            called["cube"] = True
        raise AssertionError("cube env should not be created for file ops")

    monkeypatch.setattr(terminal_tool, "_create_environment", _boom)

    file_tools.write_file_tool("no-cube.txt", "hello")
    assert called["cube"] is False
    assert terminal_tool._active_environments == {}


def test_resolve_base_dir_uses_workspace(monkeypatch, tmp_path):
    ws = Path("/tmp") / f"hermes-split3-{tmp_path.name}"
    ws.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(ws))
    assert file_tools._resolve_base_dir() == ws.resolve()


def test_sandbox_type_enables_session_key_without_terminal_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_INSTANCE_NAME", "inst-split")
    set_current_session_id("sess-split")
    try:
        assert (
            terminal_tool._resolve_container_task_id("subagent-0-deadbeef")
            == "inst-split-sess-split"
        )
    finally:
        set_current_session_id("")


def test_cube_sandbox_session_id_not_collapsed(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "cube_sandbox")
    monkeypatch.setenv("HERMES_INSTANCE_NAME", "inst-test")
    set_current_session_id("sess-abc")
    try:
        assert (
            terminal_tool._resolve_container_task_id("subagent-0-deadbeef")
            == "inst-test-sess-abc"
        )
    finally:
        set_current_session_id("")


def test_cube_sandbox_without_session_still_default(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "cube_sandbox")
    monkeypatch.delenv("HERMES_INSTANCE_NAME", raising=False)
    set_current_session_id("")
    try:
        assert terminal_tool._resolve_container_task_id("subagent-1-beef") == "default"
    finally:
        set_current_session_id("")


def test_sandbox_task_id_for_session_composite(monkeypatch):
    from tools.cube_split import env_cache_keys_for_cleanup, sandbox_task_id_for_session

    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_INSTANCE_NAME", "inst-a")
    assert sandbox_task_id_for_session("sess-1") == "inst-a-sess-1"
    assert sandbox_task_id_for_session("inst-a-sess-1") == "inst-a-sess-1"
    assert env_cache_keys_for_cleanup("sess-1") == ["sess-1", "inst-a-sess-1"]
    assert env_cache_keys_for_cleanup("inst-a-sess-1") == ["inst-a-sess-1"]


def test_cleanup_vm_finds_cube_env_by_bare_session_id(monkeypatch):
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_INSTANCE_NAME", "inst-close")
    cache_key = "inst-close-sess-close"
    cleaned = {"called": False}

    class _FakeEnv:
        def cleanup(self):
            cleaned["called"] = True

    terminal_tool._active_environments[cache_key] = _FakeEnv()
    try:
        terminal_tool.cleanup_vm("sess-close")
        assert cleaned["called"]
        assert cache_key not in terminal_tool._active_environments
    finally:
        terminal_tool._active_environments.pop(cache_key, None)


def test_kill_all_finds_cube_process_by_bare_session_id(monkeypatch):
    import time

    from tools.process_registry import ProcessRegistry, ProcessSession

    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("HERMES_INSTANCE_NAME", "inst-kill")
    registry = ProcessRegistry()
    composite = "inst-kill-sess-bg"
    session = ProcessSession(
        id="proc_cube_bg",
        command="sleep 999",
        task_id=composite,
        started_at=time.time(),
    )
    registry._running[session.id] = session

    killed_ids: list[str] = []

    def _fake_kill(session_id: str, *, source: str = "process.kill") -> dict:
        killed_ids.append(session_id)
        session.exited = True
        return {"status": "killed", "session_id": session_id}

    monkeypatch.setattr(registry, "kill_process", _fake_kill)

    assert registry.kill_all("sess-bg") == 1
    assert killed_ids == ["proc_cube_bg"]
