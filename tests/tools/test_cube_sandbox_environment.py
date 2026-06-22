"""Unit tests for CubeSandboxEnvironment."""

from __future__ import annotations

import importlib
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_check_cube_sandbox_requirements_needs_credentials(monkeypatch):
    monkeypatch.setenv("CUBE_API_URL", "http://cube.test:3000")
    monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl-test")
    monkeypatch.delenv("CUBE_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("SANDBOX_TOKEN_API_URL", raising=False)
    monkeypatch.delenv("CUBE_TOKEN_API_URL", raising=False)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_a, **_k: None)

    mod = importlib.import_module("tools.environments.cube_sandbox")
    assert mod.check_cube_sandbox_requirements() is False


def test_check_cube_sandbox_requirements_accepts_static_key(monkeypatch):
    monkeypatch.setenv("CUBE_API_URL", "http://cube.test:3000")
    monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl-test")
    monkeypatch.setenv("CUBE_API_KEY", "test-key")
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_a, **_k: None)

    mod = importlib.import_module("tools.environments.cube_sandbox")
    assert mod.check_cube_sandbox_requirements() is True


def _patch_cube_sdk(monkeypatch, *, mock_sandbox: MagicMock) -> None:
    monkeypatch.setattr("tools.environments.cube_sandbox._ensure_cube_sdk", lambda: None)

    e2b_mod = types.ModuleType("e2b_code_interpreter")
    e2b_mod.Sandbox = MagicMock()
    e2b_mod.Sandbox.create = MagicMock(return_value=mock_sandbox)
    monkeypatch.setitem(__import__("sys").modules, "e2b_code_interpreter", e2b_mod)

    cmd_mod = types.ModuleType("e2b.sandbox.commands.command_handle")
    cmd_mod.CommandExitException = type("CommandExitException", (Exception,), {})
    monkeypatch.setitem(
        __import__("sys").modules,
        "e2b.sandbox.commands.command_handle",
        cmd_mod,
    )


@pytest.fixture()
def cube_env_factory(monkeypatch):
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    monkeypatch.setenv("CUBE_API_URL", "http://cube.test:3000")
    monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl-test")
    monkeypatch.setenv("CUBE_API_KEY", "test-key")

    def _factory(*, run_side_effect=None):
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "sb-test"
        if run_side_effect is not None:
            mock_sb.commands.run = run_side_effect
        else:
            mock_sb.commands.run = MagicMock(
                return_value=SimpleNamespace(
                    stdout="ok",
                    stderr="",
                    exit_code=0,
                    error=None,
                )
            )
        _patch_cube_sdk(monkeypatch, mock_sandbox=mock_sb)

        from tools.environments.cube_sandbox import CubeSandboxEnvironment

        env = CubeSandboxEnvironment(task_id="task-1")
        return env, mock_sb

    return _factory


def test_run_bash_allows_concurrent_commands(cube_env_factory):
    """execute_code RPC must poll while python3 script.py holds a command slot."""
    gate = threading.Event()
    active = {"count": 0, "max": 0}
    counter_lock = threading.Lock()

    def mock_run(cmd, timeout=None):
        with counter_lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        try:
            if "sleep-long" in cmd:
                if not gate.wait(timeout=2):
                    return SimpleNamespace(
                        stdout="",
                        stderr="timed out",
                        exit_code=124,
                        error=None,
                    )
                return SimpleNamespace(
                    stdout="long-done",
                    stderr="",
                    exit_code=0,
                    error=None,
                )
            return SimpleNamespace(
                stdout="fast",
                stderr="",
                exit_code=0,
                error=None,
            )
        finally:
            with counter_lock:
                active["count"] -= 1

    env, _mock_sb = cube_env_factory(run_side_effect=mock_run)

    results: dict[str, tuple[int | None, str]] = {}

    def run_cmd(label: str, cmd: str) -> None:
        handle = env._run_bash(cmd, timeout=5)
        handle.wait(timeout=10)
        output = handle.stdout.read() if handle.stdout else ""
        results[label] = (handle.returncode, output)

    def long_run():
        run_cmd("long", "echo sleep-long marker")

    def fast_run():
        run_cmd("fast", "echo fast marker")

    t_long = threading.Thread(target=long_run, daemon=True)
    t_long.start()
    time.sleep(0.05)
    t_fast = threading.Thread(target=fast_run, daemon=True)
    t_fast.start()

    t_fast.join(timeout=2)
    assert not t_fast.is_alive(), "RPC poll command blocked on Cube global lock"
    assert results["fast"][0] == 0
    assert active["max"] >= 2, "expected overlapping commands.run calls"

    gate.set()
    t_long.join(timeout=2)
    assert not t_long.is_alive()
    assert results["long"][0] == 0
