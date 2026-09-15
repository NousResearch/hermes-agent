"""Terminal pre-execution guards must not outlive the tool deadline."""

from __future__ import annotations

import time
from types import SimpleNamespace

import tools.terminal_tool as terminal_module


def _plan(timeout: float = 0.05) -> SimpleNamespace:
    return SimpleNamespace(
        config={},
        env_type="local",
        effective_task_id="pre-guard-deadline-test",
        cwd="/tmp",
        effective_timeout=timeout,
        promoted_from_foreground_timeout=None,
    )


def test_terminal_tool_bounds_a_wedged_pre_execution_guard(monkeypatch):
    """A stalled supervised-gateway identity probe cannot wedge terminal_tool."""
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan())
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(terminal_module, "_run_approval_guards", lambda *_a, **_k: terminal_module._ApprovalVerdict())
    monkeypatch.setattr(terminal_module, "_run_foreground", lambda *_a, **_k: "foreground-ran")

    def _wedged_supervised_gateway_probe(*_a, **_k):
        time.sleep(1)

    monkeypatch.setattr(terminal_module, "_pre_exec_block", _wedged_supervised_gateway_probe)

    start = time.monotonic()
    result = terminal_module.terminal_tool("echo ok")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"pre-execution guard wedged terminal_tool for {elapsed:.2f}s"
    assert result == "foreground-ran"


def test_terminal_tool_runs_normal_pre_execution_guard(monkeypatch):
    """A normal guard result still reaches foreground execution unchanged."""
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan())
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(terminal_module, "_run_approval_guards", lambda *_a, **_k: terminal_module._ApprovalVerdict())
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: None)
    monkeypatch.setattr(terminal_module, "_run_foreground", lambda *_a, **_k: "foreground-ran")

    assert terminal_module.terminal_tool("echo ok") == "foreground-ran"


def test_terminal_tool_preserves_pre_execution_rejection(monkeypatch):
    """A completed guard rejection still returns its original tool result."""
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan())
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: (_ for _ in ()).throw(
        terminal_module._Rejected('{"status":"blocked"}')
    ))

    assert terminal_module.terminal_tool("echo ok") == '{"status":"blocked"}'
