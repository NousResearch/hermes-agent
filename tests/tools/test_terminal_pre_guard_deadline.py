"""Terminal pre-execution guards share the command's wall-clock deadline (#111922).

The supervised-gateway identity probe inside ``_pre_exec_block`` ends in a kernel process
query that can wedge; before the deadline wrap, ``terminal_tool`` never returned and the
cron run held its slot forever.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

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


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Stub planning/env/approval/execution; returns the list of executions that happened."""
    calls: list[str] = []
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan())
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(
        terminal_module, "_run_approval_guards", lambda *_a, **_k: terminal_module._ApprovalVerdict(),
    )
    monkeypatch.setattr(
        terminal_module, "_run_foreground", lambda *_a, **_k: calls.append("foreground") or "foreground-ran",
    )
    return calls


def test_wedged_pre_execution_guard_returns_bounded_error_without_running(monkeypatch, stubbed_pipeline):
    """A stalled identity probe returns a retryable error within the deadline; the command does not run."""

    def _wedged_probe(*_a, **_k):
        time.sleep(1)

    monkeypatch.setattr(terminal_module, "_pre_exec_block", _wedged_probe)

    start = time.monotonic()
    result = json.loads(terminal_module.terminal_tool("echo ok"))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"pre-execution guard wedged terminal_tool for {elapsed:.2f}s"
    assert result["status"] == "error"
    assert "did not finish" in result["error"]
    assert stubbed_pipeline == [], "a guard with no verdict must not fail open into execution"


def test_completed_pre_execution_guard_verdicts_pass_through(monkeypatch, stubbed_pipeline):
    """A finished guard keeps its outcome: pass → execution, rejection → its own blocked result."""
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: None)
    assert terminal_module.terminal_tool("echo ok") == "foreground-ran"

    def _rejecting_probe(*_a, **_k):
        raise terminal_module._Rejected('{"status":"blocked"}')

    monkeypatch.setattr(terminal_module, "_pre_exec_block", _rejecting_probe)
    assert terminal_module.terminal_tool("echo ok") == '{"status":"blocked"}'
    assert stubbed_pipeline == ["foreground"]
