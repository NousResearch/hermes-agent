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
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0)
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


def test_pre_execution_guard_on_the_deadline_worker_sees_the_tool_threads_interrupt(monkeypatch, stubbed_pipeline):
    """/stop keys on the tool thread's tid; the guard chain moved onto a worker must still see it."""
    from tools.interrupt import is_interrupted, set_interrupt

    seen: list[bool] = []
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: seen.append(is_interrupted()))

    set_interrupt(True)
    try:
        terminal_module.terminal_tool("echo ok")
    finally:
        set_interrupt(False)

    assert seen == [True], "guard on the deadline worker was blind to the tool thread's interrupt bit"
    assert is_interrupted() is False, "the tool thread's own interrupt view must not leak past the guard"


def test_guard_wait_is_capped_well_under_a_long_command_deadline(monkeypatch, stubbed_pipeline):
    """A wedged probe must not inherit a long command's whole deadline (#board-497: 60-400s spent
    per refused call, one abandoned thread each). The guard's share is capped; the refusal is the same."""
    seen: list[float] = []

    def _bounded(fn, timeout, *, label="operation", **_k):
        seen.append(timeout)
        fn()
        return SimpleNamespace(timed_out=False)

    monkeypatch.setattr(
        terminal_module, "_plan_execution", lambda *_a, **_k: _plan(timeout=600.0),
    )
    monkeypatch.setattr("agent.deadline.run_bounded_sync", _bounded)
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: None)

    terminal_module.terminal_tool("echo ok")

    assert seen == [terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S]
    assert seen[0] <= 30, f"guard inherited a {seen[0]}s wait from the command deadline"


def test_supervised_gateway_verdict_is_probed_once_per_process(monkeypatch):
    """The identity probe is invariant for a process, and every terminal command used to pay for
    it: when one probe wedges, each call parks another guard thread in the same open() (#board-497)."""
    import os

    import gateway.restart
    import gateway.status
    import tools.process_registry as registry

    probe_calls: list[int] = []

    def _counted_running_pid(*_a, **_k):
        probe_calls.append(1)
        return os.getpid()

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
    monkeypatch.setattr(gateway.restart, "is_supervised_gateway_launch", lambda *_a, **_k: True)
    monkeypatch.setattr(gateway.status, "get_running_pid", _counted_running_pid)
    registry._reset_supervised_gateway_verdict_cache()

    assert registry._is_supervised_gateway_process() is True
    assert registry._is_supervised_gateway_process() is True
    assert len(probe_calls) == 1, "the identity probe re-ran on a cached verdict"

    # A changed marker is a different identity question (re-exec, tests, launcher shapes): re-probe.
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    assert registry._is_supervised_gateway_process() is True
    assert len(probe_calls) == 2

    # An exception must never be cached as "not supervised".
    monkeypatch.setattr(
        gateway.status, "get_running_pid",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("pid file unreadable")),
    )
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "2")
    assert registry._is_supervised_gateway_process() is False
    monkeypatch.setattr(gateway.status, "get_running_pid", _counted_running_pid)
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "3")
    assert registry._is_supervised_gateway_process() is True
    registry._reset_supervised_gateway_verdict_cache()
