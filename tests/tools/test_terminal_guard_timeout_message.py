"""The pre-exec guard's timeout message must not name a suspect (board #826).

``terminal_tool``'s deadline wraps the whole of ``_pre_exec_block``: the gateway-lifecycle
scan (which takes the SQLite connection-lifecycle lock for every script the command
EXECUTES), the workdir validation, and the self-repo guard. A timeout from that wrap is
therefore stage-agnostic, and the message has to be too.

It previously read "(process-identity probe wedged)" -- naming the one stage measurement
rules out. On this host the entire identity chain renders in 0.28s (psutil create_time
0.10s, cmdline 0.11s, `_is_supervised_gateway_process` 0.68s, `get_running_pid` 0.28s),
while the reproducible 180-420s block is a raw SQLite reader parked under an unbounded
connection-lifecycle lock. Five remediation passes read that label, re-timed psutil, and
reported "root cause unaddressed" without ever looking at the lock. The label is the
reason the defect stayed unfound, so it gets a test of its own.

If a future change DOES make the deadline wrap only one stage, this test is what forces
the message to be re-justified rather than silently re-narrowed.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import tools.terminal_tool as terminal_module

# Every subsystem the deadline can actually be waiting on. Naming any one of them in a
# stage-agnostic timeout message is the defect.
STAGE_NAMES = (
    "process-identity",
    "identity probe",
    "psutil",
    "create_time",
)


def _plan(timeout: float = 0.05) -> SimpleNamespace:
    return SimpleNamespace(
        config={},
        env_type="local",
        effective_task_id="guard-timeout-message-826",
        cwd="/tmp",
        effective_timeout=timeout,
        promoted_from_foreground_timeout=None,
    )


@pytest.fixture
def wedged_guard(monkeypatch):
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0)
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan())
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(
        terminal_module, "_run_approval_guards", lambda *_a, **_k: terminal_module._ApprovalVerdict(),
    )
    monkeypatch.setattr(terminal_module, "_run_foreground", lambda *_a, **_k: pytest.fail("command ran without a verdict"))

    def _stall(*_a, **_k):
        time.sleep(1)

    monkeypatch.setattr(terminal_module, "_pre_exec_block", _stall)
    return None


def test_timeout_message_does_not_blame_a_single_subsystem(wedged_guard):
    """The refusal must be stage-agnostic: the wrap cannot tell which check stalled."""
    result = json.loads(terminal_module.terminal_tool("echo ok"))
    error = result["error"]
    assert result["status"] == "error"

    for name in STAGE_NAMES:
        assert name not in error.lower(), (
            "the guard timeout message names %r, but the deadline wraps the lifecycle scan, "
            "the workdir validation and the self-repo guard alike -- it cannot know which "
            "one stalled (board #826). Message was: %r" % (name, error)
        )


def test_timeout_message_still_says_nothing_ran(wedged_guard):
    """Stage-agnostic must not become uninformative: the caller still learns both facts."""
    result = json.loads(terminal_module.terminal_tool("echo ok"))
    error = result["error"]
    assert "did not finish" in error, "the message no longer says the guard timed out"
    assert "was not run" in error, "the message no longer says the command did not execute"


def test_message_does_not_assert_a_stage_the_deadline_cannot_distinguish():
    """Structural pin: the error is built from one f-string, so the stage list is auditable."""
    import inspect

    src = inspect.getsource(terminal_module.terminal_tool)
    assert "no verdict from the pre-exec checks" in src, (
        "the stage-agnostic wording is gone; if the deadline was narrowed to one stage, "
        "update this test and the message together (board #826)"
    )
