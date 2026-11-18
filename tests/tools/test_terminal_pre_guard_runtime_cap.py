"""The guard ceiling must be applied at the CALL SITE, not just in arithmetic (#741).

``test_terminal_pre_guard_cap.py`` reconstructs ``max(timeout, MIN)`` then ``min(_, MAX)``
in its own helper. That asserts the arithmetic is right, but it cannot notice if
``terminal_tool()`` stops applying the second clamp: delete the ``min(...)`` line at the
call site and every assertion in that file stays green while every 420s command goes back
to waiting 420s before the identical refusal.

This test drives the real ``terminal_tool()`` with the real defaults and observes the
deadline actually handed to ``run_bounded_sync`` -- the value the guard is bounded by.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tools.terminal_tool as terminal_module


def _plan(timeout: float) -> SimpleNamespace:
    return SimpleNamespace(
        config={},
        env_type="local",
        effective_task_id="pre-guard-runtime-cap-test",
        cwd="/tmp",
        effective_timeout=timeout,
        promoted_from_foreground_timeout=None,
    )


@pytest.fixture
def observed_deadline(monkeypatch):
    """Run the real sizing code; capture the timeout passed to the guard's deadline wrap."""
    seen: list[float] = []
    cmds: list[str] = []

    monkeypatch.setattr(
        terminal_module,
        "_plan_execution",
        lambda cmd, **_k: cmds.append(cmd) or _plan(420.0),
    )
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: None)
    monkeypatch.setattr(
        terminal_module,
        "_run_approval_guards",
        lambda *_a, **_k: terminal_module._ApprovalVerdict(),
    )
    monkeypatch.setattr(
        terminal_module, "_run_foreground", lambda *_a, **_k: "foreground-ran"
    )

    def _record(fn, timeout, **_k):
        seen.append(timeout)
        return SimpleNamespace(timed_out=False, value=None)

    import agent.deadline as deadline_module

    monkeypatch.setattr(deadline_module, "run_bounded_sync", _record)
    return seen


def test_long_command_hands_the_guard_the_capped_deadline_not_its_own_timeout(
    observed_deadline,
):
    """A 420s command must bound the guard at 90s -- the cap must be live at the call site."""
    assert terminal_module.terminal_tool("echo ok", timeout=420) == "foreground-ran"

    assert observed_deadline, "the guard was never run through run_bounded_sync"
    guard_deadline = observed_deadline[0]
    assert guard_deadline == terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S, (
        f"guard deadline for a 420s command was {guard_deadline}s, expected the {terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S}s ceiling: the call-site "
        "min() is missing, so a wedged probe costs the command's full timeout"
    )
    assert guard_deadline < 420, (
        "the command's own timeout leaked into the guard budget"
    )


def test_short_command_still_gets_the_floor_at_the_call_site(
    observed_deadline, monkeypatch
):
    """The ceiling must not swallow the 30s floor: a 1s command still gets MIN, not 1s."""
    monkeypatch.setattr(
        terminal_module, "_plan_execution", lambda cmd, **_k: _plan(1.0)
    )
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 30)

    assert terminal_module.terminal_tool("echo ok", timeout=1) == "foreground-ran"
    assert observed_deadline[0] == 30, (
        f"a 1s command handed the guard {observed_deadline[0]}s; the floor must still win below the ceiling"
    )


def test_wedged_guard_refusal_message_names_the_capped_budget(
    observed_deadline, monkeypatch
):
    """The refusal text must report the budget actually used, or the log lies about the cap."""

    def _timed_out(fn, timeout, **_k):
        return SimpleNamespace(timed_out=True, value=None)

    import agent.deadline as deadline_module

    monkeypatch.setattr(deadline_module, "run_bounded_sync", _timed_out)

    result = json.loads(terminal_module.terminal_tool("echo ok", timeout=420))
    assert result["status"] == "error"
    assert "within 90s" in result["error"], (
        "refusal reported a budget the guard did not use: {!r}".format(result["error"])
    )
