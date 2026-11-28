"""The guard ceiling must also bind the SECOND call site: execute_code (#741).

``test_terminal_pre_guard_runtime_cap.py`` pins ``terminal_tool()``. But the same
identity probe runs again in ``code_execution_tool`` for its lifecycle guard, and
that site sized its deadline as ``max(cell_timeout, MIN)`` with no ceiling at all.
With ``terminal.timeout: 180`` on this host and MIN=30, a wedged probe therefore
cost 180s on that path however tight the terminal cap was -- the class survived in
the sibling file.

Measured on 2026-11-18: refusal deadlines in ``~/.hermes/logs`` for the day were
60/180/240/300s, 36 of 42 above 90s. The 180s arm is exactly this call site's
arithmetic.

This drives the real ``execute_code`` entry point and asserts the deadline handed
to ``run_bounded_sync``, so deleting the ``min()`` at that call site fails here.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import agent.deadline as deadline_module
from tools.terminal_tool import (
    _PRE_EXEC_GUARD_MAX_TIMEOUT_S,
    _PRE_EXEC_GUARD_MIN_TIMEOUT_S,
)

mod = importlib.import_module("tools.code_execution_tool")

CAP = _PRE_EXEC_GUARD_MAX_TIMEOUT_S
FLOOR = _PRE_EXEC_GUARD_MIN_TIMEOUT_S


class _GuardRan(Exception):
    """Raised from the stubbed probe wrapper so the entry point stops at the guard."""


@pytest.fixture
def captured(monkeypatch):
    """Record the deadline execute_code hands its lifecycle probe, then bail out."""
    seen: list[float] = []

    def _record(fn, timeout, **_k):
        seen.append(timeout)
        raise _GuardRan

    monkeypatch.setattr(deadline_module, "run_bounded_sync", _record)
    # A big cell timeout is the point: it must NOT become the probe budget.
    monkeypatch.setattr(mod, "_load_config", lambda: {"timeout": 300})
    return seen


def _drive(code: str = "print(1)") -> str:
    try:
        return mod.execute_code(code=code)
    except _GuardRan:
        return ""


def test_probe_deadline_is_capped_below_the_cell_timeout(captured):
    """A 300s cell must bound its lifecycle probe at the 90s ceiling."""
    _drive()

    assert captured, "the lifecycle guard never ran through run_bounded_sync"
    deadline = captured[0]
    assert deadline == CAP, (
        "execute_code handed its lifecycle probe %ss; the %ss ceiling is not applied at "
        "this call site, so a wedged probe burns the whole cell timeout (#741)"
        % (deadline, CAP)
    )
    assert deadline < 300, "the cell timeout leaked into the probe budget"


def test_probe_deadline_keeps_the_floor(captured, monkeypatch):
    """The ceiling must not swallow the 30s floor for a tiny cell timeout."""
    monkeypatch.setattr(mod, "_load_config", lambda: {"timeout": 1})

    _drive()

    assert captured and captured[0] == FLOOR, (
        "a 1s cell handed the lifecycle probe %ss; the floor must still win"
        % (captured[0] if captured else "nothing")
    )


def test_refusal_names_the_capped_budget(monkeypatch):
    """The refusal text must report the budget actually used, or the log lies about the cap."""

    def _timed_out(fn, timeout, **_k):
        return SimpleNamespace(timed_out=True, value=None)

    monkeypatch.setattr(deadline_module, "run_bounded_sync", _timed_out)
    monkeypatch.setattr(mod, "_load_config", lambda: {"timeout": 300})

    out = mod.execute_code(code="print(1)")
    assert isinstance(out, str)
    assert "within %ss" % CAP in out, (
        "refusal reported a budget the probe did not use: %r" % out[:300]
    )
