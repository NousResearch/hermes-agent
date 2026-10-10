"""The delegation wall cap must work for the deadline-EXEMPT tools.

``delegate_task`` is in ``_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS``, so ``timeout_s`` is ``None`` on its
sequential path. The wall cap reports a timeout anyway, which means the timeout branch of
``_run_sequential_call`` must not assume a per-call deadline exists - an ``assert timeout_s is not
None`` there fires precisely in the wedge this cap exists to prevent.
"""

import concurrent.futures
import time
import types

import pytest

from agent import tool_executor as te


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setattr(te, "_delegation_wall_cap", lambda fn: 0.05 if fn == "delegate_task" else None)


def _exempt(fn: str) -> bool:
    return fn in te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS


def test_delegate_task_is_still_deadline_exempt():
    """Guards the premise: if upstream ever un-exempts it, the cap's fallback path is dead code."""
    assert _exempt("delegate_task")


def test_delegate_task_resolves_no_per_call_deadline(capped):
    """With the setting on but no per-call timeout, the tool still reports a clean tool_timeout.

    Before the fix this path hit ``assert timeout_s is not None`` and raised AssertionError inside the
    timeout branch - converting a hang into a crash.
    """
    resolved = te._resolve_sequential_tool_timeout()
    assert resolved is None or isinstance(resolved, (int, float))

    agent = types.SimpleNamespace(
        _interrupt_requested=False,
        _tool_interrupt_reason=None,
        _touch_activity=lambda *a, **k: None,
    )
    gate = te._ConcurrentToolAuthorizationGate() if hasattr(te, "_ConcurrentToolAuthorizationGate") else None

    def _never():
        time.sleep(30)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_never)
        started = time.monotonic()
        state, result = te._poll_sequential_future(agent, future, "delegate_task", None, started, gate)
        elapsed = time.monotonic() - started

        assert state == "timeout", f"expected a timeout, got {state!r} after {elapsed:.1f}s"
        assert elapsed < 10, "the cap did not actually bound the wait"
    finally:
        executor.shutdown(wait=False)


def test_non_delegation_tools_get_no_cap(capped):
    """The cap must not leak onto unrelated sequential tools."""
    assert te._delegation_wall_cap("terminal") is None
    assert te._delegation_wall_cap("web_search") is None
