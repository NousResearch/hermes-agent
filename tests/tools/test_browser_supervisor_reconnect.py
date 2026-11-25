"""CDPSupervisor reconnect budget: a supervisor that attached once must not retry a dead
endpoint forever (#114172). No real Chrome — ``websockets.connect`` is stubbed."""

from __future__ import annotations

import asyncio
import time

import pytest
import websockets

from tools import browser_supervisor as bs


class _ClosingWebSocket:
    async def close(self):
        pass


@pytest.fixture
def unregister():
    task_ids: list[str] = []
    yield task_ids
    for task_id in task_ids:
        bs.SUPERVISOR_REGISTRY._pop(task_id)


def test_post_attach_reconnects_stop_at_budget_and_unregister(monkeypatch, unregister, caplog):
    """After the first attach, a dead endpoint gets MAX_POST_ATTACH_RECONNECT_FAILURES dials,
    one final warning, and the supervisor leaves the registry — never an unbounded loop.

    The first dial attaches and is closed at once: a connection that dies before
    RECONNECT_STABLE_AFTER_S never served anything, so it spends budget instead of resetting it.
    """
    supervisor = bs.CDPSupervisor(task_id="bounded-reconnect", cdp_url="ws://127.0.0.1:9222")
    bs.SUPERVISOR_REGISTRY._by_task[supervisor.task_id] = supervisor
    unregister.append(supervisor.task_id)
    budget = bs.MAX_POST_ATTACH_RECONNECT_FAILURES
    dials = 0

    async def connect(*_args, **_kwargs):
        nonlocal dials
        dials += 1
        if dials == 1:
            return _ClosingWebSocket()
        if dials <= budget:
            raise ConnectionError("CDP endpoint is gone")
        await asyncio.Event().wait()  # would hang forever: the loop must never get here

    async def _noop(*_a, **_k):
        pass

    real_sleep = asyncio.sleep

    async def fast_sleep(_delay):
        await real_sleep(0)  # yield so wait_for's deadline can fire on an unbounded loop

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(supervisor, "_attach_initial_page", _noop)
    monkeypatch.setattr(supervisor, "_read_loop", _noop)
    monkeypatch.setattr(bs.asyncio, "sleep", fast_sleep)

    with caplog.at_level("WARNING", logger="tools.browser_supervisor"):
        asyncio.run(asyncio.wait_for(supervisor._run(), timeout=2.0))

    assert dials == budget  # the attach that died instantly + the remaining failed dials
    final = [r.getMessage() for r in caplog.records if "stopped after" in r.getMessage()]
    assert len(final) == 1 and f"{budget} failed reconnect" in final[0]
    assert supervisor.snapshot().active is False
    assert bs.SUPERVISOR_REGISTRY.get(supervisor.task_id) is None


def test_initial_connect_failure_stays_fatal_for_start(monkeypatch):
    """The budget is post-attach only: a first-dial failure still propagates to ``start()``."""
    supervisor = bs.CDPSupervisor(task_id="initial-failure", cdp_url="ws://127.0.0.1:9222")

    async def connect(*_args, **_kwargs):
        raise ConnectionError("CDP endpoint is unavailable")

    monkeypatch.setattr(websockets, "connect", connect)

    asyncio.run(supervisor._run())

    assert isinstance(supervisor._start_error, ConnectionError)
    assert supervisor._ready_event.is_set()
    assert supervisor.snapshot().active is False


def test_stable_connection_resets_the_reconnect_budget(monkeypatch, unregister):
    """A connection that outlived RECONNECT_STABLE_AFTER_S is a real recovery: the next dead
    stretch gets the full budget instead of inheriting the previous count."""
    monkeypatch.setattr(bs, "RECONNECT_STABLE_AFTER_S", 0.0)
    supervisor = bs.CDPSupervisor(task_id="stable-reset", cdp_url="ws://127.0.0.1:9222")
    bs.SUPERVISOR_REGISTRY._by_task[supervisor.task_id] = supervisor
    unregister.append(supervisor.task_id)
    budget = bs.MAX_POST_ATTACH_RECONNECT_FAILURES
    stable_dials, dials = 2, 0

    async def connect(*_args, **_kwargs):
        nonlocal dials
        dials += 1
        if dials <= stable_dials:
            return _ClosingWebSocket()
        if dials <= stable_dials + budget:
            raise ConnectionError("CDP endpoint is gone")
        await asyncio.Event().wait()  # would hang forever: the loop must never get here

    async def _noop(*_a, **_k):
        pass

    real_sleep = asyncio.sleep

    async def fast_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(supervisor, "_attach_initial_page", _noop)
    monkeypatch.setattr(supervisor, "_read_loop", _noop)
    monkeypatch.setattr(bs.asyncio, "sleep", fast_sleep)

    asyncio.run(asyncio.wait_for(supervisor._run(), timeout=2.0))

    assert dials == stable_dials + budget
    assert bs.SUPERVISOR_REGISTRY.get(supervisor.task_id) is None


def test_stop_all_spends_one_shared_budget(monkeypatch):
    """Shutdown must not pay 5 s per supervisor: one budget covers the whole registry, and a
    supervisor that never got its turn is still marked stopped so its own loop ends."""
    registry = bs._SupervisorRegistry()

    class _SlowSupervisor:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self._stop_requested = False

        def stop(self, timeout: float = 5.0) -> None:
            self._stop_requested = True
            # The delay IS the behaviour under test: one slow stop must eat the shared budget.
            time.sleep(min(timeout, 0.3))  # health: allow HX013 -- a slow stop is the case under test

    supervisors = [_SlowSupervisor(f"slow-{n}") for n in range(4)]
    with registry._lock:
        for supervisor in supervisors:
            registry._by_task[supervisor.task_id] = supervisor

    started = time.monotonic()
    registry.stop_all(budget_s=0.1)
    elapsed = time.monotonic() - started

    assert elapsed < 0.9, elapsed
    assert all(supervisor._stop_requested for supervisor in supervisors)
    assert registry._by_task == {}
