"""CDPSupervisor reconnect budget: a supervisor that attached once must not retry a dead
endpoint forever (#114172). No real Chrome — ``websockets.connect`` is stubbed."""

from __future__ import annotations

import asyncio

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
    one final warning, and the supervisor leaves the registry — never an unbounded loop."""
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
        if dials <= budget + 1:
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

    assert dials == budget + 1  # attach + budgeted reconnects
    final = [r.getMessage() for r in caplog.records if "stopped after" in r.getMessage()]
    assert len(final) == 1 and f"{budget} failed reconnect" in final[0]
    assert supervisor.snapshot().active is False
    assert bs.SUPERVISOR_REGISTRY.get(supervisor.task_id) is None


def test_initial_connect_retries_through_cold_start_then_attaches(monkeypatch):
    """A cold browser may reject early dials, but a later CDP attach still starts the supervisor."""
    supervisor = bs.CDPSupervisor(task_id="cold-start", cdp_url="ws://127.0.0.1:9222")
    dials = 0

    async def connect(*_args, **_kwargs):
        nonlocal dials
        dials += 1
        if dials < bs.MAX_INITIAL_CONNECT_ATTEMPTS:
            raise ConnectionError("Chromium is still starting")
        return _ClosingWebSocket()

    async def _noop(*_a, **_k):
        pass

    async def _stop_after_attach():
        supervisor._stop_requested = True

    real_sleep = asyncio.sleep

    async def fast_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(supervisor, "_attach_initial_page", _noop)
    monkeypatch.setattr(supervisor, "_read_loop", _stop_after_attach)
    monkeypatch.setattr(bs.asyncio, "sleep", fast_sleep)

    asyncio.run(supervisor._run())

    assert dials == bs.MAX_INITIAL_CONNECT_ATTEMPTS
    assert supervisor._start_error is None
    assert supervisor._ready_event.is_set()


def test_initial_connect_failure_stops_after_cold_start_budget(monkeypatch):
    """An endpoint that never appears remains fatal for ``start()`` after the initial retry window."""
    supervisor = bs.CDPSupervisor(task_id="initial-failure", cdp_url="ws://127.0.0.1:9222")
    dials = 0

    async def connect(*_args, **_kwargs):
        nonlocal dials
        dials += 1
        raise ConnectionError("CDP endpoint is unavailable")

    real_sleep = asyncio.sleep

    async def fast_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(bs.asyncio, "sleep", fast_sleep)

    asyncio.run(supervisor._run())

    assert dials == bs.MAX_INITIAL_CONNECT_ATTEMPTS
    assert isinstance(supervisor._start_error, ConnectionError)
    assert supervisor._ready_event.is_set()
    assert supervisor.snapshot().active is False
