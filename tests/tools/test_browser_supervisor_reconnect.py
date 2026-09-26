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


class TestColdStartReadinessPoll:
    """Regression for #121432.

    A freshly launched browser has not opened its DevTools listener yet, so the first
    WebSocket handshake loses the race and burns a retry. The contract: wait on the CDP
    HTTP endpoint before the first dial, and spend the retry budget on real failures only.
    """

    def test_first_refusal_waits_for_the_endpoint_before_spending_the_budget(self, monkeypatch):
        """A refused dial on a never-yet-ready supervisor must wait on the CDP HTTP endpoint.

        The point of the fix is that the refusal is treated as "the browser is still starting"
        and answered with a wait, rather than being counted against the retry budget and
        logged as a transient network failure.
        """
        supervisor = bs.CDPSupervisor(task_id="cold-start", cdp_url="ws://127.0.0.1:9222")
        order: list[str] = []
        ready_calls = 0

        def probe(_url, timeout=1.0):
            nonlocal ready_calls
            ready_calls += 1
            order.append("ready")
            return True

        async def connect(*_args, **_kwargs):
            order.append("dial")
            raise ConnectionRefusedError(111, "Connect call failed")

        monkeypatch.setattr(bs, "_is_browser_debug_ready", probe)
        monkeypatch.setattr(websockets, "connect", connect)
        monkeypatch.setattr(bs, "CDP_READY_POLL_INTERVAL_S", 0.01)
        monkeypatch.setattr(bs, "CDP_READY_POLL_TIMEOUT_S", 1.0)

        asyncio.run(asyncio.wait_for(supervisor._run(), timeout=5.0))

        assert ready_calls >= 1, "the refused dial never waited on the CDP endpoint"
        assert order[0] == "dial" and order[1] == "ready", (
            "a refused dial must be followed by a readiness wait (%r)" % order[:3]
        )
        # The wait is what makes the endpoint usable; the ladder still reports the failure.
        assert isinstance(supervisor._start_error, ConnectionRefusedError)

    def test_non_refusal_failure_does_not_wait_for_readiness(self, monkeypatch):
        """Only a refusal means 'not up yet'. An auth/TLS failure must not be papered over
        with a readiness wait that only adds latency."""
        supervisor = bs.CDPSupervisor(task_id="auth-failure", cdp_url="ws://127.0.0.1:9222")
        ready_calls = 0

        def probe(_url, timeout=1.0):
            nonlocal ready_calls
            ready_calls += 1
            return True

        async def connect(*_args, **_kwargs):
            raise OSError("handshake failed: 401 unauthorized")

        monkeypatch.setattr(bs, "_is_browser_debug_ready", probe)
        monkeypatch.setattr(websockets, "connect", connect)

        asyncio.run(asyncio.wait_for(supervisor._run(), timeout=10.0))

        assert ready_calls == 0, "a non-refusal failure waited on readiness for no reason"
        assert supervisor._start_error is not None

    def test_readiness_probe_derives_the_http_url_from_the_ws_url(self):
        """A ws:// or wss:// endpoint must be probed over http(s) on its own origin."""
        import asyncio as _asyncio

        seen: list[str] = []

        def fake_ready(url, timeout=1.0):
            seen.append(url)
            return True

        original = bs._is_browser_debug_ready
        bs._is_browser_debug_ready = fake_ready
        try:
            _asyncio.run(bs._await_cdp_endpoint_ready("ws://127.0.0.1:9222/devtools/browser/abc"))
            _asyncio.run(bs._await_cdp_endpoint_ready("wss://cdp.example/devtools/browser/xyz"))
        finally:
            bs._is_browser_debug_ready = original

        assert seen == ["http://127.0.0.1:9222", "https://cdp.example"], seen

    def test_probe_gives_up_and_leaves_the_ladder_in_charge(self, monkeypatch):
        """An endpoint that never answers the HTTP probe must not hang the supervisor: the
        existing WebSocket ladder still runs, and the budget still bounds it."""
        supervisor = bs.CDPSupervisor(task_id="never-ready", cdp_url="ws://127.0.0.1:9222")
        dials = 0

        def never_ready(_url, timeout=1.0):
            return False

        async def connect(*_args, **_kwargs):
            nonlocal dials
            dials += 1
            raise ConnectionRefusedError(111, "Connect call failed")

        monkeypatch.setattr(bs, "_is_browser_debug_ready", never_ready)
        monkeypatch.setattr(bs, "CDP_READY_POLL_INTERVAL_S", 0.01)
        monkeypatch.setattr(bs, "CDP_READY_POLL_TIMEOUT_S", 0.05)
        monkeypatch.setattr(websockets, "connect", connect)

        asyncio.run(asyncio.wait_for(supervisor._run(), timeout=10.0))

        assert dials >= 1, "the supervisor never attempted a dial"
        assert isinstance(supervisor._start_error, ConnectionRefusedError)
