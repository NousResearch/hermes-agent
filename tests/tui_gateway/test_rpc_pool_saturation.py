"""LONG-handler pool saturation fails fast instead of queueing behind stuck workers (#132546).

A stuck worker cannot be killed; once every ``tui-rpc`` worker is wedged, an enqueued LONG
handler only ever dies as the client's 30-second timeout. The saturation guard reserves one
semaphore slot per in-flight LONG handler and fails fast with a retryable busy error (4032)
when none is free. Fast handlers run inline and never consume a slot.
"""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import pytest


@pytest.fixture
def runtime(monkeypatch):
    from tui_gateway import server
    from hermes_cli import backend_retirement

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    # Match the stubbed one-worker pool below: a single slot, so one in-flight
    # LONG handler is enough to saturate.
    monkeypatch.setattr(server, "_rpc_pool_slots", threading.Semaphore(1))
    return server, fence


class Transport:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


def _blocking_long_method(server, monkeypatch, name, entered, release):
    def handler(rid, params):
        entered.set()
        assert release.wait(10)
        return server._ok(rid, {"done": True})

    monkeypatch.setitem(server._methods, name, handler)
    monkeypatch.setattr(server, "_LONG_HANDLERS", {name})


def test_saturated_pool_fails_fast_with_retryable_busy_error(runtime, monkeypatch):
    """The (N+1)-th LONG handler gets an immediate 4032 instead of queueing for 30s."""
    server, _fence = runtime
    entered, release = threading.Event(), threading.Event()
    _blocking_long_method(server, monkeypatch, "test.long", entered, release)
    transport = Transport()

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        # First request takes the only worker and slot, then blocks inside the handler.
        assert server.dispatch({"id": "a", "method": "test.long"}, transport) is None
        assert entered.wait(10)

        busy = server.dispatch({"id": "b", "method": "test.long"}, transport)
        assert isinstance(busy, dict), (
            "saturated dispatch must answer inline, not enqueue"
        )
        assert busy["id"] == "b"
        assert busy["error"]["code"] == 4032
        assert "retry" in busy["error"]["message"].lower()

        release.set()
        assert transport.frames.get(timeout=10)["result"] == {"done": True}


def test_slot_is_released_when_the_handler_finishes(runtime, monkeypatch):
    """After a handler completes, its slot is freed and the next LONG handler runs."""
    server, _fence = runtime
    entered, release = threading.Event(), threading.Event()
    _blocking_long_method(server, monkeypatch, "test.long", entered, release)
    transport = Transport()

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        assert server.dispatch({"id": "a", "method": "test.long"}, transport) is None
        assert entered.wait(10)
        release.set()
        assert transport.frames.get(timeout=10)["result"] == {"done": True}

        entered.clear()
        release.clear()
        assert server.dispatch({"id": "b", "method": "test.long"}, transport) is None
        assert entered.wait(10)
        release.set()
        assert transport.frames.get(timeout=10)["result"] == {"done": True}


def test_slot_is_released_when_the_pool_rejects_the_submission(runtime, monkeypatch):
    """A submit that raises must give the slot (and the retirement reservation) back."""
    server, fence = runtime
    entered, release = threading.Event(), threading.Event()
    _blocking_long_method(server, monkeypatch, "test.long", entered, release)
    transport = Transport()

    class BrokenPool:
        def submit(self, fn):
            raise RuntimeError("pool is shutting down")

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        assert server.dispatch({"id": "a", "method": "test.long"}, transport) is None
        assert entered.wait(10)
        release.set()
        assert transport.frames.get(timeout=10)["result"] == {"done": True}
        # Slot is free again; the next dispatch gets as far as BrokenPool.submit.

        monkeypatch.setattr(server, "_pool", BrokenPool())
        with pytest.raises(RuntimeError):
            server.dispatch({"id": "b", "method": "test.long"}, transport)

        # The failed submit freed the slot, so the same request reruns on the real pool.
        monkeypatch.setattr(server, "_pool", pool)
        entered.clear()
        assert server.dispatch({"id": "b", "method": "test.long"}, transport) is None
        assert entered.wait(10)
        release.set()
        assert transport.frames.get(timeout=10)["result"] == {"done": True}
    assert fence.prepare()["idle"] is True


def test_fast_handlers_never_consume_a_slot(runtime, monkeypatch):
    """Inline (non-LONG) methods answer normally while every pool slot is held."""
    server, _fence = runtime
    entered, release = threading.Event(), threading.Event()
    _blocking_long_method(server, monkeypatch, "test.long", entered, release)
    transport = Transport()

    def fast(rid, params):
        return server._ok(rid, {"fast": True})

    monkeypatch.setitem(server._methods, "test.fast", fast)

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        assert server.dispatch({"id": "a", "method": "test.long"}, transport) is None
        assert entered.wait(10)

        # The only slot is held by the blocked LONG handler…
        busy = server.dispatch({"id": "b", "method": "test.long"}, transport)
        assert busy["error"]["code"] == 4032
        # …yet a fast method still answers inline, unaffected.
        assert server.dispatch({"id": "c", "method": "test.fast"}, transport) == {
            "jsonrpc": "2.0",
            "id": "c",
            "result": {"fast": True},
        }

        release.set()
        assert transport.frames.get(timeout=10)["result"] == {"done": True}
