"""TUI-only mid-response stop: stdio backpressure must never block the
provider streaming thread.

Reproduction
------------
The TUI gateway streams ``message.delta`` frames to the Ink client over a
stdio pipe.  When the client stops draining stdout (heavy re-render, React
reconciliation pause, frozen terminal), the OS pipe buffer fills and a
synchronous ``stream.write()`` blocks the thread that is consuming the
provider stream — the interrupt check never runs and the turn appears hung
with no way to cancel.

``test_sync_stdio_transport_blocks_on_wedged_sink`` reproduces that stall
deterministically with a blocked sink.  The fix routes TUI frames through
:class:`~tui_gateway.transport.BufferedStreamWriter` — a bounded,
coalescing writer drained by a dedicated thread — so the streaming
producer never blocks, ordering is preserved, control/completion frames
survive backpressure, and a wedged sink degrades to a bounded wait +
clean peer-gone signal instead of an agent deadlock.
"""

import json
import logging
import threading
import time

import pytest

from tui_gateway.transport import (
    BufferedStreamWriter,
    StdioTransport,
)

# ── Deterministic sinks ────────────────────────────────────────────────


class BlockingSink:
    """A stream whose ``write()`` blocks until :meth:`release` — a
    deterministic stand-in for a full stdout pipe the Ink client isn't
    draining.  Writes that happen after release are recorded in order.

    ``started`` is set the moment the first ``write()`` is entered (before
    it blocks), so tests can wait until the writer is provably wedged
    inside the sink.
    """

    def __init__(self):
        self._release = threading.Event()
        self.started = threading.Event()
        self.lines = []
        self._lock = threading.Lock()

    def block(self):
        self._release.clear()

    def release(self):
        self._release.set()

    def write(self, s):
        self.started.set()
        # Safety bound only — tests release the sink before this expires.
        self._release.wait(timeout=30)
        with self._lock:
            self.lines.append(s)

    def flush(self):
        pass


class RecordingSink:
    """A non-blocking sink that records every write in order."""

    def __init__(self):
        self.lines = []
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            self.lines.append(s)

    def flush(self):
        pass


def _make_writer(sink, **kwargs):
    inner = StdioTransport(lambda: sink, threading.Lock())
    return BufferedStreamWriter(inner, **kwargs)


def delta_frame(text, sid="s1"):
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "message.delta",
            "session_id": sid,
            "payload": {"text": text},
        },
    }


def control_frame(event_type, payload=None, sid="s1"):
    params = {"type": event_type, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def frame_types(sink):
    return [json.loads(line)["params"]["type"] for line in sink.lines]


# ── Root-cause reproduction ────────────────────────────────────────────


def test_sync_stdio_transport_blocks_on_wedged_sink():
    """A synchronous write to a full pipe blocks the producer thread.

    This is the exact stall behind the TUI mid-response stop: the provider
    streaming thread blocks inside StdioTransport.write and never reaches
    the interrupt check.  BufferedStreamWriter exists to keep that thread
    off the sink entirely.
    """
    sink = BlockingSink()
    transport = StdioTransport(lambda: sink, threading.Lock())

    def producer():
        transport.write(delta_frame("tok"))

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive(), "sync write must block on a wedged sink"
    sink.release()
    t.join(timeout=1.0)
    assert not t.is_alive()


# ── Non-blocking push ──────────────────────────────────────────────────


def test_delta_push_never_blocks_while_sink_wedged():
    """The provider streaming thread's delta push must return instantly even
    when the writer is blocked on a full pipe."""
    sink = BlockingSink()  # wedged: writes block until released
    writer = _make_writer(sink)
    try:
        start = time.monotonic()
        for i in range(500):
            assert writer.write(delta_frame(f"tok-{i}")) is True
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"delta push blocked: {elapsed:.3f}s for 500 pushes"
    finally:
        sink.release()
        writer.close()
    # Everything coalesced into the pending batch reaches the sink once the
    # writer can drain — no delta silently lost in the healthy path.
    assert len(sink.lines) == 500


# ── Ordering ───────────────────────────────────────────────────────────


def test_control_frame_never_overtakes_queued_deltas():
    """message.complete (and any control frame) must land after the deltas
    that preceded it — even when the sink was wedged in between."""
    sink = BlockingSink()
    writer = _make_writer(sink)
    try:
        writer.write(delta_frame("a"))
        writer.write(delta_frame("b"))
        assert writer.write(control_frame("message.complete", {"text": "ab"})) is True
        # Let the writer consume the batch and wedge on the sink so the
        # assertion exercises the in-flight queue order, not the close-drain.
        time.sleep(0.05)
        assert sink.lines == []  # wedged — nothing hit the sink yet
    finally:
        sink.release()
        writer.close()
    assert frame_types(sink) == [
        "message.delta",
        "message.delta",
        "message.complete",
    ]
    texts = [json.loads(line)["params"]["payload"]["text"] for line in sink.lines]
    assert texts == ["a", "b", "ab"]


def test_rpc_response_ordered_after_pending_deltas():
    """A non-event RPC response (e.g. session.status poll during a turn)
    must not overtake deltas still in the coalescing buffer."""
    sink = BlockingSink()
    writer = _make_writer(sink)
    try:
        writer.write(delta_frame("x"))
        writer.write(delta_frame("y"))
        resp = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
        assert writer.write(resp) is True
        time.sleep(0.05)
    finally:
        sink.release()
        writer.close()
    lines = sink.lines
    assert len(lines) == 3
    assert json.loads(lines[0])["params"]["type"] == "message.delta"
    assert json.loads(lines[1])["params"]["type"] == "message.delta"
    assert json.loads(lines[2]).get("id") == 7


# ── Bounded backpressure ───────────────────────────────────────────────


def test_control_push_is_bounded_when_writer_wedged():
    """A control-frame push waits at most control_push_timeout_s for queue
    space, then signals a dead peer instead of deadlocking the agent."""
    sink = BlockingSink()
    writer = _make_writer(
        sink, queue_maxsize=2, control_push_timeout_s=0.25, coalesce_s=0.01
    )
    try:
        # Push one control batch and let the writer dequeue it and wedge on
        # the blocked sink — it cannot dequeue anything further.
        assert writer.write(control_frame("tool.start", {"name": "t0"})) is True
        time.sleep(0.05)
        assert sink.lines == []  # writer is wedged on the sink, nothing drained
        # Fill the bounded queue behind the wedged writer.
        assert writer.write(control_frame("tool.start", {"name": "t1"})) is True
        assert writer.write(control_frame("tool.start", {"name": "t2"})) is True
        start = time.monotonic()
        ok = writer.write(control_frame("tool.start", {"name": "t3"}))
        elapsed = time.monotonic() - start
        assert ok is False, "wedged control push must eventually report peer dead"
        assert 0.2 <= elapsed < 1.0, f"control push not bounded: {elapsed:.3f}s"
    finally:
        sink.release()
        writer.close()


def test_deltas_dropped_under_backpressure_but_control_survives():
    """Deltas are droppable (display-only; message.complete is canonical),
    so the pending batch stays memory-bounded and a control frame still
    enqueues and lands after the surviving deltas."""
    sink = BlockingSink()
    writer = _make_writer(
        sink, queue_maxsize=4, max_pending_deltas=8, coalesce_s=0.01
    )
    try:
        for i in range(200):
            writer.write(delta_frame(f"tok-{i}"))
        with writer._pending_lock:
            assert len(writer._pending) <= 8, "pending delta batch must stay bounded"
        assert writer.write(control_frame("message.complete", {"text": "done"})) is True
        time.sleep(0.05)
    finally:
        sink.release()
        writer.close()
    types = frame_types(sink)
    assert types[-1] == "message.complete", "canonical complete must never be dropped"
    deltas = [t for t in types if t == "message.delta"]
    assert len(deltas) < 200, "overflow deltas must be dropped, not queued forever"


# ── Writer lifecycle ───────────────────────────────────────────────────


def test_close_drains_pending_and_joins_writer():
    """close() flushes coalesced deltas that no control frame carried, and
    joins the writer thread so it cannot leak."""
    sink = RecordingSink()
    writer = _make_writer(sink, coalesce_s=0.05)
    writer.write(delta_frame("a"))
    writer.write(delta_frame("b"))
    writer.write(delta_frame("c"))
    writer.close()
    assert writer._thread is None or not writer._thread.is_alive()
    assert len(sink.lines) == 3


def test_writer_thread_does_not_leak():
    sink = RecordingSink()
    writer = _make_writer(sink)
    writer.write(delta_frame("a"))
    writer.write(control_frame("message.complete", {"text": "a"}))
    writer.close()
    assert not any(t.name == "tui-stdio-writer" for t in threading.enumerate())


def test_writer_thread_is_lazy_and_daemon():
    """No thread is spawned until the first frame — importing/creating the
    writer in a non-TUI context is side-effect free, and the writer thread
    is daemon so a wedged sink can never strand interpreter shutdown."""
    sink = RecordingSink()
    writer = _make_writer(sink)
    assert writer._thread is None
    writer.write(delta_frame("a"))
    assert writer._thread is not None
    assert writer._thread.daemon is True
    writer.close()


# ── Wiring ─────────────────────────────────────────────────────────────


def test_entry_installs_buffered_writer():
    """The TUI entry point wraps the stdio transport in the non-blocking
    writer; the module default stays the plain synchronous StdioTransport
    so non-TUI transports and stdout-monkeypatching tests are unchanged."""
    from tui_gateway import entry as entry_mod
    from tui_gateway import server

    original = server._stdio_transport
    try:
        entry_mod._install_stream_writer()
        wrapped = server._stdio_transport
        # Check against entry's own class reference: test_protocol reloads
        # tui_gateway.transport in-process, which can replace the module's
        # class object without affecting already-created instances.
        assert isinstance(wrapped, entry_mod.BufferedStreamWriter)
        assert wrapped._inner is original
    finally:
        server._stdio_transport = original


# ── Control ordering fence ────────────────────────────────────────────
# Regression: a control frame drains the pending delta batch, then blocks
# on queue.put because the queue is full.  A delta produced *after* that
# control push began must NOT be opportunistically flushed to the sink
# before the control — the writer holds its delta flush while any control
# batch is claimed-but-unwritten (`_control_claimed`).


def test_later_delta_never_reaches_sink_before_waiting_control():
    """Deterministic concurrent regression for the control/delta race.

    Setup: queue_maxsize=2, writer wedged on the blocked sink.  Push A
    (writer drains it and wedges), then fill the queue with B and C.  Push
    D from another thread — it claims the (empty) pending batch, then must
    block on queue.put because the queue is full and the writer can't
    drain.  Only once D's claim is registered (``_control_claimed == 4``)
    push a later delta.  Release the sink; the delta must land AFTER D.
    """
    sink = BlockingSink()
    writer = _make_writer(
        sink, queue_maxsize=2, control_push_timeout_s=5.0, coalesce_s=0.01
    )
    t = None
    try:
        # A drains and the writer wedges inside the sink.
        assert writer.write(control_frame("tool.start", {"name": "A"})) is True
        assert sink.started.wait(1.0), "writer never wedged on the sink"
        # Fill the queue: B and C occupy both slots.
        assert writer.write(control_frame("tool.start", {"name": "B"})) is True
        assert writer.write(control_frame("tool.start", {"name": "C"})) is True
        assert writer._queue.qsize() == 2, "queue must be full"

        # D claims pending then blocks on queue.put (full + writer wedged).
        push_result = {}

        def push_d():
            push_result["ok"] = writer.write(control_frame("tool.start", {"name": "D"}))

        t = threading.Thread(target=push_d, daemon=True)
        t.start()
        # Deterministic rendezvous: D has claimed the pending batch (under
        # _pending_lock) and is now blocked in put — with the queue full
        # and the writer wedged it cannot complete until the sink drains.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with writer._pending_lock:
                if writer._control_claimed == 4:
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("control push D never claimed the pending batch")
        # The later delta — must NOT reach the sink before D.
        assert writer.write(delta_frame("late")) is True
        time.sleep(0.02)  # give a buggy writer time to misorder
        assert sink.lines == [], "nothing may reach the sink while wedged"
    finally:
        sink.release()
        writer.close()
        # D must have completed its (bounded) push after the sink drained.
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
    # Wire order: A, B, C, D — then the delta, never before D.
    names = [
        json.loads(line)["params"].get("payload", {}).get("name") for line in sink.lines
    ]
    assert names[:-1] == ["A", "B", "C", "D"], f"order violated: {names}"
    assert names[-1] is None, f"delta must be last, got: {names}"


# ── Close without STOP ────────────────────────────────────────────────
# Regression: close() cannot enqueue _STOP when the queue is full; once
# the sink recovers and queued work drains, the closed writer must exit on
# its own instead of waiting forever for a STOP that never arrives.


def test_closed_writer_exits_after_drain_without_stop():
    """close() with a full queue: _STOP can't be enqueued, but the writer
    exits once queued work drains after the sink recovers."""
    sink = BlockingSink()
    writer = _make_writer(
        sink, queue_maxsize=2, close_join_timeout_s=0.05, coalesce_s=0.01
    )
    try:
        assert writer.write(control_frame("tool.start", {"name": "A"})) is True
        assert sink.started.wait(1.0), "writer never wedged on the sink"
        assert writer.write(control_frame("tool.start", {"name": "B"})) is True
        assert writer.write(control_frame("tool.start", {"name": "C"})) is True
        assert writer._queue.qsize() == 2, "queue must be full"
        # close() with a full queue: the _STOP enqueue times out (queue
        # still holds B and C) and the join times out (writer still wedged
        # on A).  close() returns without ever enqueueing _STOP.
        writer.close()
        assert writer._queue.qsize() == 2, "_STOP must not have been enqueued"
        assert writer._thread is not None and writer._thread.is_alive()
    finally:
        # Recover the sink ONLY — no second close() here: the writer must
        # exit on its own, never seeing a _STOP sentinel.
        sink.release()
    # Sink recovers: writer drains A, B, C, then must exit on its own.
    thread = writer._thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "closed writer must exit after draining without _STOP"
    names = [
        json.loads(line)["params"].get("payload", {}).get("name") for line in sink.lines
    ]
    assert names == ["A", "B", "C"], f"queued work not drained: {names}"
    writer.close()  # cleanup only; the writer thread is already gone


# ── Inner exception visibility ────────────────────────────────────────
# A peer-gone inner error is a clean disconnect (write returns False, no
# crash).  A programming/host error (non-JSON-safe payload, ENOSPC, ...)
# must NOT masquerade as a clean disconnect: it is logged with its
# traceback and re-raised so the gateway's thread panic hook records it in
# the crash log — the same visibility StdioTransport.write documents.


class RaisingTransport:
    """Minimal Transport whose write() raises a fixed exception."""

    def __init__(self, exc):
        self._exc = exc

    def write(self, obj):
        raise self._exc

    def close(self):
        pass


def test_peer_gone_inner_exception_is_clean_disconnect():
    """BrokenPipeError from the inner sink is peer-gone: the writer stops,
    the transport reports dead, and no exception escapes to the caller."""
    inner = RaisingTransport(BrokenPipeError("Broken pipe"))
    writer = BufferedStreamWriter(inner, coalesce_s=0.01)
    assert writer.write(delta_frame("a")) is True  # enqueue is non-blocking
    writer._thread.join(timeout=1.0)
    assert not writer._thread.is_alive()
    assert writer._closed is True
    assert writer.write(delta_frame("b")) is False


def test_programming_error_is_logged_and_re_raised(caplog):
    """A non-peer-gone inner exception (e.g. a non-JSON-serialisable
    payload) is NOT swallowed as a clean disconnect: it is logged with its
    traceback and re-raised so the crash-log/panic-hook path records it."""
    inner = RaisingTransport(TypeError("Object of type set is not JSON serializable"))
    writer = BufferedStreamWriter(inner, coalesce_s=0.01)
    with caplog.at_level(logging.ERROR, logger="tui_gateway.transport"):
        assert writer.write(delta_frame("a")) is True
        writer._thread.join(timeout=1.0)
    assert not writer._thread.is_alive()
    assert writer._closed is True
    assert writer.write(delta_frame("b")) is False
    assert any(
        r.levelno >= logging.ERROR and "inner write failed" in r.getMessage()
        for r in caplog.records
    ), "programming error must be logged, not silently peer-gone"
