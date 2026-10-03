"""Tests for tui_gateway.event_replay — per-session event seq + replay ring."""

import threading

import pytest

from tui_gateway import event_replay
from tui_gateway.event_replay import (
    latest_seq,
    reset_replay_state,
    events_since,
    replay_stats,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_replay_state()
    yield
    reset_replay_state()


def _frame(sid, etype="message.delta"):
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": etype, "session_id": sid, "payload": {}},
    }


def test_stamp_adds_monotonic_seq_per_session():
    f1 = _frame("s1")
    f2 = _frame("s1")
    other = _frame("s2")

    event_replay._stamp_event(f1)
    event_replay._stamp_event(other)
    event_replay._stamp_event(f2)

    assert f1["params"]["seq"] == 1
    assert f2["params"]["seq"] == 2  # per-session counter, unaffected by s2
    assert other["params"]["seq"] == 1


def test_stamp_ignores_non_event_and_sessionless_frames():
    rpc = {"jsonrpc": "2.0", "id": 1, "result": {}}
    no_sid = {"jsonrpc": "2.0", "method": "event", "params": {"type": "skin.changed"}}

    event_replay._stamp_event(rpc)
    event_replay._stamp_event(no_sid)

    assert "seq" not in rpc
    assert "seq" not in no_sid["params"]
    assert replay_stats()["events"] == 0


def test_events_since_returns_only_newer_frames_in_order():
    frames = [_frame("s1") for _ in range(5)]
    for f in frames:
        event_replay._stamp_event(f)

    got = events_since("s1", 3)
    assert [e["seq"] for e in got] == [4, 5]
    assert events_since("s1", 0) == [f["params"] for f in frames]
    assert events_since("s1", 99) == []
    assert latest_seq("s1") == 5


def test_events_since_returns_client_dispatchable_event_objects():
    """Cross-language contract: the client's replay loop dispatches an element
    only when it has a TOP-LEVEL ``type`` (json-rpc-gateway.ts fetchReplay:
    ``if (!event?.type) continue``). Returning full JSON-RPC envelopes here
    makes every replayed event silently droppable — the original #94219 bug.
    """
    event_replay._stamp_event(_frame("s1"))
    (event,) = events_since("s1", 0)

    # Bare event object, not an envelope.
    assert event["type"] == "message.delta"
    assert event["session_id"] == "s1"
    assert event["seq"] == 1
    assert "jsonrpc" not in event
    assert "method" not in event
    assert "params" not in event


def test_unknown_session_returns_empty():
    assert events_since("nope", 0) == []
    assert latest_seq("nope") == 0


def test_ring_buffer_is_bounded():
    for i in range(event_replay._REPLAY_BUFFER_MAX + 50):
        event_replay._stamp_event(_frame("s1"))

    stats = replay_stats()
    assert stats["events"] == event_replay._REPLAY_BUFFER_MAX
    # Oldest evicted: last_seen=0 must report truncation via the RPC contract.
    buf = event_replay._replay_buffers["s1"]
    assert buf[0][0] > 1


def test_session_count_bounded_with_fifo_eviction():
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 10):
        event_replay._stamp_event(_frame(f"s{i}"))

    stats = replay_stats()
    assert stats["sessions"] == event_replay._REPLAY_SESSIONS_MAX
    assert events_since("s0", 0) == []  # oldest session fully evicted
    assert latest_seq(f"s{event_replay._REPLAY_SESSIONS_MAX + 9}") == 1


def test_fifo_eviction_keeps_seq_monotonic_within_epoch():
    """#100122: FIFO eviction must not reset a revisited session's seq under
    the same process-wide replay epoch — clients hold their old watermark, and
    both the replay response and live parked frames with a reset (lower) seq
    are silently dropped by the client's dispatchIfNewer gate."""
    first = _frame("s0")
    second = _frame("s0")
    event_replay._stamp_event(first)
    event_replay._stamp_event(second)
    assert second["params"]["seq"] == 2

    # Evict s0's ring by pushing _REPLAY_SESSIONS_MAX newer sessions through.
    for index in range(1, event_replay._REPLAY_SESSIONS_MAX + 1):
        event_replay._stamp_event(_frame(f"s{index}"))

    assert "s0" not in event_replay._replay_buffers

    revisited = _frame("s0")
    event_replay._stamp_event(revisited)
    # Same epoch (no restart happened): the revisited session CONTINUES its
    # sequence instead of restarting at 1.
    assert revisited["params"]["seq"] == 3

    # A client holding the pre-eviction watermark still sees the new event…
    assert [event["seq"] for event in events_since("s0", 2)] == [3]
    # …while a client that saw only seq 1 is told the ring dropped what it
    # missed (seq 2 went out live but is no longer replayable): truncated,
    # refetch history.
    assert event_replay.is_truncated("s0", 1)
    # A client that saw seq 2 lost nothing: the ring resumes at 3 with no hole.
    assert not event_replay.is_truncated("s0", 2)
    assert not event_replay.is_truncated("s0", 3)


def test_fifo_eviction_marks_truncation_for_old_watermarks():
    """The whole retained ring is gone once a session is FIFO-evicted, so any
    client watermark below its latest stamped seq reports truncation —
    without the raise, a revisited session answers truncated=false and the
    client trusts a tail with a hole in it (#100122)."""
    frames = [_frame("s0") for _ in range(3)]
    for f in frames:
        event_replay._stamp_event(f)
    assert latest_seq("s0") == 3

    for index in range(1, event_replay._REPLAY_SESSIONS_MAX + 1):
        event_replay._stamp_event(_frame(f"s{index}"))

    assert event_replay.is_truncated("s0", 0)
    assert event_replay.is_truncated("s0", 2)
    assert not event_replay.is_truncated("s0", 3)  # saw everything before eviction


def test_concurrent_stamping_never_drops_or_duplicates_seq():
    errors = []

    def worker(sid):
        try:
            seen = set()
            for _ in range(200):
                f = _frame(sid)
                event_replay._stamp_event(f)
                seq = f["params"]["seq"]
                assert seq not in seen
                seen.add(seq)
        except AssertionError as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert replay_stats()["events"] == 8 * 200


def test_byte_budget_evicts_payloads_and_preserves_gap_semantics(monkeypatch):
    monkeypatch.setattr(event_replay, "_REPLAY_BUFFER_BYTES_MAX", 700)
    monkeypatch.setattr(event_replay, "_REPLAY_PROCESS_BYTES_MAX", 5_000)

    first = _frame("s1", "tool.complete")
    first["params"]["payload"] = {"result": "x" * 400}
    second = _frame("s1", "tool.complete")
    second["params"]["payload"] = {"result": "y" * 400}
    event_replay._stamp_event(first)
    event_replay._stamp_event(second)

    stats = replay_stats()
    assert stats["bytes"] <= stats["max_bytes_per_session"]
    assert [event["seq"] for event in events_since("s1", 0)] == [2]
    assert event_replay.is_truncated("s1", 0)

    reset_replay_state()
    monkeypatch.setattr(event_replay, "_REPLAY_BUFFER_BYTES_MAX", 1_000)
    monkeypatch.setattr(event_replay, "_REPLAY_PROCESS_BYTES_MAX", 700)
    first = _frame("s1", "tool.complete")
    first["params"]["payload"] = {"result": "x" * 400}
    event_replay._stamp_event(first)
    other = _frame("s2", "tool.complete")
    other["params"]["payload"] = {"result": "y" * 400}
    event_replay._stamp_event(other)

    stats = replay_stats()
    assert stats["bytes"] <= stats["max_bytes_process"]
    assert events_since("s1", 0) == []
    assert event_replay.is_truncated("s1", 0)
    assert [event["seq"] for event in events_since("s2", 0)] == [1]


def test_oversized_event_marks_gap_even_with_empty_buffer(monkeypatch):
    monkeypatch.setattr(event_replay, "_REPLAY_BUFFER_BYTES_MAX", 300)
    monkeypatch.setattr(event_replay, "_REPLAY_PROCESS_BYTES_MAX", 300)

    oversized = _frame("s1", "tool.complete")
    oversized["params"]["payload"] = {"result": "x" * 1000}
    event_replay._stamp_event(oversized)

    assert events_since("s1", 0) == []
    assert event_replay.is_truncated("s1", 0)
    assert not event_replay.is_truncated("s1", 1)

    event_replay._stamp_event(_frame("s1"))
    assert [event["seq"] for event in events_since("s1", 0)] == [2]
    assert event_replay.is_truncated("s1", 0)
    assert not event_replay.is_truncated("s1", 1)
    assert not event_replay.is_truncated("s1", 2)

    # The gap watermark never moves backwards: a small frame after an oversized one must
    # not hide the hole the oversized frame left.
    reset_replay_state()
    monkeypatch.setattr(event_replay, "_REPLAY_BUFFER_MAX", 1)
    monkeypatch.setattr(event_replay, "_REPLAY_BUFFER_BYTES_MAX", 1000)
    monkeypatch.setattr(event_replay, "_REPLAY_PROCESS_BYTES_MAX", 1000)
    event_replay._stamp_event(_frame("s"))
    large = _frame("s")
    large["params"]["payload"] = {"data": "x" * 2000}
    event_replay._stamp_event(large)
    event_replay._stamp_event(_frame("s"))
    assert event_replay.is_truncated("s", 1)


def test_truncation_detection_semantics():
    """The RPC handler's truncated flag: gap between last_seen and buffer start."""
    # Overflow the ring so the oldest events are genuinely evicted.
    for _ in range(event_replay._REPLAY_BUFFER_MAX + 10):
        event_replay._stamp_event(_frame("s1"))

    with event_replay._replay_lock:
        oldest = event_replay._replay_buffers["s1"][0][0]

    assert oldest > 1  # eviction happened

    # Client saw everything up to just before the buffer → NOT truncated.
    assert not event_replay.is_truncated("s1", oldest - 1)
    # Client saw seq 5, buffer starts later → truncated.
    assert event_replay.is_truncated("s1", 5)
    # Unknown session: nothing evicted, nothing truncated.
    assert not event_replay.is_truncated("nope", 0)


def test_evicted_live_session_keeps_its_seq_and_reports_the_gap():
    """A live session whose ring was evicted continues its numbering, and a reconnecting
    client that missed frames is told to refetch instead of getting an empty replay."""
    for _ in range(3):
        event_replay._stamp_event(_frame("main"))
    client_watermark = 2  # the socket dropped before seq 3 arrived
    for i in range(event_replay._REPLAY_SESSIONS_MAX):
        event_replay._stamp_event(_frame(f"other-{i}"))
    event_replay._stamp_event(_frame("main", "message.complete"))

    assert latest_seq("main") == 4
    assert [event["seq"] for event in events_since("main", client_watermark)] == [4]
    assert event_replay.is_truncated("main", client_watermark)
    assert not event_replay.is_truncated("main", 3)


def test_session_eviction_spares_the_recently_active_session():
    event_replay._stamp_event(_frame("main"))
    for i in range(event_replay._REPLAY_SESSIONS_MAX - 1):
        event_replay._stamp_event(_frame(f"idle-{i}"))
    event_replay._stamp_event(_frame("main"))  # the oldest session is the active one
    event_replay._stamp_event(_frame("new"))  # one over the cap

    assert [event["seq"] for event in events_since("main", 0)] == [1, 2]
    assert events_since("idle-0", 0) == []
