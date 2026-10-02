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


def test_tombstone_metadata_is_bounded_far_beyond_the_ring():
    """#127255 review: distinct session ids are not bounded by the replay ring, so
    the retained seq/truncation counters must have their own cap — otherwise a
    long-lived gateway pins two ints per historical session for the process
    lifetime, invisible in replay_stats()."""
    stamp_max = event_replay._REPLAY_TOMBSTONE_MAX
    assert stamp_max > event_replay._REPLAY_SESSIONS_MAX  # a real bound, not the ring

    for i in range(stamp_max * 3 + 25):  # far beyond the cap, all unique ids
        event_replay._stamp_event(_frame(f"s{i}"))

    stats = replay_stats()
    assert stats["tombstones"] == stamp_max
    assert stats["tombstones"] <= stats["max_tombstones"]
    assert len(event_replay._replay_next_seq) == stamp_max
    assert len(event_replay._replay_evicted_through) == stamp_max
    assert len(event_replay._replay_seq_origin) == stamp_max
    # The retained tombstones are exactly the most recently stamped sessions…
    assert list(event_replay._replay_next_seq) == [f"s{i}" for i in range(stamp_max * 3 + 25 - stamp_max, stamp_max * 3 + 25)]
    # …each still answering its own numbering.
    last_sid = f"s{stamp_max * 3 + 24}"
    assert latest_seq(last_sid) == event_replay._replay_next_seq[last_sid]


def test_retired_tombstone_restart_is_explicit_not_silent():
    """Client semantics when a tombstone is retired (#127255 review): the session
    restarts ABOVE every seq it ever used, so no live or replayed frame can carry
    a seq the client's dispatchIfNewer gate already saw — and a client holding a
    watermark from the old numbering is told truncated (refetch) instead of
    trusting a silently renumbered tail."""
    first = _frame("s-old")
    event_replay._stamp_event(first)
    old_seq = first["params"]["seq"]
    assert old_seq == 1

    # Push s-old's tombstone past the LRU cap: it is retired, its numbering
    # unobservable.
    for i in range(event_replay._REPLAY_TOMBSTONE_MAX + 5):
        event_replay._stamp_event(_frame(f"s{i}"))
    assert "s-old" not in event_replay._replay_next_seq

    # Revisit: restarts above the floor — above every seq the session ever used,
    # so no client's dispatchIfNewer gate can have seen this seq before and
    # silently drop the frame.
    revisited = _frame("s-old")
    event_replay._stamp_event(revisited)
    assert revisited["params"]["seq"] == event_replay._replay_seq_floor + 1
    assert revisited["params"]["seq"] > old_seq

    # Explicit reset signal, reusing the truncation path: a client that saw part
    # of the old numbering refetches; a client that saw nothing does not.
    assert event_replay.is_truncated("s-old", 1)
    assert not event_replay.is_truncated("s-old", 0)
    # The replay tail answers with the restarted numbering only.
    assert [event["seq"] for event in events_since("s-old", 0)] == [revisited["params"]["seq"]]


def test_retained_tombstone_keeps_exact_continuity():
    """A session whose tombstone is still retained (LRU cap not crossed) keeps the
    #127255 behavior exactly: revisit continues the seq, and old watermarks are
    served by the truncation watermark, not by a renumbering."""
    frames = [_frame("s-live") for _ in range(3)]
    for f in frames:
        event_replay._stamp_event(f)

    # Fill the ring with other sessions, evicting s-live's ring but NOT its
    # tombstone (cap is a multiple of the ring size).
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 1):
        event_replay._stamp_event(_frame(f"s{i}"))
    assert "s-live" not in event_replay._replay_buffers
    assert latest_seq("s-live") == 3

    revisited = _frame("s-live")
    event_replay._stamp_event(revisited)
    assert revisited["params"]["seq"] == 4  # exact continuity, no floor jump
    assert [event["seq"] for event in events_since("s-live", 3)] == [4]
    assert not event_replay.is_truncated("s-live", 3)


def test_seq_floor_never_recycles_a_used_number():
    """The per-epoch seq floor rises past every retired session's last seq, so a
    retirement-then-revisit can never land under any client watermark — the core
    invariant that lets the reset be safe at all."""
    high = _frame("s-high")
    event_replay._stamp_event(high)
    for _ in range(9):
        event_replay._stamp_event(_frame("s-high"))
    top = high["params"]["seq"] + 9
    assert latest_seq("s-high") == 10 and top == 10

    for i in range(event_replay._REPLAY_TOMBSTONE_MAX + 5):
        event_replay._stamp_event(_frame(f"s{i}"))  # retires s-high's tombstone

    # New sessions after retirement start above the floor…
    fresh = _frame("s-fresh")
    event_replay._stamp_event(fresh)
    assert fresh["params"]["seq"] == event_replay._replay_seq_floor + 1
    # …which is above every seq the retired session ever stamped.
    assert event_replay._replay_seq_floor >= top
    assert fresh["params"]["seq"] > top
