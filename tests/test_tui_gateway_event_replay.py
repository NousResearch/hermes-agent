"""Tests for tui_gateway.event_replay — per-session event seq + replay ring."""

import threading

import pytest

from tui_gateway import event_replay
from tui_gateway.event_replay import (
    events_since,
    latest_seq,
    replay_stats,
    reset_replay_state,
    stamp_event,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_replay_state()
    yield
    reset_replay_state()


def _frame(sid, etype="status.update"):
    # status.update is a DURABLE event type (buffered for replay).
    # message.delta / thinking.delta are transient — stamped, never buffered.
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": etype, "session_id": sid, "payload": {}},
    }


def test_stamp_adds_monotonic_seq_per_session():
    f1 = _frame("s1")
    f2 = _frame("s1")
    other = _frame("s2")

    stamp_event(f1)
    stamp_event(other)
    stamp_event(f2)

    assert f1["params"]["seq"] == 1
    assert f2["params"]["seq"] == 2  # per-session counter, unaffected by s2
    assert other["params"]["seq"] == 1


def test_stamp_ignores_non_event_and_sessionless_frames():
    rpc = {"jsonrpc": "2.0", "id": 1, "result": {}}
    no_sid = {"jsonrpc": "2.0", "method": "event", "params": {"type": "skin.changed"}}

    stamp_event(rpc)
    stamp_event(no_sid)

    assert "seq" not in rpc
    assert "seq" not in no_sid["params"]
    assert replay_stats()["events"] == 0


def test_events_since_returns_bare_params_only_newer_in_order():
    frames = [_frame("s1") for _ in range(5)]
    for f in frames:
        stamp_event(f)

    got, latest, truncated = events_since("s1", 3)
    assert [e["seq"] for e in got] == [4, 5]
    # Replay returns the bare event params (what live dispatch sees), NOT the
    # full JSON-RPC frame envelope — the client reads event.type at top level.
    assert all("jsonrpc" not in e and e["type"] == "status.update" for e in got)
    assert latest == 5
    assert truncated is False

    all_events, _, _ = events_since("s1", 0)
    assert all_events == [f["params"] for f in frames]
    assert events_since("s1", 5) == ([], 5, False)
    assert latest_seq("s1") == 5


def test_unknown_session_returns_empty():
    assert events_since("nope", 0) == ([], 0, False)
    assert latest_seq("nope") == 0


def test_transient_deltas_stamped_but_not_buffered():
    """Streaming token deltas get seqs (live wire ordering) but never enter
    the replay ring — one streaming turn must not evict durable control
    events (message.start/complete, session.info) from replay coverage."""
    start = _frame("s1", "message.start")
    deltas = [_frame("s1", "message.delta") for _ in range(100)]
    thinking = [_frame("s1", "thinking.delta") for _ in range(50)]
    complete = _frame("s1", "message.complete")

    stamp_event(start)
    for f in deltas:
        stamp_event(f)
    for f in thinking:
        stamp_event(f)
    stamp_event(complete)

    # All frames got seqs, in one monotonic namespace.
    assert start["params"]["seq"] == 1
    assert complete["params"]["seq"] == 152
    assert deltas[0]["params"]["seq"] == 2

    # But only the 2 durable frames are buffered.
    assert replay_stats()["events"] == 2

    # Replay from 0 returns just the durable frames — and the delta-only
    # gaps between them are NOT truncation (nothing recoverable was lost).
    got, latest, truncated = events_since("s1", 0)
    assert [e["type"] for e in got] == ["message.start", "message.complete"]
    assert latest == 152
    assert truncated is False

    # A client that saw the whole live stream is fully covered too.
    assert events_since("s1", 152) == ([], 152, False)


def test_ring_buffer_is_bounded_and_reports_truncation():
    for _ in range(event_replay._REPLAY_BUFFER_MAX + 50):
        stamp_event(_frame("s1"))

    stats = replay_stats()
    assert stats["events"] == event_replay._REPLAY_BUFFER_MAX

    # Client that saw nothing (last_seen=0) has a gap older than the ring —
    # durable frames were evicted, so the gap is real truncation.
    got, latest, truncated = events_since("s1", 0)
    assert truncated is True
    assert latest == event_replay._REPLAY_BUFFER_MAX + 50
    assert len(got) == event_replay._REPLAY_BUFFER_MAX

    # Client aligned with the buffer start: fully covered, not truncated.
    oldest = got[0]["seq"]
    covered, _, covered_truncated = events_since("s1", oldest - 1)
    assert covered_truncated is False
    assert len(covered) == event_replay._REPLAY_BUFFER_MAX


def test_eviction_of_durable_frames_is_precise():
    """Truncation is keyed to the highest EVICTED durable seq, not the buffer
    floor: a client whose last_seen covers everything evicted is not
    truncated even when younger frames were dropped for other clients."""
    overflow = 50
    for _ in range(event_replay._REPLAY_BUFFER_MAX + overflow):
        stamp_event(_frame("s1"))

    # The first `overflow` durable frames (seq 1..50) were evicted.
    # A client at last_seen=overflow saw all of them — not truncated.
    _, _, at_boundary = events_since("s1", overflow)
    assert at_boundary is False

    # A client one behind the boundary lost seq=overflow — truncated.
    _, _, behind = events_since("s1", overflow - 1)
    assert behind is True


def test_epoch_reset_reports_truncated():
    """Client watermark AHEAD of the server (gateway restart) must be flagged.

    Without this, a restarted server returns [] / truncated=False and the
    client's stuck watermark silently kills replay forever.
    """
    for _ in range(3):
        stamp_event(_frame("s1"))

    got, latest, truncated = events_since("s1", 97)
    assert got == []
    assert latest == 3
    assert truncated is True

    # Session the server has never seen but the client has a watermark for.
    assert events_since("gone", 42) == ([], 0, True)


def test_epoch_constant_is_stable_and_shaped():
    """EPOCH identifies this process's seq namespace: 8 hex chars, constant
    for the process lifetime (clients compare it across reconnects)."""
    assert isinstance(event_replay.EPOCH, str)
    assert len(event_replay.EPOCH) == 8
    int(event_replay.EPOCH, 16)  # hex
    assert event_replay.EPOCH == event_replay.EPOCH  # stable reference


def test_session_count_bounded_with_lru_eviction():
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 10):
        stamp_event(_frame(f"s{i}"))

    stats = replay_stats()
    assert stats["sessions"] == event_replay._REPLAY_SESSIONS_MAX
    assert events_since("s0", 0)[0] == []  # oldest session fully evicted
    assert latest_seq(f"s{event_replay._REPLAY_SESSIONS_MAX + 9}") == 1


def test_active_session_survives_eviction_lru():
    """Eviction is least-recently-ACTIVE, not first-created: a session that
    keeps streaming must outlive idle sessions created after it."""
    stamp_event(_frame("active"))
    for i in range(event_replay._REPLAY_SESSIONS_MAX - 1):
        stamp_event(_frame(f"idle{i}"))

    # "active" is now the oldest by creation. Touch it, then overflow.
    stamp_event(_frame("active"))
    stamp_event(_frame("newcomer"))

    assert latest_seq("active") == 2  # survived — it was most recently active
    assert events_since("idle0", 0) == ([], 0, False)  # idle0 evicted instead


def test_concurrent_stamping_never_drops_or_duplicates_seq():
    errors = []

    def worker(sid):
        try:
            seen = set()
            for _ in range(200):
                f = _frame(sid)
                stamp_event(f)
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
