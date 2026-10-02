"""Per-session event sequencing + bounded replay for WS reconnects.

Every event frame through :func:`server.write_json` (hence ``_emit``) gets a per-session monotonic
``seq`` and lands in a small ring per session; a reconnecting client calls ``session.events.since``
with its last seen seq and gets everything newer. Invariants: stdio TUI unaffected (``seq`` only on
event frames; Ink ignores unknown keys); one lock guards counters + buffers, and write_json already
serializes per-transport writes so stamping cannot reorder frames; memory bound =
_REPLAY_BUFFER_MAX events AND _REPLAY_BUFFER_BYTES_MAX serialized bytes per session,
_REPLAY_PROCESS_BYTES_MAX bytes across at most _REPLAY_SESSIONS_MAX sessions, oldest evicted
FIFO. A session's seq/truncation counters outlive its ring (FIFO eviction must not
reset its numbering, #100122) but only up to _REPLAY_TOMBSTONE_MAX retained tombstones: a
revisit while retained continues the seq exactly, and once the tombstone retires the session
restarts ABOVE every seq it ever used (the per-epoch seq floor) with ``truncated`` reported,
so a client sees either continuity or an explicit refetch signal — never a silently-lower
seq. Evicted or never-retained (oversized) frames leave a truncation watermark so a
reconnecting client refetches instead of trusting a replay with holes.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict, deque

# Seq counters live in-process, so a restart resets them to 1 while clients hold high
# watermarks — events_since(sid, 97) would return [] with truncated=False forever. The
# epoch lets clients detect the restart and reset their watermarks.
_REPLAY_EPOCH = uuid.uuid4().hex

# A long turn emits ~hundreds of token events; 512 covers minutes of streaming plus
# all control events. Desktop users rarely exceed a dozen live chats.
_REPLAY_BUFFER_MAX = 512
_REPLAY_SESSIONS_MAX = 64
# Session seq/truncation tombstones: one (seq, watermark) pair kept per session whose ring
# was FIFO-evicted, so a revisit continues its numbering instead of restarting at 1 under
# still-held client watermarks (#100122, #127255 review). Distinct sessions are NOT bounded
# by the ring — a long-lived gateway serving an unbounded stream of unique ids would pin two
# ints per id for the process lifetime — so the tombstones themselves get an LRU cap.
# 4× the ring: enough to cover every ring slot plus its ring eviction history, while
# retirement remains a rarity measured in distinct session ids, not traffic. When the cap is
# crossed, the least-recently-stamped tombstone is retired and the per-epoch seq floor
# rises to its last seq; a later revisit restarts ABOVE that floor (retirement
# loop in _stamp_event) so the client either sees continuity (retained) or an
# explicit truncated refetch signal.
_REPLAY_TOMBSTONE_MAX = 4 * _REPLAY_SESSIONS_MAX
# A ring may legitimately hold many bounded 64 KiB tool results (512 of them ≈ 32 MiB per
# session, ×64 sessions before any cap); bound the serialized bytes so replay memory cannot
# scale with payload size without limit.
_REPLAY_BUFFER_BYTES_MAX = 4 * 1024 * 1024
_REPLAY_PROCESS_BYTES_MAX = 64 * 1024 * 1024

_replay_lock = threading.Lock()
# sid -> deque of (seq, params dict, serialized bytes).
_replay_buffers: "OrderedDict[str, deque]" = OrderedDict()
_replay_buffer_bytes: dict[str, int] = {}
_replay_evicted_through: "OrderedDict[str, int]" = OrderedDict()
_replay_total_bytes = 0
_replay_next_seq: "OrderedDict[str, int]" = OrderedDict()
# First seq of a sid's CURRENT numbering (cold start after retirement or process
# start). A client watermark below it necessarily predates this numbering — the
# reset signal for a retired tombstone (#127255 review). Part of the tombstone:
# LRU-capped and retired with the rest.
_replay_seq_origin: "OrderedDict[str, int]" = OrderedDict()
# Per-epoch ceiling over every seq retired with a tombstone: a session whose tombstone
# was retired restarts above this floor, so no frame can ever carry a seq a client already
# saw (dispatchIfNewer would silently drop it — #100122). Only grows within the epoch.
_replay_seq_floor = 0


def replay_epoch() -> str:
    """Opaque token identifying this server process's seq numbering."""
    return _REPLAY_EPOCH


def _stamp_event(obj: dict) -> None:
    """Stamp one outgoing event frame (mutates obj in place) and record it."""
    if obj.get("method") != "event":
        return
    params = obj.get("params")
    if not isinstance(params, dict):
        return
    sid = params.get("session_id") or ""
    if not sid:
        # Session-less global events (skin.changed etc.) are re-fetchable via their own RPCs.
        return
    # Sizing stays OUTSIDE the lock (same rule as transport.write) so one large payload cannot
    # stall other threads' frames; ``seq`` is not stamped yet, a few bytes off a MiB budget.
    size = len(json.dumps(params, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8", errors="surrogatepass"))
    with _replay_lock:
        global _replay_total_bytes, _replay_seq_floor
        seq = _replay_next_seq.get(sid, 0) + 1
        if sid not in _replay_next_seq:
            # Cold start for a session with no retained tombstone (retired by the
            # LRU cap, or genuinely new). Retired sessions must never hand out a
            # seq the client's dispatchIfNewer gate already saw, so restart ABOVE
            # the floor; a genuinely-new sid also starts there, which is fine:
            # seq is opaque, only per-session monotonicity matters to clients.
            # is_truncated() reports the numbering change to a client holding a
            # watermark from the old numbering (it sits below the new origin).
            seq = max(seq, _replay_seq_floor + 1)
            _replay_seq_origin[sid] = seq
        # Most-recently stamped session stays newest in all three LRU orders.
        _replay_next_seq[sid] = seq
        _replay_evicted_through.setdefault(sid, 0)
        _replay_seq_origin.setdefault(sid, seq)
        _replay_next_seq.move_to_end(sid)
        _replay_evicted_through.move_to_end(sid)
        _replay_seq_origin.move_to_end(sid)
        params["seq"] = seq
        buf = _replay_buffers.get(sid)
        if buf is None:
            buf = _replay_buffers[sid] = deque()
            _replay_buffer_bytes[sid] = 0
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                oldest_sid, _oldest_buf = _replay_buffers.popitem(last=False)
                _replay_total_bytes -= _replay_buffer_bytes.pop(oldest_sid, 0)
                # FIFO eviction drops the ring, not the session's seq numbering
                # (#100122). Keep the counter so a revisited session continues
                # from its high seq instead of restarting at 1 under clients'
                # still-held watermarks (a reset seq is invisible to
                # dispatchIfNewer — replay AND live frames silently vanish).
                # Retain the truncation watermark too, raised to the latest
                # stamped seq: the whole retained ring is gone, so a client
                # holding any older watermark has a gap and must refetch
                # history instead of trusting the new tail. The seq/truncation
                # counters are one tombstone per session id, itself LRU-bounded
                # by _REPLAY_TOMBSTONE_MAX (distinct sessions are not bounded
                # by the ring).
                _replay_evicted_through[oldest_sid] = max(
                    _replay_evicted_through.get(oldest_sid, 0), _replay_next_seq.get(oldest_sid, 0))
        # Tombstone LRU cap (#127255 review): distinct session ids are not bounded by
        # the ring, so the retained counters themselves need a bound or a long-lived
        # gateway pins two ints per historical session for the process lifetime.
        while len(_replay_next_seq) > _REPLAY_TOMBSTONE_MAX:
            retired_sid, retired_seq = _replay_next_seq.popitem(last=False)
            _replay_evicted_through.pop(retired_sid, None)
            _replay_seq_origin.pop(retired_sid, None)
            # The retired session's numbering is now unobservable: a later revisit
            # must never reuse a seq a still-connected client may hold, so raise the
            # per-epoch floor past every seq it stamped and report the reset via
            # is_truncated to a client holding a watermark on it (refetch, like any
            # other gap). Retiring the truncation watermark too is safe for the
            # same reason: the restarted numbering's origin flags any watermark
            # from the old one.
            _replay_seq_floor = max(_replay_seq_floor, retired_seq)
        if size > _REPLAY_BUFFER_BYTES_MAX or size > _REPLAY_PROCESS_BYTES_MAX:
            _replay_evicted_through[sid] = seq
            return
        buf.append((seq, params, size))
        _replay_buffer_bytes[sid] += size
        _replay_total_bytes += size
        while len(buf) > _REPLAY_BUFFER_MAX or _replay_buffer_bytes[sid] > _REPLAY_BUFFER_BYTES_MAX:
            evicted_seq, _event, evicted_size = buf.popleft()
            _replay_buffer_bytes[sid] -= evicted_size
            _replay_total_bytes -= evicted_size
            _replay_evicted_through[sid] = max(_replay_evicted_through.get(sid, 0), evicted_seq)
        while _replay_total_bytes > _REPLAY_PROCESS_BYTES_MAX:
            for evict_sid, evict_buf in _replay_buffers.items():
                if evict_buf:
                    evicted_seq, _event, evicted_size = evict_buf.popleft()
                    _replay_buffer_bytes[evict_sid] -= evicted_size
                    _replay_total_bytes -= evicted_size
                    _replay_evicted_through[evict_sid] = max(_replay_evicted_through.get(evict_sid, 0), evicted_seq)
                    break


def events_since(sid: str, last_seen: int) -> list[dict]:
    """Recorded EVENT OBJECTS (each frame's ``params`` dict) with seq > last_seen for *sid*.

    Returning the full JSON-RPC envelope would make every replayed event fail the
    client's ``event.type`` gate and be silently dropped.
    """
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        return [event for seq, event, _size in buf if seq > last_seen] if buf else []


def is_truncated(sid: str, last_seen: int) -> bool:
    """True when events between *last_seen* and the ring's oldest retained seq were
    evicted — the client must refetch history instead of trusting the replay.

    Also true when *last_seen* predates the sid's CURRENT numbering origin: the
    session's tombstone was retired by the LRU cap (or the process restarted,
    which the epoch signals instead), so the numbering restarted above the seq
    floor and a client watermark from the old numbering must refetch rather than
    trust a tail that silently renumbers from a different origin (#127255
    review). A watermark of 0 (never saw anything) is never truncated, matching
    a genuinely-new session.
    """
    with _replay_lock:
        origin = _replay_seq_origin.get(sid or "", 1)
        if 0 < last_seen < origin:
            return True
        if last_seen > _replay_next_seq.get(sid or "", 0):
            return last_seen > 0
        return last_seen < _replay_evicted_through.get(sid or "", 0)


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def reset_replay_state() -> None:
    """Test hook."""
    with _replay_lock:
        global _replay_total_bytes, _replay_seq_floor
        _replay_buffers.clear()
        _replay_buffer_bytes.clear()
        _replay_evicted_through.clear()
        _replay_next_seq.clear()
        _replay_seq_origin.clear()
        _replay_total_bytes = 0
        _replay_seq_floor = 0


def replay_stats() -> dict:
    """Telemetry: buffer occupancy for the ops/debug surface."""
    with _replay_lock:
        return {
            "sessions": len(_replay_buffers),
            "events": sum(len(buffer) for buffer in _replay_buffers.values()),
            "bytes": _replay_total_bytes,
            "tombstones": len(_replay_next_seq),
            "max_per_session": _REPLAY_BUFFER_MAX,
            "max_bytes_per_session": _REPLAY_BUFFER_BYTES_MAX,
            "max_bytes_process": _REPLAY_PROCESS_BYTES_MAX,
            "max_tombstones": _REPLAY_TOMBSTONE_MAX}
