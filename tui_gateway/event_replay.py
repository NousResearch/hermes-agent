"""Per-session event sequencing + bounded replay for WS reconnects.

Every gateway event frame that flows through :func:`server.write_json` (and
therefore ``_emit``) is stamped with a per-session monotonic ``seq`` and
appended to a small ring buffer keyed by session id. A reconnecting client
calls the ``session.events.since`` RPC with its last observed seq; the server
replays everything newer from the buffer, then live events resume seamlessly.

Design constraints honored:
- stdio TUI path unaffected: frames gain a ``seq`` field only on event frames;
  Ink ignores unknown params keys.
- Thread safety: a single module lock guards counters + buffers, so buffer
  order always matches seq order. Wire order is enforced separately by the
  per-transport write path; two racing writers can briefly invert seq order
  on the wire, which the client tolerates (watermarks are monotonic-max).
- Memory bound: _REPLAY_BUFFER_MAX events / _REPLAY_SESSIONS_MAX sessions,
  least-recently-active session evicted first.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict, deque

# Replay ring per session. Sized for control events only — streaming token
# deltas are transient (stamped but not buffered, see _TRANSIENT_EVENT_TYPES),
# so 512 slots cover hours of durable events instead of ~one streaming burst.
_REPLAY_BUFFER_MAX = 512
# Distinct sessions remembered. Desktop users rarely exceed a dozen live chats.
_REPLAY_SESSIONS_MAX = 64

# Transient event types: delivered live but never buffered for replay.
# One streaming turn emits hundreds of per-token delta frames; buffering them
# would evict every durable control event from the ring (OpenHands makes the
# same split with StreamingDeltaEvent — published, never persisted). A
# reconnecting client recovers partial streamed text from the inflight
# snapshot in ``session.resume``, not from delta replay. These frames still
# get seqs (so ordering holds live), but replay skips them and gap detection
# ignores them via the durable-seq watermark below.
_TRANSIENT_EVENT_TYPES = frozenset({
    "message.delta",
    "thinking.delta",
})

# Server boot epoch: lets a client detect that the seq namespace was reset
# (gateway restart) — a stale high watermark from the previous process must
# not suppress live events forever (Goose clamps; we reset via epoch).
EPOCH = uuid.uuid4().hex[:8]

_replay_lock = threading.Lock()
# sid -> deque of (seq, params_dict). params is the same dict written to the
# wire and already carries its stamped "seq" key — the replay RPC returns
# these bare event objects, matching what the client's live dispatch sees.
# Manual eviction (no maxlen): we must record the seq of every DURABLE frame
# we drop, so gap detection can distinguish "durable data lost" from "the
# missing seqs were transient deltas that were never replayable anyway".
_replay_buffers: "OrderedDict[str, deque]" = OrderedDict()
_replay_next_seq: dict[str, int] = {}
# sid -> highest DURABLE seq evicted from the ring (0 = nothing evicted).
# Precise truncation signal: durable data is lost for a client at last_seen
# iff an evicted durable frame had seq > last_seen.
_replay_evicted_seq: dict[str, int] = {}


def stamp_event(obj: dict) -> None:
    """Stamp one outgoing event frame (mutates obj in place) and record it."""
    if obj.get("method") != "event":
        return
    params = obj.get("params")
    if not isinstance(params, dict):
        return
    sid = params.get("session_id") or ""
    if not sid:
        # Session-less global events (skin.changed etc.) are re-fetchable via
        # their own RPCs; no replay contract for them.
        return
    transient = params.get("type") in _TRANSIENT_EVENT_TYPES
    with _replay_lock:
        seq = _replay_next_seq.get(sid, 0) + 1
        _replay_next_seq[sid] = seq
        params["seq"] = seq
        if transient:
            # Live-only: seq stamped for wire ordering, never buffered.
            return
        buf = _replay_buffers.get(sid)
        if buf is None:
            buf = deque()
            _replay_buffers[sid] = buf
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                _oldest_sid, _oldest_buf = _replay_buffers.popitem(last=False)
                _replay_next_seq.pop(_oldest_sid, None)
                _replay_evicted_seq.pop(_oldest_sid, None)
        else:
            # LRU, not insertion-FIFO: an actively streaming session must not
            # be evicted just because it was created before idle newer ones.
            _replay_buffers.move_to_end(sid)
        buf.append((seq, params))
        while len(buf) > _REPLAY_BUFFER_MAX:
            dropped_seq, _dropped = buf.popleft()
            _replay_evicted_seq[sid] = dropped_seq


def events_since(sid: str, last_seen: int) -> tuple[list[dict], int, bool]:
    """Replay contract for one session, computed atomically.

    Returns ``(events, latest_seq, truncated)``:

    - ``events``: bare event params dicts (``type``/``session_id``/``seq``/
      ``payload``) with ``seq > last_seen``, in seq order. Transient delta
      frames are never included (they were never buffered).
    - ``latest_seq``: current highest stamped seq (0 when unknown).
    - ``truncated``: durable data the client has not seen is unrecoverable —
      either a DURABLE frame with ``seq > last_seen`` was evicted from the
      ring, or ``last_seen`` is AHEAD of ``latest_seq`` (seq namespace reset
      after a gateway restart / session eviction — the client should compare
      ``EPOCH`` and do a full state reload). Gaps consisting only of
      transient delta seqs are NOT truncation: those frames were never
      replayable and the resume snapshot covers their content.
    """
    sid = sid or ""
    with _replay_lock:
        latest = _replay_next_seq.get(sid, 0)
        evicted = _replay_evicted_seq.get(sid, 0)
        buf = _replay_buffers.get(sid)
        if not buf:
            return [], latest, last_seen > latest or evicted > last_seen
        frames = [params for seq, params in buf if seq > last_seen]
        truncated = last_seen > latest or evicted > last_seen
        return frames, latest, truncated


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def reset_replay_state() -> None:
    """Test hook."""
    with _replay_lock:
        _replay_buffers.clear()
        _replay_next_seq.clear()
        _replay_evicted_seq.clear()


def replay_epoch() -> str:
    """This process's seq-namespace epoch (see :data:`EPOCH`)."""
    return EPOCH


def replay_stats() -> dict:
    """Telemetry: buffer occupancy + per-turn timing for the ops/debug surface."""
    with _replay_lock:
        stats = {
            "sessions": len(_replay_buffers),
            "events": sum(len(b) for b in _replay_buffers.values()),
            "max_per_session": _REPLAY_BUFFER_MAX,
            "max_sessions": _REPLAY_SESSIONS_MAX,
        }
    # Per-turn timing from the live session table (not under the replay lock —
    # the session table has its own lock).
    try:
        import time as _time

        from tui_gateway.server import _sessions, _sessions_lock

        with _sessions_lock:
            active_turns = []
            for sid, session in _sessions.items():
                inflight = session.get("inflight_turn")
                if isinstance(inflight, dict) and inflight.get("started_at"):
                    active_turns.append({
                        "session_id": sid,
                        "trace_id": inflight.get("trace_id"),
                        "elapsed_s": round(_time.time() - float(inflight["started_at"]), 2),
                        "streaming": inflight.get("streaming", False),
                    })
        stats["active_turns"] = active_turns
    except Exception:
        stats["active_turns"] = []
    return stats
