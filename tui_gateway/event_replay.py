"""Per-session event sequencing + bounded replay for WS reconnects.

Every gateway event frame that flows through :func:`server.write_json` (and
therefore ``_emit``) is stamped with a per-session monotonic ``seq`` and
appended to a small ring buffer keyed by session id. A reconnecting client
calls the ``session.events.since`` RPC with its last observed seq; the server
replays everything newer from the buffer, then live events resume seamlessly.

Design constraints honored:
- stdio TUI path unaffected: frames gain a ``seq`` field only on event frames;
  Ink ignores unknown params keys.
- Thread safety: a single module lock guards counters + buffers; write_json
  already serializes per-transport writes, so stamping under the lock cannot
  reorder frames relative to each other.
- Memory bound: _REPLAY_BUFFER_MAX events / _REPLAY_SESSIONS_MAX sessions,
  oldest session evicted FIFO.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict, deque
from typing import NamedTuple

# Process identity for the replay contract. Seq counters live in-process, so
# a gateway restart silently resets them to 1 while clients still hold high
# watermarks — events_since(sid, 97) then returns [] with truncated=False and
# the client believes it missed nothing (and its stale watermark makes every
# future replay empty too). The epoch lets clients detect the restart and
# reset their watermarks.
_REPLAY_EPOCH = uuid.uuid4().hex

# Replay ring per session. A long turn emits ~hundreds of token events; this
# covers several minutes of streaming plus all control events.
_REPLAY_BUFFER_MAX = 512
# Distinct sessions remembered. Desktop users rarely exceed a dozen live chats.
_REPLAY_SESSIONS_MAX = 64

_replay_lock = threading.Lock()
# sid -> deque of (seq, event_object) where event_object is the frame's
# ``params`` dict (bare event: type/session_id/seq/payload) — the exact shape
# the client's dispatch path consumes.
_replay_buffers: "OrderedDict[str, deque]" = OrderedDict()
_replay_next_seq: dict[str, int] = {}
_replay_generations: dict[str, str] = {}


class ReplaySnapshot(NamedTuple):
    """One lock-consistent view of a session's replay state."""

    events: list[dict]
    latest_seq: int
    generation: str | None
    truncated: bool


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
        # Session-less global events (skin.changed etc.) are re-fetchable via
        # their own RPCs; no replay contract for them.
        return
    with _replay_lock:
        seq = _replay_next_seq.get(sid, 0) + 1
        _replay_next_seq[sid] = seq
        params["seq"] = seq
        buf = _replay_buffers.get(sid)
        if buf is None:
            buf = deque(maxlen=_REPLAY_BUFFER_MAX)
            _replay_buffers[sid] = buf
            _replay_generations[sid] = uuid.uuid4().hex
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                _oldest_sid, _oldest_buf = _replay_buffers.popitem(last=False)
                _replay_next_seq.pop(_oldest_sid, None)
                _replay_generations.pop(_oldest_sid, None)
        params["replay_generation"] = _replay_generations[sid]
        buf.append((seq, params))


def events_since(sid: str, last_seen: int) -> list[dict]:
    """Return recorded EVENT OBJECTS with seq > last_seen for *sid*, in order.

    Shape contract: each element is the frame's ``params`` dict — a bare event
    object with top-level ``type`` / ``session_id`` / ``seq`` — because that is
    exactly what the client's dispatch path consumes. Returning the full
    JSON-RPC envelope here would make every replayed event fail the client's
    ``event.type`` gate and be silently dropped.
    """
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        if not buf:
            return []
        return [event for seq, event in buf if seq > last_seen]


def replay_snapshot(sid: str, last_seen: int) -> ReplaySnapshot:
    """Return one atomic, detached replay response snapshot for *sid*."""
    key = sid or ""
    with _replay_lock:
        buf = _replay_buffers.get(key)
        events = [event.copy() for seq, event in buf or () if seq > last_seen]
        return ReplaySnapshot(
            events=events,
            latest_seq=_replay_next_seq.get(key, 0),
            generation=_replay_generations.get(key),
            truncated=bool(buf and last_seen + 1 < buf[0][0]),
        )


def is_truncated(sid: str, last_seen: int) -> bool:
    """True when events between *last_seen* and the ring's oldest retained
    seq were evicted — the client must refetch history instead of trusting
    the replay to be gap-free."""
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        if not buf:
            return False
        return last_seen + 1 < buf[0][0]


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def replay_generation(sid: str) -> str | None:
    """Current generation for *sid* (changes whenever its ring is recreated)."""
    with _replay_lock:
        return _replay_generations.get(sid or "")


def reset_replay_state() -> None:
    """Test hook."""
    with _replay_lock:
        _replay_buffers.clear()
        _replay_next_seq.clear()
        _replay_generations.clear()


def replay_stats() -> dict:
    """Telemetry: buffer occupancy for the ops/debug surface."""
    with _replay_lock:
        return {
            "sessions": len(_replay_buffers),
            "events": sum(len(b) for b in _replay_buffers.values()),
            "max_per_session": _REPLAY_BUFFER_MAX,
        }
