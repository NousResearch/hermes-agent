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
from collections import OrderedDict, deque

# Replay ring per session. A long turn emits ~hundreds of token events; this
# covers several minutes of streaming plus all control events.
_REPLAY_BUFFER_MAX = 512
# Distinct sessions remembered. Desktop users rarely exceed a dozen live chats.
_REPLAY_SESSIONS_MAX = 64

_replay_lock = threading.Lock()
# sid -> OrderedDict-ish deque of (seq, frame_params_dict_without_seq)
_replay_buffers: "OrderedDict[str, deque]" = OrderedDict()
_replay_next_seq: dict[str, int] = {}


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
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                _oldest_sid, _oldest_buf = _replay_buffers.popitem(last=False)
                _replay_next_seq.pop(_oldest_sid, None)
        buf.append((seq, obj))


def events_since(sid: str, last_seen: int) -> list[dict]:
    """Return recorded event FRAMES with seq > last_seen for *sid*, in order."""
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        if not buf:
            return []
        return [frame for seq, frame in buf if seq > last_seen]


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def reset_replay_state() -> None:
    """Test hook."""
    with _replay_lock:
        _replay_buffers.clear()
        _replay_next_seq.clear()


def replay_stats() -> dict:
    """Telemetry: buffer occupancy for the ops/debug surface."""
    with _replay_lock:
        return {
            "sessions": len(_replay_buffers),
            "events": sum(len(b) for b in _replay_buffers.values()),
            "max_per_session": _REPLAY_BUFFER_MAX,
        }
