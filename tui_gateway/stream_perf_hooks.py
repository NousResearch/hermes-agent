"""Per-turn LLM streaming performance metrics (TTFT / generation window / output tokens).

All data comes from Hermes' official observer hooks — nothing here touches the
agent's hot path:

  - ``post_api_request`` (synchronous, fired from conversation_loop.py):
      ``session_id / api_call_count / started_at / ended_at / usage.output_tokens``
  - ``on_stream_delta`` (async observer worker, fired from run_agent.py):
      ``session_id / iteration(api_call_count)``; the callback timestamp
      approximates the token's arrival time.

Metrics semantics:

  - TTFT (time to first token) = first-delta callback time - ``started_at``
    (API call start), i.e. "from request sent to first token back". Wall-clock
    within the same process, so the only error is the observer queue's
    millisecond-level latency.
  - Generation window ``gen_ms`` = ``ended_at`` - first-delta time: pure LLM
    output time, excluding tool execution / API queueing, so TPS is not diluted.
  - TPS accounting only counts calls that recorded a first chunk (text turns);
    tool turns (no text delta) do not contribute ``output_tokens``, so tool
    argument generation is never mixed into the text rate.

Registered the same way as first-party precedents (agent/outbound_webhooks.py,
agent/shell_hooks.py): append callbacks straight to
``get_plugin_manager()._hooks``. Registration happens once at import time.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_TURN_KEYS = ("calls", "ttft_calls", "ttft_ms", "gen_ms", "output_tokens")


class StreamPerfCollector:
    """Aggregates streaming perf metrics for one in-flight turn, per session.

    - ``begin_turn(sid)``                 : start a new turn at message.start (reset)
    - ``on_first_delta(sid, call, at)``   : first-delta time (recorded once, idempotent)
    - ``on_api_done(sid, call, started_at, ended_at, output_tokens)``: fold an API call
    - ``end_turn(sid)``                   : take the summary at message.complete (None if empty)

    Thread-safe: on_stream_delta fires on the observer worker thread while
    post_api_request fires synchronously on the agent thread — both may run
    concurrently, so everything funnels through one lock.
    """

    def __init__(self, on_update: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> None:
        self._lock = threading.Lock()
        self._turns: Dict[str, Dict[str, Any]] = {}
        # (session_id, api_call_count) -> (first-delta time, HTTP request sent time, first chunk time)
        # The first-token time is min(first chunk, first text delta): a reasoning
        # model's first text delta lands well after its first chunk, so TTFT is
        # measured from the earlier first chunk.
        self._pending: Dict[Tuple[str, int], Tuple[float, Optional[float], Optional[float]]] = {}
        # Called after each API call is folded in (live push, incremental
        # semantics). Signature: (agent_sid, per-call increment).
        self._on_update = on_update

    def set_on_update(self, on_update: Optional[Callable[[str, Dict[str, Any]], None]]) -> None:
        with self._lock:
            self._on_update = on_update

    def begin_turn(self, sid: str) -> None:
        if not sid:
            return
        with self._lock:
            self._turns[sid] = {k: (0 if k != "ttft_ms" else 0.0) for k in _TURN_KEYS}

    def on_first_delta(
        self,
        sid: str,
        call: int,
        at: float,
        request_sent_at: Optional[float] = None,
        first_chunk_at: Optional[float] = None,
    ) -> None:
        """Record the first-delta (first-token) time for a call. Only the first
        arrival is kept; later deltas / retries never overwrite it."""
        if not sid:
            return
        key = (sid, call)
        with self._lock:
            if key not in self._pending:
                self._pending[key] = (at, request_sent_at, first_chunk_at)

    def on_api_done(
        self,
        sid: str,
        call: int,
        started_at: float,
        ended_at: float,
        output_tokens: int,
    ) -> None:
        if not sid:
            return
        key = (sid, call)
        with self._lock:
            pending = self._pending.pop(key, None)
            turn = self._turns.get(sid)
            if turn is None:
                return  # No in-flight turn (late event) -> drop, never cross-pollute turns
            turn["calls"] += 1
            if pending is None:
                # Tool turn (no text delta): contributes no TTFT / generation
                # window / output_tokens, so tool argument generation and
                # queueing time never dilute TPS.
                return
            first, request_sent_at, first_chunk_at = pending
            # First-token time = min(first chunk, first text delta): a reasoning
            # model's first chunk arrives far earlier than its first text delta,
            # so TTFT is anchored to the earlier first chunk.
            first_token_at = min(first, first_chunk_at) if first_chunk_at is not None else first
            # Prefer the HTTP-request-sent time as the TTFT baseline (excludes
            # agent-side request preparation); fall back to post_api_request's
            # started_at when the backend doesn't supply the field.
            sent = request_sent_at if request_sent_at is not None else float(started_at)
            turn["ttft_calls"] += 1
            ttft_ms = max(0.0, first_token_at - float(sent)) * 1000
            turn["ttft_ms"] += ttft_ms
            api_end = float(ended_at)
            api_dur = max(0.0, api_end - float(sent)) if request_sent_at is not None else max(0.0, api_end - float(started_at))
            gen = max(0.0, api_end - first_token_at)
            if gen < 0.3 * api_dur + 0.05:
                # Batch-return signature (provider buffers, first chunk ≈ last
                # chunk): the client cannot observe a per-token generation
                # window, generation spans the whole call — degrade to the
                # total call duration so gen≈0 never inflates TPS.
                gen = api_dur
            turn["gen_ms"] += gen * 1000
            turn["output_tokens"] += max(0, int(output_tokens or 0))
            # Live push (incremental): this call's TTFT / generation window /
            # output tokens go out immediately, so the frontend can show them
            # before the turn ends instead of waiting for message.complete.
            if self._on_update is not None:
                try:
                    self._on_update(
                        sid,
                        {
                            "calls": 1,
                            "ttft_calls": 1,
                            "ttft_ms": round(ttft_ms, 1),
                            "gen_ms": round(gen * 1000, 1),
                            "output_tokens": max(0, int(output_tokens or 0)),
                        },
                    )
                except Exception:
                    logger.debug("stream_perf on_update callback failed", exc_info=True)

    def end_turn(self, sid: str) -> Optional[Dict[str, Any]]:
        if not sid:
            return None
        with self._lock:
            turn = self._turns.pop(sid, None)
            if turn is None or not turn["calls"]:
                # Empty turn (no API calls) or unknown session -> no stats
                if turn is None:
                    for key in [k for k in self._pending if k[0] == sid]:
                        self._pending.pop(key, None)
                return None
            # Defense: drop any leftover pending entries for this session (the
            # normal path pairs them before end_turn).
            for key in [k for k in self._pending if k[0] == sid]:
                self._pending.pop(key, None)
            return {
                "calls": turn["calls"],
                "ttft_calls": turn["ttft_calls"],
                "ttft_ms": round(turn["ttft_ms"], 1),
                "gen_ms": round(turn["gen_ms"], 1),
                "output_tokens": turn["output_tokens"],
            }


def _make_post_api_request_cb(collector: StreamPerfCollector):
    """post_api_request: API call done — fold TTFT / generation window / output tokens."""

    def _cb(**kwargs: Any) -> None:
        try:
            sid = str(kwargs.get("session_id") or "")
            call = int(kwargs.get("api_call_count") or 0)
            started_at = kwargs.get("started_at")
            ended_at = kwargs.get("ended_at")
            if not sid or started_at is None or ended_at is None:
                return
            usage = kwargs.get("usage") or {}
            out = (
                int(usage.get("output_tokens") or 0)
                if isinstance(usage, dict)
                else 0
            )
            collector.on_api_done(sid, call, float(started_at), float(ended_at), out)
        except Exception:
            logger.debug("stream_perf post_api_request hook failed", exc_info=True)

    return _cb


def _make_on_stream_delta_cb(collector: StreamPerfCollector):
    """on_stream_delta: the first delta time IS the first-token time.

    Prefers ``delta_at`` from the payload (recorded on the agent's synchronous
    token path — accurate); falls back to the callback time (async worker,
    compatibility only).
    """

    def _cb(**kwargs: Any) -> None:
        try:
            sid = str(kwargs.get("session_id") or "")
            call = int(kwargs.get("iteration") or 0)
            if not sid:
                return
            delta_at = kwargs.get("delta_at")
            at = float(delta_at) if delta_at is not None else time.time()
            sent = kwargs.get("request_sent_at")
            request_sent_at = float(sent) if sent is not None else None
            fca = kwargs.get("first_chunk_at")
            first_chunk_at = float(fca) if fca is not None else None
            collector.on_first_delta(sid, call, at, request_sent_at, first_chunk_at)
        except Exception:
            logger.debug("stream_perf on_stream_delta hook failed", exc_info=True)

    return _cb


_registered = False


def register_stream_perf_hooks() -> StreamPerfCollector:
    """Register the official observer hooks and return the global collector
    (idempotent — call once at import time)."""
    global _registered
    if _registered:
        return _COLLECTOR
    try:
        from hermes_cli.plugins import get_plugin_manager

        manager = get_plugin_manager()
        manager._hooks.setdefault("post_api_request", []).append(
            _make_post_api_request_cb(_COLLECTOR)
        )
        manager._hooks.setdefault("on_stream_delta", []).append(
            _make_on_stream_delta_cb(_COLLECTOR)
        )
        _registered = True
        logger.debug("stream_perf hooks registered")
    except Exception:
        logger.debug("stream_perf hooks registration failed", exc_info=True)
    return _COLLECTOR


_COLLECTOR = StreamPerfCollector()
