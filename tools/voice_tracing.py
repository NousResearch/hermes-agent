"""OTLP spans for the phases of one server-owned converse turn.

Emits a ``voice.turn`` parent span with child spans for the pipeline phases —
``voice.capture`` (VAD trip → endpoint commit, incl. Smart Turn hold/commit decisions),
``voice.stt``, ``voice.agent`` (LLM), ``voice.tts`` — so an operator can see, per turn, where
the latency goes and how the adaptive endpoint behaved.

**Content-free by construction.** Like the rest of `agent/monitoring`, NO message content
egresses: span attributes are only timings, counts, booleans, and the Smart Turn probability —
never the transcript, the reply, or the user's name.

Wiring: the tracer is built lazily from the operator's ``monitoring.export.otlp`` config (reusing
`agent.monitoring.otlp_exporter`'s SDK loader, exporter, and resource), so it rides the collector
the health export already uses. It is a no-op unless OTLP export is enabled AND
``monitoring.voice_tracing.enabled`` is true (default true when OTLP is on). Any failure degrades
to silence — tracing must never wedge or slow the voice loop.

Phases are recorded with explicit wall-clock start/end times into a :class:`PhaseRecorder` as the
turn runs (across the VAD worker thread and the driver coroutine), then materialized into spans in
ONE place at turn end (:func:`emit_turn_trace`) — so no live span is ever handed between threads.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger("hermes_cli.web_server")

_INSTRUMENTATION = "hermes.voice.converse"

_lock = threading.Lock()
_tracer: Any = None
_provider: Any = None
_span_kind: Any = None
_set_span_in_context: Any = None
_load_attempted = False


def force_flush(timeout_millis: int = 5000) -> None:
    """Best-effort flush of pending spans to the collector (verification / shutdown). No-op if
    tracing never initialized."""
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis)
        except Exception:  # noqa: BLE001
            _log.debug("voice tracing flush failed", exc_info=True)


def _primitive(value: Any) -> Optional[Any]:
    """Coerce to an OTLP-safe attribute value (str/bool/int/float), else drop it. Guards the
    content-free contract too: nothing but scalars ever reaches a span attribute here."""
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return value
    return None


class _Phase:
    __slots__ = ("name", "start_ns", "end_ns", "attrs", "events")

    def __init__(self, name: str, start_ns: int, attrs: Dict[str, Any]) -> None:
        self.name = name
        self.start_ns = start_ns
        self.end_ns: Optional[int] = None
        self.attrs: Dict[str, Any] = dict(attrs)
        self.events: List[Tuple[str, int, Dict[str, Any]]] = []

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def event(self, name: str, **attrs: Any) -> None:
        self.events.append((name, time.time_ns(), dict(attrs)))


class PhaseRecorder:
    """Collects per-turn phase timings/attributes, cheaply and always (independent of whether a
    tracer exists). :func:`emit_turn_trace` turns it into spans at the end of the turn."""

    def __init__(self, **turn_attrs: Any) -> None:
        self.turn_attrs: Dict[str, Any] = {k: v for k, v in turn_attrs.items() if v is not None}
        self.phases: List[_Phase] = []
        self.start_ns: Optional[int] = None
        self.end_ns: Optional[int] = None

    def set(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            if v is not None:
                self.turn_attrs[k] = v

    @contextmanager
    def phase(self, name: str, **attrs: Any):
        """Time a phase that runs inline (capture, STT). Yields the :class:`_Phase` so the caller
        can add attributes/events as it learns them."""
        p = _Phase(name, time.time_ns(), attrs)
        if self.start_ns is None:
            self.start_ns = p.start_ns
        try:
            yield p
        finally:
            p.end_ns = time.time_ns()
            self.end_ns = p.end_ns
            self.phases.append(p)

    def add_phase(self, name: str, start_ns: Optional[int], end_ns: Optional[int],
                  events: Optional[List[Tuple[str, int, Dict[str, Any]]]] = None,
                  **attrs: Any) -> None:
        """Record an already-timed phase (agent/TTS, whose start/end were captured at real code
        points, possibly overlapping). Skipped if either timestamp is missing."""
        if start_ns is None or end_ns is None:
            return
        p = _Phase(name, start_ns, attrs)
        p.end_ns = end_ns
        if events:
            p.events = events
        if self.start_ns is None or start_ns < self.start_ns:
            self.start_ns = start_ns
        if self.end_ns is None or end_ns > self.end_ns:
            self.end_ns = end_ns
        self.phases.append(p)


def _get_tracer():
    """Lazily build (once) the voice tracer from OTLP config; return (tracer, SpanKind,
    set_span_in_context) or (None, None, None) when disabled/unavailable."""
    global _tracer, _provider, _span_kind, _set_span_in_context, _load_attempted
    with _lock:
        if _load_attempted:
            return _tracer, _span_kind, _set_span_in_context
        _load_attempted = True
        try:
            from hermes_cli.config import load_config
            from agent.monitoring import otlp_exporter as ox

            config = load_config()
            if not ox.is_enabled(config):
                return None, None, None
            vt = ((config.get("monitoring") or {}).get("voice_tracing") or {})
            if not vt.get("enabled", True):  # default on when OTLP export is on
                return None, None, None
            sdk = ox._require_sdk(prompt=False)
            from opentelemetry.trace import set_span_in_context

            resource = sdk["Resource"].create(
                ox._runtime_resource_attributes(config, telemetry_scope="voice_converse"))
            provider = sdk["TracerProvider"](resource=resource)
            provider.add_span_processor(sdk["BatchSpanProcessor"](ox.build_exporter(config)))
            _provider = provider
            _tracer = provider.get_tracer(_INSTRUMENTATION)
            _span_kind = sdk["SpanKind"]
            _set_span_in_context = set_span_in_context
            _log.info("voice converse tracing enabled (OTLP spans)")
        except Exception as exc:  # noqa: BLE001 - never let tracing setup break voice
            _log.debug("voice tracing unavailable (%s)", exc, exc_info=True)
            _tracer = None
        return _tracer, _span_kind, _set_span_in_context


def _format_traceparent(span) -> Optional[str]:
    """W3C ``traceparent`` for a span: ``00-<trace_id:032x>-<span_id:016x>-<flags:02x>``. The
    client roots its own turn span under this so both sides land in one trace."""
    try:
        ctx = span.get_span_context()
        return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{int(ctx.trace_flags):02x}"
    except Exception:  # noqa: BLE001
        return None


class TurnTrace:
    """A LIVE ``voice.turn`` parent span whose ``traceparent`` is available immediately (so the
    server can stamp it on the transcript/speaking/turn_done frames and the client's turn span
    nests underneath). Capture/STT phases (already timed on the worker) are absorbed from the
    bound recorder; agent/TTS phases are added as the driver learns them; :meth:`end` materializes
    the children under the parent and closes it. A no-op stand-in (from :func:`start_turn` when
    tracing is off) still carries a ``turn_id`` and a ``None`` ``traceparent``."""

    def __init__(self, tracer, span_kind, set_ctx, parent, recorder, turn_id):
        self._tracer, self._kind, self._set_ctx = tracer, span_kind, set_ctx
        self._parent = parent
        self._recorder = recorder or PhaseRecorder()
        self.turn_id = turn_id
        self.traceparent = _format_traceparent(parent) if parent is not None else None

    def add_phase(self, name, start_ns, end_ns, **attrs):
        self._recorder.add_phase(name, start_ns, end_ns, **attrs)

    def set(self, **attrs):
        self._recorder.set(**attrs)

    def end(self, end_ns: Optional[int] = None) -> None:
        if self._parent is None:
            return
        last = self._recorder.start_ns or time.time_ns()
        try:
            for k, v in self._recorder.turn_attrs.items():
                pv = _primitive(v)
                if pv is not None:
                    self._parent.set_attribute(k, pv)
            ctx = self._set_ctx(self._parent)
            for ph in self._recorder.phases:
                child = self._tracer.start_span(
                    ph.name, context=ctx, kind=self._kind.INTERNAL, start_time=ph.start_ns)
                try:
                    for k, v in ph.attrs.items():
                        pv = _primitive(v)
                        if pv is not None:
                            child.set_attribute(k, pv)
                    for ev_name, ev_ts, ev_attrs in ph.events:
                        safe = {k: _primitive(v) for k, v in ev_attrs.items()
                                if _primitive(v) is not None}
                        child.add_event(ev_name, safe, timestamp=ev_ts)
                finally:
                    child.end(end_time=ph.end_ns or ph.start_ns)
                last = max(last, ph.end_ns or ph.start_ns)
        except Exception as exc:  # noqa: BLE001
            _log.debug("voice turn trace child build failed (%s)", exc, exc_info=True)
        finally:
            try:
                self._parent.end(end_time=end_ns or self._recorder.end_ns or last)
            except Exception:  # noqa: BLE001
                _log.debug("voice turn parent end failed", exc_info=True)


def start_turn(*, recorder: Optional["PhaseRecorder"] = None, turn_id: str,
               **turn_attrs: Any) -> TurnTrace:
    """Open a live ``voice.turn`` parent span rooted at the capture start (so the trace timeline
    begins where the user began speaking), returning a :class:`TurnTrace`. When tracing is off the
    returned trace is inert (``traceparent=None``) but still carries ``turn_id``. Never raises."""
    tracer, span_kind, set_ctx = _get_tracer()
    if tracer is None:
        return TurnTrace(None, None, None, None, recorder, turn_id)
    try:
        start_ns = (recorder.start_ns if recorder and recorder.start_ns else time.time_ns())
        parent = tracer.start_span("voice.turn", kind=span_kind.SERVER, start_time=start_ns)
        tt = TurnTrace(tracer, span_kind, set_ctx, parent, recorder, turn_id)
        for k, v in turn_attrs.items():
            tt._recorder.set(**{k: v})
        return tt
    except Exception as exc:  # noqa: BLE001
        _log.debug("voice start_turn failed (%s)", exc, exc_info=True)
        return TurnTrace(None, None, None, None, recorder, turn_id)


def emit_turn_trace(recorder: Optional["PhaseRecorder"]) -> None:
    """Materialize a recorder into a ``voice.turn`` span + one child span per recorded phase, with
    their explicit start/end times. No-op if there's nothing to emit or tracing is off. Never
    raises — telemetry must not affect the turn."""
    if recorder is None or not recorder.phases:
        return
    tracer, span_kind, set_ctx = _get_tracer()
    if tracer is None:
        return
    try:
        start_ns = recorder.start_ns or min(p.start_ns for p in recorder.phases)
        end_ns = recorder.end_ns or max((p.end_ns or p.start_ns) for p in recorder.phases)
        parent = tracer.start_span("voice.turn", kind=span_kind.SERVER, start_time=start_ns)
        try:
            for k, v in recorder.turn_attrs.items():
                pv = _primitive(v)
                if pv is not None:
                    parent.set_attribute(k, pv)
            ctx = set_ctx(parent)
            for ph in recorder.phases:
                child = tracer.start_span(
                    ph.name, context=ctx, kind=span_kind.INTERNAL, start_time=ph.start_ns)
                try:
                    for k, v in ph.attrs.items():
                        pv = _primitive(v)
                        if pv is not None:
                            child.set_attribute(k, pv)
                    for ev_name, ev_ts, ev_attrs in ph.events:
                        safe = {k: _primitive(v) for k, v in ev_attrs.items()}
                        child.add_event(ev_name, {k: v for k, v in safe.items() if v is not None},
                                        timestamp=ev_ts)
                finally:
                    child.end(end_time=ph.end_ns or ph.start_ns)
        finally:
            parent.end(end_time=end_ns)
    except Exception as exc:  # noqa: BLE001
        _log.debug("voice turn trace emit failed (%s)", exc, exc_info=True)
