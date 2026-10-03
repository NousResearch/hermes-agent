"""Execution traces (``methods_trace.py``): the NeMo Relay events Hermes recorded for a session.

Events are Relay's own ATOF records, bounded but never reshaped, so they stay ``JsonValue``: the
producer that owns their shape is Relay, not this contract.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Payload, Result
from .common import SessionParams
from .registry import event, method


class TraceEventsParams(SessionParams):
    """``session_id`` is a live gateway session id or a stored session id; a live one is also
    subscribed, so its ``trace.event`` frames start flowing."""


class TraceEventsResult(Result):
    session_id: str
    """The stored session the events belong to (every delegated descendant is included)."""
    events: list[JsonValue] = Field(default_factory=list)
    truncated: bool = False
    recording: bool = True
    """False when this profile has trace recording off (``telemetry.traces.enabled``)."""


method("trace.events", params=TraceEventsParams, result=TraceEventsResult,
       doc="Recorded Relay events for one session and its subagents, oldest first.")


class TraceEventPayload(Payload):
    """``methods_trace._forward_trace_event`` — one Relay event recorded for a watched session."""

    trace_session_id: str
    """The watched (root) stored session; ``event`` may belong to one of its subagents."""
    event: JsonValue


event("trace.event", TraceEventPayload, doc="A Relay event recorded live for a session a client is tracing.")
