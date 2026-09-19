"""Display-only execution provenance. Never an identity, lease, or durable receipt.

No session-wide last-origin fallback: an event without a captured execution stays
unattributed. Each wire dict is a fresh copy so replay cannot observe later mutation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from _thread import RLock
from uuid import uuid4


def connection_source(transport) -> dict:
    socket_id = getattr(transport, "socket_id", None)
    return ({"kind": "connection", "socket_id": socket_id}
            if isinstance(socket_id, str) and socket_id else {"kind": "unknown"})


def merge_sources(first: dict, second: dict) -> dict:
    return dict(first) if first == second else {"kind": "mixed"}


@dataclass
class TurnObservation:
    sid: str
    session: dict
    id: str
    source_kind: str
    socket_id: str | None = None
    inputs: dict | None = None
    gate: RLock = field(default_factory=RLock)
    terminal: bool = False
    terminal_status: str | None = None
    registered: bool = False

    def wire(self) -> dict:
        source = {"kind": self.source_kind}
        if self.socket_id:
            source["socket_id"] = self.socket_id
        return {"id": self.id, "source": source}

    def terminal_evidence(self) -> dict | None:
        """Text-free result for the trusted compute protocol, never inferred from worker exit."""
        with self.gate:
            return ({"turn": self.wire(), "status": self.terminal_status}
                    if self.terminal else None)

    def is_current(self) -> bool:
        inflight = self.session.get("inflight_turn") or {}
        return (not self.terminal and self.owns_record() and inflight.get("streaming") is True
                and inflight.get("turn", {}).get("id") == self.id)

    def owns_record(self):
        from tui_gateway import server
        return not self.registered or server._sessions.get(self.sid) is self.session

    @contextmanager
    def publication(self):
        with self.gate:
            yield self.is_current()


_current: ContextVar[TurnObservation | None] = ContextVar("turn_observation", default=None)


def make_turn(sid: str, session: dict, source: dict | None = None, *, wire: dict | None = None, inputs=None):
    """``wire`` comes only from the parent compute-host dispatch, never RPC params."""
    source = (wire or {}).get("source") or source or {"kind": "unknown"}
    from tui_gateway import server
    return TurnObservation(sid, session, (wire or {}).get("id") or uuid4().hex,
                           source["kind"], source.get("socket_id"), inputs=inputs,
                           registered=server._sessions.get(sid) is session)


@contextmanager
def turn_scope(turn):
    token = _current.set(turn)
    try:
        yield
    finally:
        _current.reset(token)


def stamp_turn(params: dict, sessions: dict) -> None:
    turn = _current.get()
    if (turn is not None and params.get("session_id") == turn.sid and sessions.get(turn.sid) is turn.session
            and (not turn.terminal or params.get("type") == "message.complete")):
        params["turn"] = turn.wire()


def emit_observed(event, sid, payload, write):
    """Serialize final publication, including the terminal, after slow rendering."""
    turn = _current.get()
    if turn is None or turn.sid != sid:
        return write(event, sid, payload)
    with turn.gate:
        if not turn.owns_record():
            return False
        if turn.terminal:
            if event.startswith(("message.", "tool.")):
                return False
            with turn_scope(None):
                return write(event, sid, payload)
        if event == "message.complete":
            from tui_gateway.input_observation import project_inputs, record_outcome
            with turn.session["history_lock"]:
                payload = {**(payload or {}), **project_inputs(turn.session, turn.inputs)}
                record_outcome(turn.session, turn.inputs, "terminal", turn=turn.wire(),
                               status=payload.get("status", "complete"))
                if isinstance(inflight := turn.session.get("inflight_turn"), dict):
                    inflight["streaming"] = False
                turn.terminal_status = payload.get("status", "complete")
                turn.terminal = True
        return write(event, sid, payload)


def capture_turn_callback(callback):
    """Provider callbacks can run on another thread, or survive into the next turn."""
    turn = _current.get()

    @wraps(callback)
    def captured(*args, **kwargs):
        with turn_scope(turn):
            # Check and mutation share the history lock inside the callback where needed.
            if turn is None or turn.is_current():
                return callback(*args, **kwargs)
        return None

    return captured


def current_turn():
    return _current.get()
