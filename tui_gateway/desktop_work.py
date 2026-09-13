"""Read-only Desktop work events. Identity is explicit, never inferred from a session.

Unsupported continuations have no admission proof and cannot inherit a newer turn.
This metadata never enters messages, prompts, model kwargs or persistent history.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import UUID, NAMESPACE_URL, uuid5
from pathlib import Path
import logging
from threading import RLock

logger = logging.getLogger(__name__)


current_work = ContextVar("desktop_work", default=None)


@dataclass
class DesktopWork:
    root_id: str
    sid: str
    stored_session_id: str
    profile: str
    title: str
    emitted: set = field(default_factory=set)
    session: dict | None = field(default=None, repr=False)

    pending: set = field(default_factory=set)
    active_turns: int = 0
    closed: bool = False
    _candidate: tuple | None = field(default=None, repr=False)
    _publish: object | None = field(default=None, repr=False)
    _lock: object = field(default_factory=RLock, repr=False)

    def begin_turn(self):
        with self._lock:
            self.active_turns += 1
            self._candidate = None

    def retain_async(self, delegation_id):
        with self._lock:
            self.pending.add(delegation_id)

    def consume_async(self, delegation_id):
        with self._lock:
            self.pending.discard(delegation_id)
            if not self.pending and not self.active_turns and self._candidate:
                publish, kind = self._candidate
                self.emit(publish, kind)

    def abandon_async(self, delegation_id):
        """Close a root whose child result cannot be delivered in this process."""
        with self._lock:
            self.pending.discard(delegation_id)
            if self._publish is not None:
                self._emit_locked(self._publish, "interrupted")

    def end_turn(self, publish, kind):
        with self._lock:
            self.active_turns = max(0, self.active_turns - 1)
            self._candidate = (publish, kind)
            if kind == "completed" and (self.pending or self.active_turns):
                self.emit(publish, "waiting")
            else:
                self.emit(publish, kind)

    def emit(self, publish, kind, request_id=None):
        with self._lock:
            self._publish = publish
            self._emit_locked(publish, kind, request_id)

    def _emit_locked(self, publish, kind, request_id=None):
        key = (kind, request_id)
        if self.closed or key in self.emitted:
            return
        if kind in {"completed", "provider_failed", "interrupted"}:
            self.closed = True
        self.emitted.add(key)
        payload = {
            "schema_version": 1,
            "event_id": str(uuid5(NAMESPACE_URL, f"desktop.work:{self.profile}:{self.root_id}:{kind}:{request_id or ''}")),
            "root_id": self.root_id, "stored_session_id": self.stored_session_id,
            "profile": self.profile, "origin": "desktop_user", "kind": kind,
            "title": read_title(self.session) if self.session is not None else self.title,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        try:
            publish("desktop.work", self.sid, payload)
        except Exception:
            logger.warning("Desktop work observer failed", exc_info=True)


def read_title(session):
    from tui_gateway import server
    title = None
    try:
        with server._session_db(session) as db:
            title = db.get_session_title(session.get("session_key")) if db else None
    except Exception:
        logger.debug("Desktop work title unavailable", exc_info=True)
    if not title:
        title = session.get("pending_title") or session.get("title")
    return " ".join(str(title or "Chat senza titolo").split())[:160] or "Chat senza titolo"


def admit(sid, session, proof, display_kind=None):
    if (not isinstance(proof, dict) or proof.get("origin") != "desktop_user"
            or session.get("source") != "desktop" or display_kind is not None
            or session.get("hidden") or session.get("parent_session_id") or session.get("lazy")):
        return None
    root = proof.get("root_id")
    if not isinstance(root, str):
        return None
    try:
        if str(UUID(root)) != root:
            return None
    except ValueError:
        return None
    stored = session.get("session_key")
    if not isinstance(stored, str) or not stored:
        return None
    from tui_gateway import server
    profile = (server._response_profile_name(Path(session["profile_home"]).name)
               if session.get("profile_home") else server._current_profile_name())
    return DesktopWork(root, sid, stored, profile, read_title(session), session=session)


def interrupt_before_turn(sid, session, proof, display_kind, publish):
    """Close an accepted human root that cannot reach the agent turn."""
    work = admit(sid, session, proof, display_kind)
    if work is None:
        return
    work.begin_turn()
    work.emit(publish, "started")
    finish(work, publish, {"interrupted": True})


def finish(work, publish, result, *, uncertain=False, terminal_surface=None):
    if work is None:
        return
    result = result if isinstance(result, dict) else {}
    if terminal_surface is not None:
        kind = "provider_failed" if terminal_surface.get("code") not in {None, "unknown"} and terminal_surface.get("layer") in {
            "provider", "auth", "billing", "streaming", "endpoint"
        } else "interrupted"
    elif result.get("interrupted"):
        kind = "interrupted"
    elif result.get("error") or result.get("failed"):
        # Unknown dispatcher/tool errors must never masquerade as provider quota.
        from agent.error_surface import build_error_surface_from_result
        surface = (build_error_surface_from_result(result)
                   if result.get("failure_reason") or result.get("billing_block") else None)
        kind = "provider_failed" if surface and surface.get("layer") in {
            "provider", "auth", "billing", "streaming", "endpoint"
        } else "interrupted"
    elif uncertain or result.get("completed") is not True:
        kind = "waiting"
    else:
        kind = "completed"
    work.end_turn(publish, kind)
