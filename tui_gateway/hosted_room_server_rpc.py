"""In-process session adapter for the hosted room driver: the room worker uses the same
installed session handlers as every TUI/Desktop turn (no WebSocket transport), passing the
task proof as an in-process-only Python object that JSON clients cannot forge."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from types import ModuleType
from pathlib import Path
from typing import Any, Callable

from gateway import hosted_room_driver as state

_LockType = type(threading.Lock())


class HostedRoomSessionError(RuntimeError):
    """Raised when an in-process session operation is rejected."""

    def __init__(self, method: str, code: int, message: str) -> None:
        super().__init__(f"{method} failed: {message}")
        self.method = method
        self.code = code


class HostedRoomServerRPC:
    """Normalize the installed server handlers for :class:`HostedRoomRuntime`."""

    def __init__(self, server: ModuleType, *, db_path: Path | str | None = None) -> None:
        self.server = server
        self.db_path = db_path
        self._ids = itertools.count(1)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        envelope = self.server._methods[method](f"hosted-room-{next(self._ids)}", params)
        if not isinstance(envelope, dict):
            envelope = {}
        error = envelope.get("error")
        if isinstance(error, dict):
            raise HostedRoomSessionError(
                method, int(error.get("code") or 5000),
                str(error.get("message") or "gateway rejected the request"))
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise HostedRoomSessionError(method, 5000, "gateway returned no result")
        return result

    def resolve_exact(self, *, profile: str, title: str, source: str) -> Mapping[str, Any] | None:
        del source
        if self.db_path is not None:
            from gateway.hosted_room_local_sessions import LocalSessionBindingError, lookup_binding
            if not title.startswith("Group: "):
                raise LocalSessionBindingError("The Group Chat session title is invalid.")
            binding = lookup_binding(self.db_path, room_id=title.removeprefix("Group: "), profile=profile)
            if binding is not None:
                with self.server._profile_db({"profile": profile}) as db:
                    chain = self._verified_context_chain(db, binding["session_id"], binding)
                return {"session_id": chain[-1][0], "title": title}
        result = self._call(
            "session.list", {"profile": profile, "title": title, "include_hidden": True})
        rows = result.get("sessions")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        row = rows[0]
        return {"session_id": row.get("resolved_id") or row.get("id"),
                "title": row.get("title") or title}

    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]:
        return self._call("session.create", {
            "profile": profile, "title": title, "source": source, "hidden": True,
            "room_plumbing": True, "follow_profile_config": True, "close_on_disconnect": False})

    def resume(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        return self._call("session.resume", {
            "profile": profile, "session_id": session_id, "omit_messages": True, "source": source})

    def submit(
        self, *, profile: str, session_id: str, prompt: str, source: str, task: state.TaskIdentity,
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None],
    ) -> Mapping[str, Any]:
        try:
            params = {
                "profile": profile, "session_id": session_id, "text": prompt, "source": source,
                "_hosted_task": {
                    "room_id": task.room_id, "task_id": task.task_id, "thread_id": task.thread_id,
                    "turn_id": task.turn_id, "execution_generation": execution_generation},
                "_hosted_terminal_callback": on_terminal}
            if self.db_path is not None:
                params["_hosted_session_guard"] = lambda session: self._record_session_binding(
                    session, profile=profile, task=task, execution_generation=execution_generation)
            return self._call("prompt.submit", params)
        except HostedRoomSessionError as exc:
            # In-process prompt.submit error envelopes come back before the background turn is
            # admitted; keep that proof so the driver can defer/requeue without an ambiguity lease.
            exc.not_admitted = True
            raise

    def _record_session_binding(self, session, *, profile, task, execution_generation):
        """Called by prompt.submit while it owns the actual session's history lock."""
        from gateway.hosted_room_local_sessions import LocalSessionBindingError, lookup_binding, record_binding
        if session.get("source") != "bot_room" or self.server._response_profile_name(profile) != profile:
            raise LocalSessionBindingError("The Bot session does not match the requested profile.")
        with self.server._session_db(session) as db, self.server._profile_db({"profile": profile}) as expected_db:
            if db is None or expected_db is None or Path(db.db_path).resolve() != Path(expected_db.db_path).resolve():
                raise LocalSessionBindingError("The Bot session's profile store is unavailable or changed.")
            stored_id = session.get("session_key")
            binding = lookup_binding(self.db_path, room_id=task.room_id, profile=profile)
            current = db.get_session(stored_id) if isinstance(stored_id, str) else None
            if current is None and binding is None and not session.get("lazy") and session.get("room_plumbing") is True:
                # Ordinary drafts stay lazy. This is the first real hosted submit,
                # and only its freshly created session may be persisted here.
                if session.get("pending_title") != f"Group: {task.room_id}" or self.server._ensure_session_db_row(session) is False:
                    raise LocalSessionBindingError("The new Group Chat session could not be saved.")
                current = db.get_session(stored_id)
            if current is None or current.get("source") != "bot_room" or current.get("archived"):
                raise LocalSessionBindingError("The private Bot session is unavailable.")
            anchor = binding["session_id"] if binding is not None else stored_id
            chain = self._verified_context_chain(db, anchor, binding)
            if chain[-1][0] != stored_id:
                raise LocalSessionBindingError("The private Bot context no longer matches its recorded identity.")
            record_binding(self.db_path, task=task, execution_generation=execution_generation,
                           profile=profile, session_id=anchor, session_started_at=chain[0][1], context_chain=chain)

    @staticmethod
    def _verified_context_chain(db, anchor, binding):
        from gateway.hosted_room_local_sessions import LocalSessionBindingError
        if db is None:
            raise LocalSessionBindingError("The private Bot conversation store is unavailable.")
        chain = []
        for key in db.get_compression_chain(anchor):
            row = db.get_session(key)
            if row is None or row.get("source") != "bot_room" or row.get("archived"):
                raise LocalSessionBindingError("The private Bot continuation is unavailable.")
            chain.append((key, row["started_at"]))
        if not chain or row.get("end_reason") == "compression":
            raise LocalSessionBindingError("The private Bot continuation is incomplete.")
        if binding is not None and (
            chain[0] != (binding["session_id"], binding["session_started_at"])
            or (binding["last_session_id"], binding["last_session_started_at"]) not in chain
        ):
            raise LocalSessionBindingError("The private Bot continuation was lost or replaced.")
        return chain

    def history(self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]:
        del source
        result = self._call("session.history", {"profile": profile, "session_id": session_id})
        rows = result.get("messages")
        return tuple(row for row in rows if isinstance(row, dict)) if isinstance(rows, list) else ()

    def _session_record(self, session_id: str) -> dict[str, Any] | None:
        with self.server._sessions_lock:
            record = self.server._sessions.get(session_id)
            if record is not None:
                return record
            return next((c for c in self.server._sessions.values()
                         if str(c.get("session_key") or "") == session_id), None)

    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        del profile, source
        record = self._session_record(session_id)
        if record is None:
            return {"active": False, "task_id": None}
        lock = record.get("history_lock")
        if not isinstance(lock, _LockType):
            return {"active": bool(record.get("running")), "task_id": None}
        with lock:
            task = record.get("_hosted_room_task")
            result = {"active": bool(record.get("running")),
                      "task_id": task.get("task_id") if isinstance(task, dict) else None}
            pending_reader = getattr(self.server, "_pending_approval_request_payload", None)
            if callable(pending_reader) and (pending := pending_reader(str(record.get("session_key") or ""))):
                result["status"] = "waiting_for_approval"
                result["pending_approval"] = pending
            return result

    def approve(self, *, session_id: str, request_id: str, choice: str) -> Mapping[str, Any]:
        """Resolve one exact local room approval without broad policy changes."""
        return self._call("approval.respond", {
            "session_id": session_id, "request_id": request_id, "choice": choice, "all": False})

    def interrupt(
        self, *, profile: str, session_id: str, source: str, expected_task_id: str
    ) -> Mapping[str, Any] | None:
        del source
        return self._call("session.interrupt", {
            "profile": profile, "session_id": session_id,
            "expected_hosted_task_id": expected_task_id})
