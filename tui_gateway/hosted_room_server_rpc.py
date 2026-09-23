"""In-process session adapter for the hosted room driver: the room worker uses the same
installed session handlers as every TUI/Desktop turn (no WebSocket transport), passing the
task proof as an in-process-only Python object that JSON clients cannot forge."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, Callable

from gateway import hosted_room_driver as state
from tui_gateway.session_registry import _live_profile_matches

_LockType = type(threading.Lock())


class HostedRoomSessionError(RuntimeError):
    """Raised when an in-process session operation is rejected."""

    def __init__(self, method: str, code: int, message: str) -> None:
        super().__init__(f"{method} failed: {message}")
        self.method = method
        self.code = code


class HostedRoomServerRPC:
    """Normalize the installed server handlers for :class:`HostedRoomRuntime`."""

    def __init__(self, server: ModuleType) -> None:
        self.server = server
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
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None], member_id: str,
    ) -> Mapping[str, Any]:
        try:
            return self._call("prompt.submit", {
                "profile": profile, "session_id": session_id, "text": prompt, "source": source,
                "_hosted_task": {
                    "room_id": task.room_id, "task_id": task.task_id, "thread_id": task.thread_id,
                    "turn_id": task.turn_id, "execution_generation": execution_generation,
                    "member_id": member_id},
                "_hosted_terminal_callback": on_terminal})
        except HostedRoomSessionError as exc:
            # In-process prompt.submit error envelopes come back before the background turn is
            # admitted; keep that proof so the driver can defer/requeue without an ambiguity lease.
            exc.not_admitted = True
            raise

    def history(self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]:
        del source
        result = self._call("session.history", {"profile": profile, "session_id": session_id})
        rows = result.get("messages")
        return tuple(row for row in rows if isinstance(row, dict)) if isinstance(rows, list) else ()

    def _session_record(self, session_id: str, *, profile: str) -> tuple[str, dict[str, Any]] | None:
        home = self.server._profile_home(profile)
        with self.server._sessions_lock:
            record = self.server._sessions.get(session_id)
            if record is not None:
                if not _live_profile_matches(record, home):
                    raise HostedRoomSessionError("session.info", 4001, "session profile mismatch")
                return session_id, record
            return next(((sid, c) for sid, c in self.server._sessions.items()
                         if str(c.get("session_key") or "") == session_id
                         and _live_profile_matches(c, home)), None)

    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        del source
        selected = self._session_record(session_id, profile=profile)
        if selected is None:
            return {"active": False, "task_id": None, "execution_generation": None}
        runtime_id, record = selected
        # The request registry (including compute-host mirrors) owns runtime-ID
        # routing. The approval queue has only a stored key, which can coincide
        # across profiles. Never use that queue to recover private prompt data.
        reader = getattr(self.server, "_open_requests", None)
        requests = reader(runtime_id) if callable(reader) else []
        if not isinstance(requests, list) or any(not isinstance(row, dict) for row in requests):
            raise HostedRoomSessionError("session.info", 4001, "request snapshot unavailable")
        pending = next((dict(row["params"]) for row in requests
                        if row.get("method") == "approval"
                        and isinstance(row.get("params"), dict)
                        and row["params"].get("session_id") == runtime_id), None)
        with self.server._sessions_lock:
            if self.server._sessions.get(runtime_id) is not record:
                raise HostedRoomSessionError("session.info", 4001, "session owner changed")
            lock = record.get("history_lock")
            if isinstance(lock, _LockType):
                with lock:
                    task = record.get("_hosted_room_task")
                    result = {"active": bool(record.get("running")),
                              "task_id": task.get("task_id") if isinstance(task, dict) else None,
                              "execution_generation": task.get("execution_generation") if isinstance(task, dict) else None}
                    key = str(record.get("session_key") or "")
            else:
                result = {"active": bool(record.get("running")), "task_id": None,
                          "execution_generation": None}
                key = str(record.get("session_key") or "")
        if pending is not None:
            result["status"] = "waiting_for_approval"
            result["pending_approval"] = pending
        else:
            from tools.approval import has_blocking_approval

            if has_blocking_approval(key):
                # Absence from the owned request registry does not prove idle:
                # publication may still be in progress, or belong to another home.
                result["status"] = "unknown"
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
