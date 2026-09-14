"""In-process session adapter for the hosted room driver."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, Callable

from gateway import hosted_room_driver as state
from tui_gateway.contracts.prompt_voice import ApprovalRespondParams, PromptSubmitParams
from tui_gateway.contracts.sessions import (
    SessionCreateParams, SessionHistoryParams, SessionInterruptParams, SessionListParams,
    SessionResumeParams,
)

_LockType = type(threading.Lock())


class HostedRoomSessionError(RuntimeError):
    """Raised when an in-process session operation is rejected."""

    def __init__(self, method: str, code: int, message: str) -> None:
        super().__init__(f"{method} failed: {message}")
        self.method = method
        self.code = code


class HostedRoomServerRPC:
    """Normalize installed model-in/model-out session handlers for :class:`HostedRoomRuntime`."""

    def __init__(self, server: ModuleType) -> None:
        self.server = server
        self._ids = itertools.count(1)

    def _call(self, method: str, params, **trusted) -> dict[str, Any]:
        result = self.server.invoke(method, params, **trusted)
        if isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, dict):
                raise HostedRoomSessionError(
                    method, int(error.get("code") or 5000),
                    str(error.get("message") or "gateway rejected the request"))
            raise HostedRoomSessionError(method, 5000, "gateway returned an invalid error frame")
        return result.model_dump(mode="json")

    def resolve_exact(self, *, profile: str, title: str, source: str) -> Mapping[str, Any] | None:
        del source
        result = self._call("session.list", SessionListParams(profile=profile, title=title, include_hidden=True))
        rows = result.get("sessions")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        row = rows[0]
        return {"session_id": row.get("resolved_id") or row.get("id"), "title": row.get("title") or title}

    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]:
        return self._call("session.create", SessionCreateParams(
            profile=profile, title=title, source=source, hidden=True, room_plumbing=True,
            follow_profile_config=True, close_on_disconnect=False))

    def resume(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        return self._call("session.resume", SessionResumeParams(
            profile=profile, session_id=session_id, omit_messages=True, source=source))

    def submit(
        self, *, profile: str, session_id: str, prompt: str, source: str, task: state.TaskIdentity,
        execution_generation: int, on_terminal: Callable[[Mapping[str, Any]], None], member_id: str,
    ) -> Mapping[str, Any]:
        del source
        try:
            return self._call(
                "prompt.submit", PromptSubmitParams(profile=profile, session_id=session_id, text=prompt),
                _hosted_task={
                    "room_id": task.room_id, "task_id": task.task_id, "thread_id": task.thread_id,
                    "turn_id": task.turn_id, "execution_generation": execution_generation, "member_id": member_id},
                _hosted_terminal_callback=on_terminal,
            )
        except HostedRoomSessionError as exc:
            exc.not_admitted = True
            raise

    def history(self, *, profile: str, session_id: str, source: str) -> Sequence[Mapping[str, Any]]:
        del source
        result = self._call("session.history", SessionHistoryParams(profile=profile, session_id=session_id))
        rows = result.get("messages")
        return tuple(row for row in rows if isinstance(row, dict)) if isinstance(rows, list) else ()

    def _session_record(self, session_id: str) -> dict[str, Any] | None:
        with self.server._sessions_lock:
            record = self.server._sessions.get(session_id)
            if record is not None:
                return record
            return next((candidate for candidate in self.server._sessions.values()
                         if str(candidate.get("session_key") or "") == session_id), None)

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
        return self._call("approval.respond", ApprovalRespondParams(
            session_id=session_id, request_id=request_id, choice=choice, all=False))

    def interrupt(
        self, *, profile: str, session_id: str, source: str, expected_task_id: str
    ) -> Mapping[str, Any] | None:
        del source
        return self._call("session.interrupt", SessionInterruptParams(
            profile=profile, session_id=session_id, expected_hosted_task_id=expected_task_id))
