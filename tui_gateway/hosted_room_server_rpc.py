"""In-process session adapter for the hosted room driver.

The room worker must not depend on a Desktop/WebSocket transport, but it should
still use the same session handlers as every other TUI/Desktop turn. This
adapter calls the installed handler registry directly and keeps the extra
task proof as an in-process-only Python object that JSON clients cannot forge.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, Callable

from gateway import hosted_room_driver as state
from tui_gateway.hosted_room_driver import HostedRoomProfileUnavailableError
from tui_gateway.transport import Transport, bind_transport, reset_transport


class _InternalDropTransport:
    """Accept internal frames without publishing private room traffic."""

    def write(self, obj: dict) -> bool:
        del obj
        return True

    def close(self) -> None:
        return None


class HostedRoomSessionError(RuntimeError):
    """Raised when an in-process session operation is rejected."""

    def __init__(self, method: str, code: int, message: str) -> None:
        super().__init__(f"{method} failed: {message}")
        self.method = method
        self.code = code
        self.not_admitted = False


class HostedRoomServerRPC:
    """Normalize the installed server handlers for :class:`HostedRoomRuntime`."""

    def __init__(
        self,
        server: ModuleType,
        *,
        profile_available: Callable[[str], bool] | None = None,
    ) -> None:
        self.server = server
        self.profile_available = profile_available or (lambda _profile: True)
        self._ids = itertools.count(1)
        self._transport: Transport = _InternalDropTransport()

    def _require_profile_available(self, profile: str) -> None:
        try:
            available = self.profile_available(profile)
        except Exception as exc:
            raise HostedRoomProfileUnavailableError(
                "hosted room target profile availability could not be verified"
            ) from exc
        if not available:
            raise HostedRoomProfileUnavailableError(
                "hosted room target profile is unavailable"
            )

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        handler = self.server._methods[method]
        token = bind_transport(self._transport)
        try:
            envelope = handler(f"hosted-room-{next(self._ids)}", params)
        finally:
            reset_transport(token)
        error = envelope.get("error") if isinstance(envelope, dict) else None
        if isinstance(error, dict):
            raise HostedRoomSessionError(
                method,
                int(error.get("code") or 5000),
                str(error.get("message") or "gateway rejected the request"),
            )
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(result, dict):
            raise HostedRoomSessionError(method, 5000, "gateway returned no result")
        return result

    def resolve_exact(
        self, *, profile: str, title: str, source: str
    ) -> Mapping[str, Any] | None:
        self._require_profile_available(profile)
        result = self._call(
            "session.list",
            {
                "profile": profile,
                "title": title,
                "source": source,
                "include_hidden": True,
            },
        )
        rows = result.get("sessions")
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        if not isinstance(row, dict):
            return None
        session_id = row.get("resolved_id") or row.get("id")
        return {"session_id": session_id, "title": row.get("title") or title}

    def create(self, *, profile: str, title: str, source: str) -> Mapping[str, Any]:
        self._require_profile_available(profile)
        return self._call(
            "session.create",
            {
                "profile": profile,
                "title": title,
                "source": source,
                "hidden": True,
                "room_plumbing": True,
                "follow_profile_config": True,
                "close_on_disconnect": False,
            },
        )

    def resume(
        self, *, profile: str, session_id: str, source: str
    ) -> Mapping[str, Any]:
        self._require_profile_available(profile)
        return self._call(
            "session.resume",
            {
                "profile": profile,
                "session_id": session_id,
                "omit_messages": True,
                "source": source,
            },
        )

    def submit(
        self,
        *,
        profile: str,
        session_id: str,
        prompt: str,
        source: str,
        task: state.TaskIdentity,
        execution_generation: int,
        on_terminal: Callable[[Mapping[str, Any]], None],
    ) -> Mapping[str, Any]:
        self._require_profile_available(profile)
        try:
            return self._call(
                "prompt.submit",
                {
                    "profile": profile,
                    "session_id": session_id,
                    "text": prompt,
                    "source": source,
                    "_hosted_task": {
                        "room_id": task.room_id,
                        "task_id": task.task_id,
                        "thread_id": task.thread_id,
                        "turn_id": task.turn_id,
                        "execution_generation": execution_generation,
                    },
                    "_hosted_terminal_callback": on_terminal,
                },
            )
        except HostedRoomSessionError as exc:
            # In-process prompt.submit error envelopes are returned before the
            # background turn is admitted. Preserve that proof so the driver
            # can defer or requeue without waiting out an ambiguity lease.
            exc.not_admitted = True
            raise

    def history(
        self, *, profile: str, session_id: str, source: str
    ) -> Sequence[Mapping[str, Any]]:
        self._require_profile_available(profile)
        del source
        result = self._call(
            "session.history",
            {"profile": profile, "session_id": session_id},
        )
        rows = result.get("messages")
        return tuple(row for row in rows if isinstance(row, dict)) if isinstance(rows, list) else ()

    def _session_record(self, session_id: str) -> dict[str, Any] | None:
        with self.server._sessions_lock:
            record = self.server._sessions.get(session_id)
            if record is not None:
                return record
            for candidate in self.server._sessions.values():
                if str(candidate.get("session_key") or "") == session_id:
                    return candidate
        return None

    def info(self, *, profile: str, session_id: str, source: str) -> Mapping[str, Any]:
        del profile, source
        record = self._session_record(session_id)
        if record is None:
            return {"active": False, "task_id": None}
        lock = record.get("history_lock")
        if not isinstance(lock, type(threading.Lock())):
            return {"active": bool(record.get("running")), "task_id": None}
        with lock:
            task = record.get("_hosted_room_task")
            result = {
                "active": bool(record.get("running")),
                "task_id": task.get("task_id") if isinstance(task, dict) else None,
            }
            pending_reader = getattr(
                self.server, "_pending_approval_request_payload", None
            )
            pending = (
                pending_reader(str(record.get("session_key") or ""))
                if callable(pending_reader)
                else None
            )
            if pending:
                result["status"] = "waiting_for_approval"
                result["pending_approval"] = pending
            return result

    def approve(
        self,
        *,
        session_id: str,
        request_id: str,
        choice: str,
    ) -> Mapping[str, Any]:
        """Resolve one exact local room approval without broad policy changes."""
        return self._call(
            "approval.respond",
            {
                "session_id": session_id,
                "request_id": request_id,
                "choice": choice,
                "all": False,
            },
        )

    def _find_admitted_session(
        self,
        *,
        task: state.TaskIdentity,
        execution_generation: int,
        source: str,
    ) -> tuple[str, bool] | None:
        method = "session.interrupt_admitted"
        if source != "bot_room":
            raise HostedRoomSessionError(method, 4000, "source must be bot_room")
        if (
            isinstance(execution_generation, bool)
            or not isinstance(execution_generation, int)
            or execution_generation <= 0
        ):
            raise HostedRoomSessionError(
                method,
                4000,
                "execution_generation must be a positive integer",
            )

        expected_identity = (
            task.room_id,
            task.task_id,
            task.thread_id,
            task.turn_id,
        )
        matches: list[tuple[str, bool]] = []
        generation_conflicts: list[str] = []
        lock_type = type(threading.Lock())

        # Admission proof is process-local. Lock membership first, then each
        # session record, so deleted profiles and the persisted session index
        # are not involved in cancellation ownership.
        with self.server._sessions_lock:
            candidates = tuple(self.server._sessions.items())
        for session_id, record in candidates:
            if not isinstance(record, dict):
                continue
            if (
                record.get("room_plumbing") is not True
                or record.get("source") != source
            ):
                continue
            lock = record.get("history_lock")
            if not isinstance(lock, lock_type):
                raise HostedRoomSessionError(
                    method,
                    4090,
                    "admitted hosted room session has no usable history lock",
                )
            with lock:
                if (
                    record.get("room_plumbing") is not True
                    or record.get("source") != source
                ):
                    continue
                proof = record.get("_hosted_room_task")
                if not isinstance(proof, Mapping):
                    continue
                observed_identity = (
                    proof.get("room_id"),
                    proof.get("task_id"),
                    proof.get("thread_id"),
                    proof.get("turn_id"),
                )
                if observed_identity != expected_identity:
                    continue
                observed_generation = proof.get("execution_generation")
                if (
                    isinstance(observed_generation, bool)
                    or not isinstance(observed_generation, int)
                    or observed_generation != execution_generation
                ):
                    generation_conflicts.append(str(session_id))
                    continue
                matches.append((str(session_id), bool(record.get("running"))))

        if generation_conflicts:
            raise HostedRoomSessionError(
                method,
                4092,
                "admitted hosted room task has a conflicting execution generation",
            )
        if len(matches) > 1:
            raise HostedRoomSessionError(
                method,
                4091,
                "admitted hosted room task is ambiguous across local sessions",
            )
        return matches[0] if matches else None

    def interrupt_admitted(
        self,
        *,
        task: state.TaskIdentity,
        execution_generation: int,
        source: str,
    ) -> Mapping[str, Any]:
        """Interrupt one exact process-local admission without profile lookup."""
        match = self._find_admitted_session(
            task=task,
            execution_generation=execution_generation,
            source=source,
        )
        if match is None:
            return {
                "status": "absent",
                # Absence is only an observation. The driver combines it with
                # its process-local submit lifecycle before acknowledging the
                # durable Stop, so a concurrent admission cannot be cancelled
                # and then start afterwards.
                "acknowledged": False,
                "active": False,
                "interrupted": False,
                "session_id": None,
            }

        session_id, active = match
        if not active:
            return {
                "status": "inactive",
                "acknowledged": True,
                "active": False,
                "interrupted": False,
                "session_id": session_id,
            }

        result = self._call(
            "session.interrupt",
            {
                "session_id": session_id,
                "expected_hosted_task_id": task.task_id,
                "expected_hosted_execution_generation": execution_generation,
            },
        )
        interrupted = bool(result.get("interrupted")) or (
            result.get("status") == "interrupted"
        )
        if not interrupted:
            observed = self._find_admitted_session(
                task=task,
                execution_generation=execution_generation,
                source=source,
            )
            if observed is None:
                return {
                    "status": "absent",
                    "acknowledged": False,
                    "active": False,
                    "interrupted": False,
                    "session_id": None,
                }
            observed_session_id, observed_active = observed
            if observed_active:
                # Another exact Stop can own the process-local interrupt claim.
                # A negative signal result is not proof that the turn stopped.
                return {
                    "status": "pending",
                    "acknowledged": False,
                    "active": True,
                    "interrupted": False,
                    "session_id": observed_session_id,
                }
            return {
                "status": "inactive",
                "acknowledged": True,
                "active": False,
                "interrupted": False,
                "session_id": observed_session_id,
            }
        return {
            **result,
            "status": "interrupted",
            "acknowledged": True,
            "active": True,
            "interrupted": True,
            "session_id": session_id,
        }

    def interrupt(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        expected_task_id: str,
    ) -> Mapping[str, Any] | None:
        self._require_profile_available(profile)
        del source
        return self._call(
            "session.interrupt",
            {
                "profile": profile,
                "session_id": session_id,
                "expected_hosted_task_id": expected_task_id,
            },
        )
