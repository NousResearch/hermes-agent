"""Scoped HTTP client for peer hosted-room member turns."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gateway.hosted_room_peer import (
    HostedMemberDispatch,
    validate_room_link_url,
)


logger = logging.getLogger(__name__)


_NOT_ADMITTED_ERRNOS = frozenset(
    value
    for name in (
        "ECONNREFUSED",
        "ENETDOWN",
        "ENETUNREACH",
        "EHOSTDOWN",
        "EHOSTUNREACH",
    )
    if (value := getattr(errno, name, None)) is not None
)
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# A replay page may legitimately contain many bounded 64 KiB room events. Keep
# enough room for the largest normal page while preventing peer-sized responses
# from scaling memory use without limit.
MAX_PEER_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PEER_ERROR_RESPONSE_BYTES = 16 * 1024
_PEER_RESPONSE_CHUNK_BYTES = 64 * 1024


class _PeerResponseTooLarge(ValueError):
    """A peer response exceeded its endpoint-specific byte budget."""


class _PeerResponseDeadlineExceeded(TimeoutError):
    """A peer response exceeded the request's monotonic wall-clock budget."""


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _set_response_socket_timeout(response: Any, remaining: float) -> None:
    """Best-effort urllib socket timeout tightened to the remaining budget."""

    frontier = [response]
    seen: set[int] = set()
    for _depth in range(5):
        next_frontier = []
        for value in frontier:
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            setter = getattr(value, "settimeout", None)
            if callable(setter):
                try:
                    setter(max(0.001, remaining))
                except (OSError, ValueError):
                    pass
                return
            next_frontier.extend(
                getattr(value, field, None) for field in ("fp", "raw", "_sock")
            )
        frontier = next_frontier


def _read_bounded_response(
    response: Any,
    *,
    max_bytes: int,
    deadline: float,
) -> bytes:
    declared = _content_length(response)
    if declared is not None and declared > max_bytes:
        raise _PeerResponseTooLarge

    reader = getattr(response, "read1", None)
    if not callable(reader):
        reader = response.read
    body = bytearray()
    while len(body) <= max_bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _PeerResponseDeadlineExceeded
        _set_response_socket_timeout(response, remaining)
        try:
            chunk = reader(
                min(_PEER_RESPONSE_CHUNK_BYTES, max_bytes + 1 - len(body))
            )
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise _PeerResponseDeadlineExceeded from exc
            raise
        if not chunk:
            return bytes(body)
        if not isinstance(chunk, (bytes, bytearray)):
            raise ValueError("peer returned a non-byte response")
        body.extend(chunk)
        if len(body) > max_bytes:
            raise _PeerResponseTooLarge
    raise _PeerResponseTooLarge


def _is_proven_pre_admission_failure(exc: BaseException) -> bool:
    """Return whether no HTTP connection could have carried the request."""
    reason: Any = exc
    while isinstance(reason, urllib.error.URLError):
        reason = reason.reason
    if isinstance(reason, socket.gaierror):
        return True
    return isinstance(reason, OSError) and reason.errno in _NOT_ADMITTED_ERRNOS


def _response_error_code(detail: str) -> str | None:
    """Extract a machine error code without returning response credentials."""
    try:
        payload = json.loads(detail)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        code = error["code"]
        if _ERROR_CODE_RE.fullmatch(code) is None:
            return None
        message = str(error.get("message") or "").lower()
        # Older target gateways wrap grant expiry inside the generic dispatch
        # error. Normalize it locally until their wire code becomes specific.
        if code == "invalid_room_dispatch" and "room grant" in message:
            return "invalid_room_grant"
        if code == "invalid_room_dispatch" and "capability catalog changed" in message:
            return "room_capability_catalog_changed"
        if code == "invalid_room_dispatch" and "execution policy changed" in message:
            return "room_execution_policy_changed"
        return code
    code = payload.get("code")
    return (
        code
        if isinstance(code, str) and _ERROR_CODE_RE.fullmatch(code) is not None
        else None
    )


class PeerRunsHTTPError(RuntimeError):
    """Controlled peer HTTP failure."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        ambiguous: bool = False,
        not_admitted: bool = False,
        status_code: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.ambiguous = ambiguous
        self.not_admitted = not_admitted
        self.status_code = status_code
        self.error_code = error_code
        self.error_message = error_message
        self.needs_reauthorization = bool(
            status_code in {401, 403}
            and error_code
            in {
                "invalid_room_grant",
                "room_capability_catalog_changed",
                "room_execution_policy_changed",
                "room_reauthorization_required",
            }
        )
        self.needs_capability_refresh = bool(
            status_code == 403
            and error_code == "room_capability_catalog_changed"
        )
        self.needs_execution_policy_refresh = bool(
            status_code == 403
            and error_code == "room_execution_policy_changed"
        )


class PeerRunsHTTPClient:
    """Drive a peer's dedicated group session via scoped async Runs APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 30,
        receipt_db_path: Path | str | None = None,
        poll_min_seconds: float = 0.1,
        poll_max_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        base_url, self.transport_security = validate_room_link_url(base_url)
        if api_key and len(api_key) < 16:
            raise ValueError("peer API key is missing or too short")
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.receipt_db_path = Path(receipt_db_path) if receipt_db_path else None
        if poll_min_seconds <= 0 or poll_max_seconds < poll_min_seconds:
            raise ValueError("peer polling bounds are invalid")
        self.poll_min_seconds = float(poll_min_seconds)
        self.poll_max_seconds = float(poll_max_seconds)
        self.clock = clock
        self._runs: dict[tuple[str, int], dict[str, Any]] = {}
        self._observation_key: tuple[str, int] | None = None
        self._status_cache: dict[str, dict[str, Any]] = {}
        self._recovery_backoff: dict[tuple[str, int], dict[str, Any]] = {}
        self._terminal_receipts: set[tuple[str, int]] = set()
        self._room_scope: dict[str, Any] | None = None

    def bind_receipt_store(self, db_path: Path | str) -> None:
        """Attach the gateway-wide durable receipt store idempotently."""
        path = Path(db_path)
        if self.receipt_db_path not in {None, path}:
            raise PeerRunsHTTPError("peer receipt store changed")
        self.receipt_db_path = path

    def bind_room_scope(
        self,
        *,
        room_id: str,
        home_install_id: str,
        authority_gateway_id: str,
        authority_epoch: int,
        member_id: str,
        target_install_id: str,
        target_profile: str,
    ) -> None:
        """Fence every in-memory and durable receipt to one room authority."""

        scope = {
            "room_id": str(room_id or ""),
            "home_install_id": str(home_install_id or ""),
            "authority_gateway_id": str(authority_gateway_id or ""),
            "authority_epoch": int(authority_epoch or 0),
            "member_id": str(member_id or ""),
            "target_install_id": str(target_install_id or ""),
            "target_profile": str(target_profile or ""),
        }
        if not all(value for key, value in scope.items() if key != "authority_epoch"):
            raise PeerRunsHTTPError("peer room receipt scope is incomplete")
        if scope["authority_epoch"] < 1:
            raise PeerRunsHTTPError("peer room receipt authority epoch is invalid")
        if self._room_scope == scope:
            return
        self._room_scope = scope
        self._runs.clear()
        self._observation_key = None
        self._status_cache.clear()
        self._recovery_backoff.clear()
        self._terminal_receipts.clear()

    def _bind_dispatch_scope(self, dispatch: HostedMemberDispatch) -> None:
        self.bind_room_scope(
            room_id=dispatch.room_id,
            home_install_id=dispatch.home_install_id,
            authority_gateway_id=dispatch.authority_gateway_id,
            authority_epoch=dispatch.authority_epoch,
            member_id=dispatch.member_id,
            target_install_id=dispatch.target_install_id,
            target_profile=dispatch.target_profile,
        )

    def _receipt_identity(
        self,
        *,
        task_id: str,
        execution_generation: int,
    ) -> dict[str, Any] | None:
        if self._room_scope is None:
            return None
        return {
            **self._room_scope,
            "task_id": task_id,
            "execution_generation": execution_generation,
        }

    def bind_observation(self, *, task_id: str, execution_generation: int) -> None:
        """Pin history/status reads to one exact logical task attempt."""
        key = (str(task_id or ""), int(execution_generation or 0))
        if not key[0] or key[1] < 1:
            raise PeerRunsHTTPError("peer observation identity is invalid")
        if self._observation_key != key:
            for terminal_key in self._terminal_receipts - {key}:
                self._runs.pop(terminal_key, None)
            self._terminal_receipts.intersection_update({key})
            self._observation_key = key
            self._status_cache.clear()
            self._recovery_backoff.clear()

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        room_grant: str | None = None,
    ) -> dict[str, Any]:
        from hermes_cli.urllib_security import open_credentialed_url

        deadline = time.monotonic() + self.timeout_seconds
        ambiguous = method == "POST"
        request_headers = {
            "Authorization": (
                f"HermesRoom {room_grant}" if room_grant else f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
            "User-Agent": "Hermes-RoomLink/1.0",
        }
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=(
                json.dumps(body, separators=(",", ":")).encode("utf-8")
                if body is not None
                else None
            ),
            method=method,
            headers=request_headers,
        )
        try:
            with open_credentialed_url(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = _read_bounded_response(
                    response,
                    max_bytes=MAX_PEER_RESPONSE_BYTES,
                    deadline=deadline,
                ).decode("utf-8", "replace")
        except _PeerResponseTooLarge as exc:
            raise PeerRunsHTTPError(
                "peer response exceeded the RoomLink size limit",
                ambiguous=ambiguous,
            ) from exc
        except _PeerResponseDeadlineExceeded as exc:
            raise PeerRunsHTTPError(
                "peer response exceeded the RoomLink time budget",
                retryable=True,
                ambiguous=ambiguous,
            ) from exc
        except urllib.error.HTTPError as exc:
            pre_admission = bool(
                method == "POST"
                and path == "/v1/runs"
                and 400 <= exc.code < 500
            )
            try:
                detail = _read_bounded_response(
                    exc,
                    max_bytes=MAX_PEER_ERROR_RESPONSE_BYTES,
                    deadline=deadline,
                ).decode("utf-8", "replace")[:500]
            except _PeerResponseTooLarge as body_exc:
                raise PeerRunsHTTPError(
                    "peer error response exceeded the RoomLink size limit",
                    ambiguous=method == "POST" and exc.code >= 500,
                    not_admitted=pre_admission,
                    status_code=exc.code,
                ) from body_exc
            except _PeerResponseDeadlineExceeded as body_exc:
                raise PeerRunsHTTPError(
                    "peer error response exceeded the RoomLink time budget",
                    retryable=True,
                    ambiguous=method == "POST" and exc.code >= 500,
                    not_admitted=pre_admission,
                    status_code=exc.code,
                ) from body_exc
            except Exception:
                detail = ""
            error_code = _response_error_code(detail)
            logger.debug(
                "Peer RoomLink request returned HTTP %s (%s)",
                exc.code,
                error_code or "no-code",
            )
            message = (
                "peer room authorization needs renewal"
                if exc.code in {401, 403}
                and error_code in {"invalid_room_grant", "room_reauthorization_required"}
                else "peer room execution policy needs reauthorization"
                if exc.code == 403 and error_code == "room_execution_policy_changed"
                else "peer room capabilities need reauthorization"
                if exc.code == 403 and error_code == "room_capability_catalog_changed"
                else f"peer rejected {method} {path} with HTTP {exc.code}"
            )
            raise PeerRunsHTTPError(
                message,
                retryable=exc.code in {408, 425, 429} or exc.code >= 500,
                ambiguous=method == "POST" and exc.code >= 500,
                not_admitted=pre_admission,
                status_code=exc.code,
                error_code=error_code,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            not_admitted = method == "POST" and _is_proven_pre_admission_failure(
                exc
            )
            raise PeerRunsHTTPError(
                "peer RoomLink endpoint is unreachable",
                retryable=True,
                ambiguous=ambiguous and not not_admitted,
                not_admitted=not_admitted,
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PeerRunsHTTPError("peer returned non-JSON data") from exc
        if not isinstance(payload, dict):
            raise PeerRunsHTTPError("peer returned a non-object response")
        return payload

    def prepare(
        self,
        *,
        room_id: str,
        profile: str,
        source: str,
        grant: str,
        create: bool,
        expected_session_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        if source != "bot_room":
            raise PeerRunsHTTPError("peer room source must be bot_room")
        self._require_room_grant(grant)
        logical_session = (
            "roomlink_"
            + hashlib.sha256(f"{room_id}\0{profile}".encode("utf-8")).hexdigest()[
                :32
            ]
        )
        if expected_session_id and expected_session_id != logical_session:
            raise PeerRunsHTTPError("peer room session identity changed")
        return {
            "session_id": logical_session,
            "title": f"Group: {room_id}",
            "source": source,
        }

    def dispatch(
        self,
        *,
        dispatch: Mapping[str, Any],
        grant: str,
    ) -> Mapping[str, Any]:
        checked = HostedMemberDispatch.from_mapping(dispatch)
        self._require_room_grant(grant)
        self._bind_dispatch_scope(checked)
        self.bind_observation(
            task_id=checked.task_id,
            execution_generation=checked.execution_generation,
        )
        return self._admit_dispatch(checked, grant=grant)

    def recover_dispatch(
        self,
        *,
        dispatch: Mapping[str, Any],
        grant: str,
    ) -> Mapping[str, Any]:
        """Recover one exact admission by receipt or idempotent POST replay."""
        checked = HostedMemberDispatch.from_mapping(dispatch)
        self._require_room_grant(grant)
        self._bind_dispatch_scope(checked)
        self.bind_observation(
            task_id=checked.task_id,
            execution_generation=checked.execution_generation,
        )
        existing = self._receipt_for_dispatch(checked)
        if existing is not None:
            expected = (
                checked.room_id,
                checked.home_install_id,
                checked.authority_gateway_id,
                checked.authority_epoch,
                checked.member_id,
                checked.target_install_id,
                checked.target_profile,
            )
            stored = (
                existing["room_id"],
                existing["home_install_id"],
                existing["authority_gateway_id"],
                existing["authority_epoch"],
                existing["member_id"],
                existing["target_install_id"],
                existing["target_profile"],
            )
            if stored != expected:
                raise PeerRunsHTTPError(
                    "peer run receipt conflicts with the recovered dispatch"
                )
            return {
                "status": "accepted",
                "task_id": checked.task_id,
                "execution_generation": checked.execution_generation,
                "run_id": str(existing["run_id"]),
                "session_id": str(existing["session_id"]),
                "replayed": True,
            }
        key = (checked.task_id, checked.execution_generation)
        now = self.clock()
        backoff = self._recovery_backoff.get(key)
        if backoff is not None and now < float(backoff["next_attempt_at"]):
            raise PeerRunsHTTPError(
                "peer admission recovery is backing off",
                retryable=True,
                ambiguous=True,
            )
        try:
            recovered = self._admit_dispatch(checked, grant=grant)
        except PeerRunsHTTPError as exc:
            if exc.retryable or exc.ambiguous:
                delay = self._next_poll_delay(backoff)
                self._recovery_backoff = {
                    key: {
                        "delay": delay,
                        "next_attempt_at": now + delay,
                    }
                }
            raise
        self._recovery_backoff.pop(key, None)
        return recovered

    def _admit_dispatch(
        self,
        checked: HostedMemberDispatch,
        *,
        grant: str,
    ) -> Mapping[str, Any]:
        session_id = self._session_id(checked, grant=grant)
        idempotency_key = f"room:{checked.task_id}:{checked.execution_generation}"

        def admit(dispatch: HostedMemberDispatch) -> dict[str, Any]:
            return self._request(
                "/v1/runs",
                method="POST",
                body={
                    "input": dispatch.prompt,
                    "hosted_room_dispatch": dispatch.as_mapping(),
                },
                headers={"Idempotency-Key": idempotency_key},
                room_grant=grant,
            )

        try:
            result = admit(checked)
        except PeerRunsHTTPError as exc:
            if exc.ambiguous:
                result = admit(checked)
            else:
                raise
        run_id = str(result.get("run_id") or "")
        if not run_id:
            raise PeerRunsHTTPError("peer did not return a run id")
        receipt = {
            "run_id": run_id,
            "session_id": session_id,
            "room_id": checked.room_id,
            "home_install_id": checked.home_install_id,
            "authority_gateway_id": checked.authority_gateway_id,
            "authority_epoch": checked.authority_epoch,
            "member_id": checked.member_id,
            "task_id": checked.task_id,
            "execution_generation": checked.execution_generation,
            "target_install_id": checked.target_install_id,
            "target_profile": checked.target_profile,
        }
        if self.receipt_db_path is not None:
            from gateway import hosted_rooms

            hosted_rooms.upsert_remote_run_receipt(
                self.receipt_db_path,
                record=receipt,
            )
        self._runs[(checked.task_id, checked.execution_generation)] = receipt
        self._status_cache.pop(run_id, None)
        return {
            "status": "accepted",
            "task_id": checked.task_id,
            "execution_generation": checked.execution_generation,
            "run_id": run_id,
            "session_id": session_id,
            "replayed": bool(result.get("replayed", False)),
        }

    def _session_id(self, dispatch: HostedMemberDispatch, *, grant: str) -> str:
        existing = self._receipt_for_dispatch(dispatch)
        if existing:
            return str(existing["session_id"])
        prepared = self.prepare(
            room_id=dispatch.room_id,
            profile=dispatch.target_profile,
            source="bot_room",
            grant=grant,
            create=True,
        )
        if prepared is None:
            raise PeerRunsHTTPError("peer room session is unavailable")
        return str(prepared.get("session_id") or prepared.get("id") or "")

    def _observation_receipt(
        self, *, room_id: str, profile: str, session_id: str
    ) -> dict[str, Any] | None:
        record = None
        if self._observation_key is not None:
            task_id, execution_generation = self._observation_key
            record = self._runs.get(self._observation_key)
            if record is None and self.receipt_db_path is not None:
                from gateway import hosted_rooms

                identity = self._receipt_identity(
                    task_id=task_id,
                    execution_generation=execution_generation,
                )
                if identity is not None:
                    record = hosted_rooms.remote_run_receipt(
                        self.receipt_db_path,
                        record=identity,
                    )
        if record is None:
            return None
        if (
            record["room_id"] != room_id
            or record["target_profile"] != profile
            or record["session_id"] != session_id
        ):
            raise PeerRunsHTTPError("peer observation receipt changed scope")
        return record

    @staticmethod
    def _compact_run_status(status: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: status[key]
            for key in (
                "run_id",
                "status",
                "output",
                "error",
                "approval",
                "last_event",
            )
            if key in status
        }

    def _next_poll_delay(self, cached: Mapping[str, Any] | None) -> float:
        previous = (
            float(cached["delay"])
            if cached is not None
            else self.poll_min_seconds / 2
        )
        return min(
            self.poll_max_seconds,
            max(self.poll_min_seconds, previous * 2),
        )

    @staticmethod
    def _run_is_terminal(status: Mapping[str, Any]) -> bool:
        return status.get("status") in {
            "completed",
            "failed",
            "interrupted",
            "cancelled",
        }

    def _poll_receipt(
        self,
        record: Mapping[str, Any],
        *,
        grant: str,
    ) -> dict[str, Any]:
        run_id = str(record["run_id"])
        now = self.clock()
        cached = self._status_cache.get(run_id)
        if cached is not None:
            status = cached["status"]
            if self._run_is_terminal(status):
                return status
            if now < float(cached["next_poll_at"]):
                error = cached.get("error")
                if isinstance(error, PeerRunsHTTPError):
                    raise error
                return status
        try:
            status = self._compact_run_status(
                self._request(
                    f"/v1/runs/{urllib.parse.quote(run_id, safe='')}",
                    room_grant=self._require_room_grant(grant),
                )
            )
            if (
                str(status.get("run_id") or "") != run_id
                or status.get("status")
                not in {
                    "queued",
                    "running",
                    "waiting_for_approval",
                    "stopping",
                    "completed",
                    "failed",
                    "interrupted",
                    "cancelled",
                }
            ):
                raise PeerRunsHTTPError("peer returned a mismatched run status")
        except PeerRunsHTTPError as exc:
            delay = self._next_poll_delay(cached)
            self._status_cache = {
                run_id: {
                    "status": cached["status"] if cached is not None else {},
                    "error": exc,
                    "delay": delay,
                    "next_poll_at": now + delay,
                }
            }
            raise
        delay = self._next_poll_delay(cached)
        self._status_cache = {
            run_id: {
                "status": status,
                "delay": delay,
                "next_poll_at": now + delay,
            }
        }
        if self._run_is_terminal(status):
            self._terminal_receipts.add(
                (str(record["task_id"]), int(record["execution_generation"]))
            )
        return status

    def _receipt_for_dispatch(
        self, dispatch: HostedMemberDispatch
    ) -> dict[str, Any] | None:
        key = (dispatch.task_id, dispatch.execution_generation)
        record = self._runs.get(key)
        if record is not None or self.receipt_db_path is None:
            return record
        from gateway import hosted_rooms

        identity = self._receipt_identity(
            task_id=dispatch.task_id,
            execution_generation=dispatch.execution_generation,
        )
        if identity is None:
            return None
        return hosted_rooms.remote_run_receipt(
            self.receipt_db_path,
            record=identity,
        )

    def history(
        self,
        *,
        room_id: str,
        profile: str,
        session_id: str,
        grant: str,
    ) -> Sequence[Mapping[str, Any]]:
        receipt = self._observation_receipt(
            room_id=room_id,
            profile=profile,
            session_id=session_id,
        )
        if receipt is None:
            return []
        status = self._poll_receipt(receipt, grant=grant)
        state = str(status.get("status") or "")
        if state not in {"completed", "failed", "interrupted"}:
            return []
        return [
            {
                "role": "assistant",
                "task_id": receipt["task_id"],
                "execution_generation": receipt["execution_generation"],
                "status": "settled" if state == "completed" else "failed",
                "message_id": f"peer-run:{status.get('run_id')}",
                "content": status.get("output") or status.get("error") or "",
            }
        ]

    def status(
        self,
        *,
        room_id: str,
        profile: str,
        session_id: str,
        grant: str,
    ) -> Mapping[str, Any]:
        receipt = self._observation_receipt(
            room_id=room_id,
            profile=profile,
            session_id=session_id,
        )
        if receipt is None:
            return {"active": False, "task_id": None}
        status = self._poll_receipt(receipt, grant=grant)
        active_states = {"queued", "running", "waiting_for_approval", "stopping"}
        return {
            "active": status.get("status") in active_states,
            "task_id": receipt["task_id"],
            "execution_generation": receipt["execution_generation"],
            "status": status.get("status"),
            "run_id": status.get("run_id"),
            "approval": status.get("approval"),
        }

    def approve_receipt(
        self,
        *,
        task_id: str,
        execution_generation: int,
        request_id: str,
        choice: str,
        grant: str,
    ) -> Mapping[str, Any] | None:
        """Resolve approval for the exact durable remote run."""
        record = self._runs.get((task_id, execution_generation))
        if record is None and self.receipt_db_path is not None:
            from gateway import hosted_rooms

            identity = self._receipt_identity(
                task_id=task_id,
                execution_generation=execution_generation,
            )
            if identity is not None:
                record = hosted_rooms.remote_run_receipt(
                    self.receipt_db_path,
                    record=identity,
                )
        if record is None:
            return None
        request_id = str(request_id or "").strip()
        if not request_id:
            raise PeerRunsHTTPError("an exact approval request_id is required")
        self._require_room_grant(grant)
        result = self._request(
            f"/v1/runs/{urllib.parse.quote(str(record['run_id']), safe='')}/approval",
            method="POST",
            body={"choice": choice, "request_id": request_id},
            room_grant=grant,
        )
        self._status_cache.pop(str(record["run_id"]), None)
        return result

    def stop(
        self,
        *,
        dispatch: Mapping[str, Any],
        grant: str,
    ) -> Mapping[str, Any] | None:
        checked = HostedMemberDispatch.from_mapping(dispatch)
        self._bind_dispatch_scope(checked)
        return self.stop_receipt(
            task_id=checked.task_id,
            execution_generation=checked.execution_generation,
            grant=grant,
        )

    def stop_receipt(
        self,
        *,
        task_id: str,
        execution_generation: int,
        grant: str,
    ) -> Mapping[str, Any] | None:
        """Stop the exact durable remote run after a home restart."""
        record = self._runs.get((task_id, execution_generation))
        if record is None and self.receipt_db_path is not None:
            from gateway import hosted_rooms

            identity = self._receipt_identity(
                task_id=task_id,
                execution_generation=execution_generation,
            )
            if identity is not None:
                record = hosted_rooms.remote_run_receipt(
                    self.receipt_db_path,
                    record=identity,
                )
        if record is None:
            return None
        result = self._request(
            f"/v1/runs/{urllib.parse.quote(str(record['run_id']), safe='')}/stop",
            method="POST",
            body={},
            room_grant=self._require_room_grant(grant),
        )
        self._status_cache.pop(str(record["run_id"]), None)
        if self._run_is_terminal(result):
            self._terminal_receipts.add((str(task_id), int(execution_generation)))
        return result

    def issue_invitation(
        self,
        *,
        room_id: str,
        home_install_id: str,
        authority_gateway_id: str,
        authority_epoch: int,
        member_id: str,
        grant_id: str,
        ttl_seconds: float = 3600,
        status_ttl_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        """Ask the target gateway to mint a scoped room-member grant."""
        if not self.api_key:
            raise PeerRunsHTTPError(
                "issuing an invitation requires the target gateway API key"
            )
        return self._request(
            "/v1/room-members/invitations",
            method="POST",
            body={
                "room_id": room_id,
                "home_install_id": home_install_id,
                "authority_gateway_id": authority_gateway_id,
                "authority_epoch": authority_epoch,
                "member_id": member_id,
                "grant_id": grant_id,
                "ttl_seconds": ttl_seconds,
                **(
                    {"status_ttl_seconds": status_ttl_seconds}
                    if status_ttl_seconds is not None
                    else {}
                ),
            },
        )

    def refresh_grant(
        self,
        *,
        grant: str,
        ttl_seconds: float = 24 * 60 * 60,
        capability_digest: str | None = None,
        execution_policy_digest: str | None = None,
    ) -> Mapping[str, Any]:
        """Renew dispatch access only while its frozen authority is unchanged."""
        self._require_room_grant(grant)
        refreshed = self._request(
            "/v1/room-members/grants/refresh",
            method="POST",
            body={"ttl_seconds": ttl_seconds},
            room_grant=grant,
        )
        replacement = str(refreshed.get("grant") or "")
        if not replacement:
            raise PeerRunsHTTPError("peer returned no refreshed room grant")
        # Persist only after the target proves the replacement can authorize
        # the same scoped capability endpoint.
        probe = self.probe(grant=replacement)
        from gateway.hosted_room_peer import GatewayRoomCatalog

        catalog = GatewayRoomCatalog.from_mapping(probe.get("catalog"))
        if (
            execution_policy_digest is not None
            and catalog.execution_policy.policy_digest
            != execution_policy_digest
        ):
            raise PeerRunsHTTPError(
                "peer room execution policy needs reauthorization",
                status_code=403,
                error_code="room_execution_policy_changed",
                not_admitted=True,
            )
        if (
            capability_digest is not None
            and catalog.catalog_digest != capability_digest
        ):
            raise PeerRunsHTTPError(
                "peer room capabilities need reauthorization",
                status_code=403,
                error_code="room_capability_catalog_changed",
                not_admitted=True,
            )
        return {**refreshed, "catalog": probe.get("catalog")}

    def revoke_grant(self, *, grant: str) -> Mapping[str, Any]:
        """Revoke this grant's exact room/home/target/profile scope."""
        self._require_room_grant(grant)
        return self._request(
            "/v1/room-members/grants/revoke",
            method="POST",
            body={},
            room_grant=grant,
        )

    def probe(self, *, grant: str) -> Mapping[str, Any]:
        """Verify gateway reachability and the live scoped capability catalog."""
        self._require_room_grant(grant)
        return self._request(
            "/v1/room-members/capabilities",
            room_grant=grant,
        )

    @staticmethod
    def _require_room_grant(grant: str) -> str:
        """Prevent scoped operations from falling back to broad Bearer auth."""
        value = str(grant or "")
        if not value or value in {"compat", "compatibility-only"}:
            raise PeerRunsHTTPError("a scoped room grant is required")
        return value
