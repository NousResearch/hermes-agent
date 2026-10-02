"""Scoped HTTP client for peer hosted-room member turns."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any, NoReturn

from gateway.hosted_room_peer import (
    GatewayRoomCatalog, HostedMemberDispatch, validate_room_link_url)


logger = logging.getLogger(__name__)


_NOT_ADMITTED_ERRNOS = frozenset(
    value for name in ("ECONNREFUSED", "ENETDOWN", "ENETUNREACH", "EHOSTDOWN", "EHOSTUNREACH")
    if (value := getattr(errno, name, None)) is not None)
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# A replay page may legitimately hold many bounded 64 KiB room events; cap it so a peer-sized
# response cannot scale memory use without limit.
MAX_PEER_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PEER_ERROR_RESPONSE_BYTES = 16 * 1024
_PEER_RESPONSE_CHUNK_BYTES = 64 * 1024

# Receipt fields that must match between a stored receipt and a dispatch.
_RECEIPT_SCOPE_FIELDS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
    "member_id", "target_install_id", "target_profile")
_TERMINAL_RUN_STATES = frozenset({"completed", "failed", "interrupted", "cancelled"})
_ACTIVE_RUN_STATES = frozenset({"queued", "running", "waiting_for_approval", "stopping"})
_KNOWN_RUN_STATES = _TERMINAL_RUN_STATES | _ACTIVE_RUN_STATES
_RUN_STATUS_KEYS = ("run_id", "status", "output", "error", "approval", "last_event",
                    "pending_controls", "execution_generation", "artifacts", "artifact_scope", "peer_output_empty", "peer_output_unresolved")
# Older target gateways wrap these inside the generic dispatch error; normalize locally.
_LEGACY_DISPATCH_MESSAGE_CODES = (
    ("room grant", "invalid_room_grant"),
    ("capability catalog changed", "room_capability_catalog_changed"),
    ("execution policy changed", "room_execution_policy_changed"))
_GRANT_RENEWAL_CODES = frozenset({"invalid_room_grant", "room_reauthorization_required"})
# (error_code, human message) for the two digest checks after a grant refresh.
_EXECUTION_POLICY_CHANGED = (
    "room_execution_policy_changed", "peer room execution policy needs reauthorization")
_CAPABILITY_CHANGED = (
    "room_capability_catalog_changed", "peer room capabilities need reauthorization")
_REAUTHORIZATION_CODES = _GRANT_RENEWAL_CODES | {
    _EXECUTION_POLICY_CHANGED[0], _CAPABILITY_CHANGED[0]}
# Human messages for 401/403 reauthorization codes (any other status keeps the generic text).
_REAUTHORIZATION_MESSAGES = {
    **dict.fromkeys(_GRANT_RENEWAL_CODES, "peer room authorization needs renewal"),
    _EXECUTION_POLICY_CHANGED[0]: _EXECUTION_POLICY_CHANGED[1],
    _CAPABILITY_CHANGED[0]: _CAPABILITY_CHANGED[1]}
_BUDGET_MESSAGES = {
    "size": "peer{kind} response exceeded the RoomLink size limit",
    "time": "peer{kind} response exceeded the RoomLink time budget"}
# A grant the member's gateway refused is not sent to the same check again for a while.
_AUTH_PROBE_COOLDOWN_SECONDS = 60.0
_MAX_AUTH_PROBE_REJECTIONS = 64
_AUTH_PROBE_ENDPOINTS = frozenset({
    ("GET", "/v1/room-members/capabilities"), ("POST", "/v1/room-members/grants/refresh")})


class _PeerResponseTooLarge(ValueError):
    """A peer response exceeded its endpoint-specific byte budget."""


class _PeerResponseDeadlineExceeded(TimeoutError):
    """A peer response exceeded the request's monotonic wall-clock budget."""


def _set_response_socket_timeout(response: Any, remaining: float) -> None:
    """Best-effort urllib socket timeout tightened to the remaining budget."""
    frontier, seen = [response], set()
    for _depth in range(5):
        next_frontier = []
        for value in frontier:
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            setter = getattr(value, "settimeout", None)
            if callable(setter):
                with suppress(OSError, ValueError):
                    setter(max(0.001, remaining))
                return
            next_frontier.extend(getattr(value, field, None) for field in ("fp", "raw", "_sock"))
        frontier = next_frontier


def _read_bounded_response(response: Any, *, max_bytes: int, deadline: float) -> bytes:
    try:
        declared = int(response.headers.get("Content-Length"))
    except (AttributeError, TypeError, ValueError):
        declared = -1
    if declared > max_bytes:
        raise _PeerResponseTooLarge
    reader = getattr(response, "read1", None)
    reader = reader if callable(reader) else response.read
    body = bytearray()
    while len(body) <= max_bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _PeerResponseDeadlineExceeded
        _set_response_socket_timeout(response, remaining)
        try:
            chunk = reader(min(_PEER_RESPONSE_CHUNK_BYTES, max_bytes + 1 - len(body)))
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


def _read_body(response: Any, *, max_bytes: int, deadline: float, kind: str, as_bytes: bool = False,
               **flags: Any) -> str | bytes:
    """Read a bounded body as text; budget overruns become classified ``PeerRunsHTTPError``."""
    try:
        raw = _read_bounded_response(response, max_bytes=max_bytes, deadline=deadline)
        return raw if as_bytes else raw.decode("utf-8", "replace")
    except _PeerResponseTooLarge as exc:
        raise PeerRunsHTTPError(_BUDGET_MESSAGES["size"].format(kind=kind), **flags) from exc
    except _PeerResponseDeadlineExceeded as exc:
        raise PeerRunsHTTPError(
            _BUDGET_MESSAGES["time"].format(kind=kind), retryable=True, **flags) from exc


def _is_proven_pre_admission_failure(exc: BaseException) -> bool:
    """Return whether no HTTP connection could have carried the request."""
    reason: Any = exc
    while isinstance(reason, urllib.error.URLError):
        reason = reason.reason
    return isinstance(reason, socket.gaierror) or (
        isinstance(reason, OSError) and reason.errno in _NOT_ADMITTED_ERRNOS)


def _valid_code(code: Any) -> str | None:
    return code if isinstance(code, str) and _ERROR_CODE_RE.fullmatch(code) else None


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
        code = _valid_code(error["code"])
        if code == "invalid_room_dispatch":
            message = str(error.get("message") or "").lower()
            for needle, normalized in _LEGACY_DISPATCH_MESSAGE_CODES:
                if needle in message:
                    return normalized
        return code
    return _valid_code(payload.get("code"))


class PeerRunsHTTPError(RuntimeError):
    """Controlled peer HTTP failure."""

    def __init__(
        self, message: str, *, retryable: bool = False, ambiguous: bool = False,
        not_admitted: bool = False, status_code: int | None = None, error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable, self.ambiguous, self.not_admitted = retryable, ambiguous, not_admitted
        self.status_code, self.error_code = status_code, error_code
        self.needs_reauthorization = (
            status_code in {401, 403} and error_code in _REAUTHORIZATION_CODES)


_ROOM_GRANT_REQUEST_BUDGET: ContextVar[tuple[float, float, Callable[[], float]] | None] = ContextVar(
    "room_grant_request_budget", default=None)


def room_grant_request_budget_remaining() -> float | None:
    """Seconds left in the current renewal budget, or None outside one."""
    budget = _ROOM_GRANT_REQUEST_BUDGET.get()
    if budget is None:
        return None
    wall_end, clock_end, clock = budget
    return min(wall_end - time.monotonic(), clock_end - clock())


@contextmanager
def room_grant_request_budget(seconds: float, *, clock: Callable[[], float] = time.monotonic):
    """Bound every request one renewal makes, cleanup included, by one shared deadline.

    Context-local: a concurrent foreground request keeps its own timeout.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("room grant request budget must be finite and positive")
    outer = room_grant_request_budget_remaining()
    seconds = min(seconds, outer) if outer is not None else seconds
    token = _ROOM_GRANT_REQUEST_BUDGET.set((time.monotonic() + seconds, clock() + seconds, clock))
    try:
        yield
    finally:
        _ROOM_GRANT_REQUEST_BUDGET.reset(token)


def digest_reauthorization_error(
    catalog: GatewayRoomCatalog, *, capability_digest: str | None,
    execution_policy_digest: str | None) -> PeerRunsHTTPError | None:
    """Return the fail-closed error when a refreshed catalog drifted from the frozen digests.

    Execution policy is checked before capabilities; a ``None`` digest skips its check.
    """
    for expected, actual, (code, message) in (
        (
            execution_policy_digest, catalog.execution_policy.policy_digest,
            _EXECUTION_POLICY_CHANGED),
        (capability_digest, catalog.catalog_digest, _CAPABILITY_CHANGED)):
        if expected is not None and actual != expected:
            return PeerRunsHTTPError(message, status_code=403, error_code=code, not_admitted=True)
    return None


def _run_path(record: Mapping[str, Any], *suffix: str) -> str:
    return "/".join(("/v1/runs", urllib.parse.quote(str(record["run_id"]), safe=""), *suffix))


class PeerRunsHTTPClient:
    """Drive a peer's dedicated group session via scoped async Runs APIs."""

    @property
    def timeout_seconds(self) -> float:
        remaining = room_grant_request_budget_remaining()
        if remaining is None:
            return self._timeout_seconds
        if remaining <= 0:
            raise PeerRunsHTTPError("peer renewal request budget exhausted", retryable=True)
        return min(self._timeout_seconds, 1.0, remaining)

    @timeout_seconds.setter
    def timeout_seconds(self, value: float) -> None:
        self._timeout_seconds = float(value)

    def __init__(
        self, *, base_url: str, api_key: str, timeout_seconds: float = 30,
        receipt_db_path: Path | str | None = None, proof_install_id: str | None = None, poll_min_seconds: float = 0.1,
        poll_max_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
        base_url, self.transport_security = validate_room_link_url(base_url)
        if api_key and len(api_key) < 16:
            raise ValueError("peer API key is missing or too short")
        self.base_url, self.api_key, self.clock = base_url, api_key, clock
        self.proof_install_id = proof_install_id
        self.timeout_seconds = float(timeout_seconds)
        self.receipt_db_path = Path(receipt_db_path) if receipt_db_path else None
        if poll_min_seconds <= 0 or poll_max_seconds < poll_min_seconds:
            raise ValueError("peer polling bounds are invalid")
        self.poll_min_seconds = float(poll_min_seconds)
        self.poll_max_seconds = float(poll_max_seconds)
        self._runs: dict[tuple[str, int], dict[str, Any]] = {}
        self._observation_key: tuple[str, int] | None = None
        self._status_cache: dict[str, dict[str, Any]] = {}
        self._recovery_backoff: dict[tuple[str, int], dict[str, Any]] = {}
        self._terminal_receipts: set[tuple[str, int]] = set()
        self._room_scope: dict[str, Any] | None = None
        self._auth_probe_lock = threading.Lock()
        self._auth_probe_rejections: OrderedDict[tuple[str, str, str], tuple[float, str, int, str | None]] = (
            OrderedDict())

    def bind_receipt_store(self, db_path: Path | str) -> None:
        """Attach the gateway-wide durable receipt store idempotently."""
        path = Path(db_path)
        if self.receipt_db_path not in {None, path}:
            raise PeerRunsHTTPError("peer receipt store changed")
        self.receipt_db_path = path

    def bind_room_scope(
        self, *, room_id: str, home_install_id: str, authority_gateway_id: str,
        authority_epoch: int, member_id: str, target_install_id: str, target_profile: str) -> None:
        """Fence every in-memory and durable receipt to one room authority."""
        epoch = int(authority_epoch or 0)
        names = [str(v or "") for v in (
            room_id, home_install_id, authority_gateway_id, member_id, target_install_id,
            target_profile)]
        if not all(names):
            raise PeerRunsHTTPError("peer room receipt scope is incomplete")
        if epoch < 1:
            raise PeerRunsHTTPError("peer room receipt authority epoch is invalid")
        scope = dict(zip(_RECEIPT_SCOPE_FIELDS, names[:3] + [epoch] + names[3:]))
        if self._room_scope == scope:
            return
        self._room_scope, self._observation_key = scope, None
        for table in (
                self._runs, self._status_cache, self._recovery_backoff, self._terminal_receipts):
            table.clear()

    def _receipt(self, task_id: str, execution_generation: int) -> dict[str, Any] | None:
        """Return the in-memory receipt, falling back to the durable store."""
        record = self._runs.get((task_id, execution_generation))
        if record is not None or self.receipt_db_path is None or self._room_scope is None:
            return record
        from gateway import hosted_rooms
        identity = {"task_id": task_id, "execution_generation": execution_generation}
        stored = hosted_rooms.remote_run_receipt(
            self.receipt_db_path, record={**self._room_scope, **identity})
        if stored is not None and any(not isinstance(stored.get(key), str) or not stored[key]
                                      for key in ('run_id', 'session_id')):
            raise PeerRunsHTTPError('stored peer receipt is unreadable; admission remains unknown', ambiguous=True)
        return stored

    def bind_observation(self, *, task_id: str, execution_generation: int) -> None:
        """Pin history/status reads to one exact logical task attempt."""
        key = (str(task_id or ""), int(execution_generation or 0))
        if not key[0] or key[1] < 1:
            raise PeerRunsHTTPError("peer observation identity is invalid")
        if self._observation_key == key:
            return
        for terminal_key in self._terminal_receipts - {key}:
            self._runs.pop(terminal_key, None)
        self._terminal_receipts.intersection_update({key})
        self._observation_key = key
        self._status_cache.clear()
        self._recovery_backoff.clear()

    def _request(
        self, path: str, *, method: str = "GET", body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None, room_grant: str | None = None,
        ensure_ascii: bool = True) -> dict[str, Any]:
        # A renewal never follows a redirect: it would restart the request outside its budget.
        reject_redirects = self.proof_install_id is not None or _ROOM_GRANT_REQUEST_BUDGET.get() is not None
        timeout = self.timeout_seconds
        deadline, ambiguous = time.monotonic() + timeout, method == "POST"
        body_bytes = None if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=ensure_ascii).encode("utf-8")
        proof_state = None
        if self.proof_install_id is not None and room_grant:
            from gateway.hosted_room_proof import request_proof
            authorization, proof_key, proof_mac, encrypted_body = request_proof(
                room_grant, installation_id=self.proof_install_id, method=method, path=path, body=body_bytes or b'',
                headers={"Content-Type": "application/json", **(headers or {})})
            body_bytes = encrypted_body if body_bytes is not None else None
            proof_state = proof_key, proof_mac
            headers = {**(headers or {}), 'Authorization': authorization}
        request = urllib.request.Request(
            f"{self.base_url}{path}", method=method,
            data=body_bytes,
            headers={
                "Authorization": (
                    f"HermesRoom {room_grant}" if room_grant else f"Bearer {self.api_key}"),
                "Content-Type": "application/json", "User-Agent": "Hermes-RoomLink/1.0",
                **(headers or {})})
        probe_key = None
        if room_grant and (method, path) in _AUTH_PROBE_ENDPOINTS:
            probe_key = (method, path, hashlib.sha256(room_grant.encode()).hexdigest())
            self._check_auth_probe_cooldown(probe_key)
        try:
            with _open_roomlink_url(request, timeout=timeout, reject_redirects=reject_redirects) as response:
                raw = _read_body(
                    response, max_bytes=MAX_PEER_RESPONSE_BYTES + (16 if proof_state else 0), deadline=deadline, kind="",
                    ambiguous=ambiguous, as_bytes=proof_state is not None)
                raw = self._verify_proof_response(proof_state, response, raw, ambiguous=ambiguous)
                if proof_state and len(raw) > MAX_PEER_RESPONSE_BYTES:
                    raise PeerRunsHTTPError('peer response exceeded plaintext limit', ambiguous=ambiguous)
        except urllib.error.HTTPError as exc:
            try:
                if reject_redirects and exc.code in {301, 302, 303, 307, 308}:
                    raise PeerRunsHTTPError(
                        "peer renewal request refused an HTTP redirect", status_code=exc.code) from exc
                if proof_state is not None:
                    import io
                    raw_error = _read_body(exc, max_bytes=MAX_PEER_ERROR_RESPONSE_BYTES + 16,
                                           deadline=deadline, kind=" error", ambiguous=ambiguous, as_bytes=True)
                    raw_error = self._verify_proof_response(proof_state, exc, raw_error, ambiguous=ambiguous)
                    if len(raw_error) > MAX_PEER_ERROR_RESPONSE_BYTES:
                        raise PeerRunsHTTPError('peer error exceeded plaintext limit', ambiguous=ambiguous)
                    exc = urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.headers,
                                                 io.BytesIO(raw_error))
                self._raise_http_error(exc, method=method, path=path, deadline=deadline)
            except PeerRunsHTTPError as failure:
                if probe_key is not None and exc.code in {401, 403}:
                    self._remember_auth_probe_rejection(probe_key, failure)
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            not_admitted = ambiguous and _is_proven_pre_admission_failure(exc)
            raise PeerRunsHTTPError(
                "peer RoomLink endpoint is unreachable", retryable=True,
                ambiguous=ambiguous and not not_admitted, not_admitted=not_admitted,
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PeerRunsHTTPError(
                "peer returned non-JSON data",
                retryable=ambiguous,
                ambiguous=ambiguous,
            ) from exc
        if not isinstance(payload, dict):
            raise PeerRunsHTTPError(
                "peer returned a non-object response",
                retryable=ambiguous,
                ambiguous=ambiguous,
            )
        return payload

    @staticmethod
    def _verify_proof_response(state, response, body, *, ambiguous):
        if state is None:
            return body
        from gateway.hosted_room_proof import verify_response, RESPONSE_HEADER, RESPONSE_NONCE_HEADER
        try:
            for header in (RESPONSE_HEADER, RESPONSE_NONCE_HEADER):
                if hasattr(response.headers, 'get_all') and len(response.headers.get_all(header, [])) != 1:
                    raise ValueError('duplicate or missing room response proof header')
            return verify_response(*state, response.status, body if isinstance(body, bytes) else body.encode('utf-8'),
                                   response.headers.get(RESPONSE_HEADER), response.headers.get(RESPONSE_NONCE_HEADER))
        except ValueError as exc:
            # An unverified endpoint's 4xx is no evidence that the intended target rejected admission.
            raise PeerRunsHTTPError('peer installation response proof failed', ambiguous=ambiguous) from exc

    @staticmethod
    def _raise_http_error(
        exc: urllib.error.HTTPError, *, method: str, path: str, deadline: float) -> NoReturn:
        """Raise the classified PeerRunsHTTPError for an HTTP error response."""
        # A 4xx on admission proves the peer never admitted the run.
        flags = {
            "ambiguous": method == "POST" and exc.code >= 500, "status_code": exc.code,
            "not_admitted": method == "POST" and path == "/v1/runs" and 400 <= exc.code < 500}
        try:
            detail = _read_body(
                exc, max_bytes=MAX_PEER_ERROR_RESPONSE_BYTES, deadline=deadline, kind=" error",
                **flags)[:500]
        except PeerRunsHTTPError:
            raise
        except Exception:
            detail = ""
        error_code = _response_error_code(detail)
        logger.debug(
            "Peer RoomLink request returned HTTP %s (%s)", exc.code, error_code or "no-code")
        renewal = exc.code in {401, 403} and error_code in _GRANT_RENEWAL_CODES
        drift = exc.code == 403 and error_code in {
            _EXECUTION_POLICY_CHANGED[0], _CAPABILITY_CHANGED[0]}
        raise PeerRunsHTTPError(
            _REAUTHORIZATION_MESSAGES[error_code] if renewal or drift
            else f"peer rejected {method} {path} with HTTP {exc.code}",
            retryable=exc.code in {408, 425, 429} or exc.code >= 500,
            error_code=error_code, **flags,
        ) from exc

    def prepare(
        self, *, room_id: str, profile: str, source: str, grant: str, create: bool,
        expected_session_id: str | None = None) -> Mapping[str, Any] | None:
        if source != "bot_room":
            raise PeerRunsHTTPError("peer room source must be bot_room")
        self._require_room_grant(grant)
        logical_session = "roomlink_" + hashlib.sha256(
            f"{room_id}\0{profile}".encode("utf-8")).hexdigest()[:32]
        if expected_session_id and expected_session_id != logical_session:
            raise PeerRunsHTTPError("peer room session identity changed")
        return {"session_id": logical_session, "title": f"Group: {room_id}", "source": source}

    def _checked_dispatch(self, dispatch: Mapping[str, Any], grant: str) -> HostedMemberDispatch:
        """Validate a dispatch, its grant, and pin scope + observation to it."""
        checked = HostedMemberDispatch.from_mapping(dispatch)
        self._require_room_grant(grant)
        self.bind_room_scope(**{f: getattr(checked, f) for f in _RECEIPT_SCOPE_FIELDS})
        self.bind_observation(
            task_id=checked.task_id, execution_generation=checked.execution_generation)
        return checked

    def dispatch(self, *, dispatch: Mapping[str, Any], grant: str) -> Mapping[str, Any]:
        return self._admit_dispatch(self._checked_dispatch(dispatch, grant), grant=grant)

    def recover_dispatch(self, *, dispatch: Mapping[str, Any], grant: str) -> Mapping[str, Any]:
        """Recover one exact admission by receipt or idempotent POST replay."""
        accepted = self.recover_accepted_dispatch(dispatch=dispatch, grant=grant)
        if accepted is not None:
            return accepted
        checked = self._checked_dispatch(dispatch, grant)
        key, now = (checked.task_id, checked.execution_generation), self.clock()
        backoff = self._recovery_backoff.get(key)
        if backoff is not None and now < float(backoff["next_attempt_at"]):
            raise PeerRunsHTTPError(
                "peer admission recovery is backing off", retryable=True, ambiguous=True)
        try:
            recovered = self._admit_dispatch(checked, grant=grant)
        except PeerRunsHTTPError as exc:
            failure = exc
            if (checked.document_inputs or checked.document_output) and exc.not_admitted:
                failure = PeerRunsHTTPError(str(exc), ambiguous=True, retryable=exc.retryable,
                                           status_code=exc.status_code, error_code=exc.error_code)
            if failure.retryable or failure.ambiguous:
                delay = self._next_poll_delay(backoff)
                self._recovery_backoff = {key: {"delay": delay, "next_attempt_at": now + delay}}
            if failure is not exc:
                raise failure from exc
            raise
        self._recovery_backoff.pop(key, None)
        return recovered

    def recover_accepted_dispatch(self, *, dispatch: Mapping[str, Any], grant: str) -> Mapping[str, Any] | None:
        """Recover only an existing exact receipt; never send an admission request."""
        checked = self._checked_dispatch(dispatch, grant)
        existing = self._receipt(checked.task_id, checked.execution_generation)
        if existing is None:
            return None
        if any(existing[field] != getattr(checked, field) for field in _RECEIPT_SCOPE_FIELDS):
            raise PeerRunsHTTPError("peer run receipt conflicts with the recovered dispatch")
        return self._accepted(checked, run_id=str(existing["run_id"]),
                              session_id=str(existing["session_id"]), replayed=True)

    @staticmethod
    def _accepted(
        checked: HostedMemberDispatch, *, run_id: str, session_id: str, replayed: bool,
    ) -> dict[str, Any]:
        return {
            "status": "accepted", "task_id": checked.task_id,
            "execution_generation": checked.execution_generation, "run_id": run_id,
            "session_id": session_id, "replayed": replayed}

    def _admit_dispatch(self, checked: HostedMemberDispatch, *, grant: str) -> Mapping[str, Any]:
        if checked.document_output is not None:
            from tui_gateway.hosted_room_peer_output import mark_dispatched, stored_consent
            saved = stored_consent(self.receipt_db_path, checked.as_mapping()) if self.receipt_db_path else None
            if saved is None:
                receipt = self._recover_output_receipt(checked, grant=grant)
                return self._accepted(checked, run_id=receipt['run_id'], session_id=receipt['session_id'], replayed=True)
            mark_dispatched(self.receipt_db_path, checked)
        session_id = self._session_id(checked, grant=grant)

        def admit() -> dict[str, Any]:
            body = {"input": checked.prompt, "hosted_room_dispatch": checked.as_mapping()}
            def post():
                return self._request("/v1/runs", method="POST", body=body,
                    headers={"Idempotency-Key": f"room:{checked.task_id}:{checked.execution_generation}"},
                    room_grant=grant)
            try:
                result = post()
            except PeerRunsHTTPError as exc:
                if not (checked.document_inputs and exc.error_code == "room_document_input_required"
                        and exc.status_code == 409 and exc.not_admitted and not exc.ambiguous):
                    raise
                # Only an exact no-admission receipt permits new ingress checks/source reads.
                # Capability changes never prevent observation of previously accepted input.
                capability = self._request("/v1/room-members/capabilities", room_grant=grant,
                                           headers={"Hermes-Room-Features": "document-input-v1"})
                live = GatewayRoomCatalog.from_mapping(capability.get("catalog"))
                if (live.catalog_digest != checked.capability_digest or live.installation_id != checked.target_install_id
                        or any(capability.get(field) != getattr(checked, field)
                               for field in _RECEIPT_SCOPE_FIELDS if field != "target_install_id")):
                    raise PeerRunsHTTPError("peer document capability scope changed", ambiguous=True)
                from gateway.hosted_room_documents import manifest, check_capability
                try:
                    limits = check_capability(capability.get("document_inputs"))
                    if limits is None:
                        raise ValueError("peer supports text only")
                    manifest(checked.document_inputs, member_id=checked.member_id, capability=limits)
                except ValueError as error:
                    raise PeerRunsHTTPError("peer document availability changed", ambiguous=True) from error
                from tui_gateway.hosted_room_peer_documents import transfer_documents
                try:
                    body["document_bytes"] = transfer_documents(self.receipt_db_path, checked)
                except (OSError, ValueError) as error:
                    raise PeerRunsHTTPError("peer document source unavailable", not_admitted=True) from error
                result = post()

            if not str(result.get("run_id") or ""):
                raise PeerRunsHTTPError(
                    "peer did not return a run id",
                    retryable=True,
                    ambiguous=True,
                )
            return result

        try:
            result = admit()
        except PeerRunsHTTPError as first_error:
            if not first_error.ambiguous:
                raise
            try:
                result = admit()
            except PeerRunsHTTPError as replay_error:
                raise PeerRunsHTTPError(
                    str(replay_error),
                    retryable=(first_error.retryable or replay_error.retryable),
                    ambiguous=True,
                    not_admitted=False,
                    status_code=replay_error.status_code,
                    error_code=replay_error.error_code,
                ) from replay_error
        run_id = str(result.get("run_id") or "")
        receipt = {
            "run_id": run_id, "session_id": session_id,
            **{field: getattr(checked, field) for field in _RECEIPT_SCOPE_FIELDS},
            "task_id": checked.task_id, "execution_generation": checked.execution_generation}
        if self.receipt_db_path is not None:
            from gateway import hosted_rooms
            hosted_rooms.upsert_remote_run_receipt(self.receipt_db_path, record=receipt)
        self._runs[(checked.task_id, checked.execution_generation)] = receipt
        self._status_cache.pop(run_id, None)
        return self._accepted(
            checked, run_id=run_id, session_id=session_id,
            replayed=bool(result.get("replayed", False)))

    def _session_id(self, dispatch: HostedMemberDispatch, *, grant: str) -> str:
        existing = self._receipt(dispatch.task_id, dispatch.execution_generation)
        if existing:
            return str(existing["session_id"])
        prepared = self.prepare(
            room_id=dispatch.room_id, profile=dispatch.target_profile, source="bot_room",
            grant=grant, create=True)
        if prepared is None:
            raise PeerRunsHTTPError("peer room session is unavailable")
        return str(prepared.get("session_id") or prepared.get("id") or "")

    def _observation_receipt(
        self, *, room_id: str, profile: str, session_id: str) -> dict[str, Any] | None:
        record = None if self._observation_key is None else self._receipt(*self._observation_key)
        if record is None:
            return None
        scope = (record["room_id"], record["target_profile"], record["session_id"])
        if scope != (room_id, profile, session_id):
            raise PeerRunsHTTPError("peer observation receipt changed scope")
        return record

    def _next_poll_delay(self, cached: Mapping[str, Any] | None) -> float:
        previous = float(cached["delay"]) if cached is not None else self.poll_min_seconds / 2
        return min(self.poll_max_seconds, max(self.poll_min_seconds, previous * 2))

    def _poll_receipt(self, record: Mapping[str, Any], *, grant: str) -> dict[str, Any]:
        run_id, now = str(record["run_id"]), self.clock()
        cached = self._status_cache.get(run_id)
        fingerprint = hashlib.sha256(grant.encode()).hexdigest()
        if (cached is not None and getattr(cached.get("error"), "needs_reauthorization", False)
                and cached.get("grant_sha256") != fingerprint):
            cached = None  # a retired grant's refusal must not answer for its replacement
        if cached is not None:
            status = cached["status"]
            if status.get("status") in _TERMINAL_RUN_STATES and not status.get("peer_output_unresolved"):
                return status
            if now < float(cached["next_poll_at"]):
                error = cached.get("error")
                if isinstance(error, PeerRunsHTTPError):
                    raise error
                return status
        delay = self._next_poll_delay(cached)
        entry = {"delay": delay, "next_poll_at": now + delay, "grant_sha256": fingerprint}
        try:
            full = self._request(_run_path(record), room_grant=self._require_room_grant(grant))
            status = {key: full[key] for key in _RUN_STATUS_KEYS if key in full}
            if (
                str(status.get("run_id") or "") != run_id
                or status.get("status") not in _KNOWN_RUN_STATES):
                raise PeerRunsHTTPError("peer returned a mismatched run status")
        except PeerRunsHTTPError as exc:
            previous = cached["status"] if cached is not None else {}
            self._status_cache = {run_id: {"status": previous, "error": exc, **entry}}
            raise
        self._status_cache = {run_id: {"status": status, **entry}}
        if status.get("status") in _TERMINAL_RUN_STATES and not status.get("peer_output_unresolved"):
            self._terminal_receipts.add(
                (str(record["task_id"]), int(record["execution_generation"])))
        return status

    def history(
        self, *, room_id: str, profile: str, session_id: str, grant: str,
    ) -> Sequence[Mapping[str, Any]]:
        receipt = self._observation_receipt(room_id=room_id, profile=profile, session_id=session_id)
        if receipt is None:
            return []
        status = self._poll_receipt(receipt, grant=grant)
        state = str(status.get("status") or "")
        if status.get("peer_output_unresolved") or state not in {"completed", "failed", "interrupted"}:
            return []
        return [{
            "role": "assistant", "task_id": receipt["task_id"],
            "execution_generation": receipt["execution_generation"],
            "status": "settled" if state == "completed" else "failed",
            "message_id": f"peer-run:{status.get('run_id')}",
            "content": status.get("output") or status.get("error") or "",
            **{key: status[key] for key in ("artifacts", "artifact_scope", "peer_output_empty") if key in status}}]

    def status(
        self, *, room_id: str, profile: str, session_id: str, grant: str) -> Mapping[str, Any]:
        receipt = self._observation_receipt(room_id=room_id, profile=profile, session_id=session_id)
        if receipt is None:
            return {"active": False, "task_id": None}
        status = self._poll_receipt(receipt, grant=grant)
        approval = next((p for p in status.get("pending_controls", []) if p.get("kind") == "approval"), None)
        if approval is not None:
            approval = {**approval, "request_id": approval["prompt_id"]}
        return {
            "active": status.get("status") in _ACTIVE_RUN_STATES, "task_id": receipt["task_id"],
            "execution_generation": receipt["execution_generation"],
            "status": "unknown" if status.get("peer_output_unresolved") else status.get("status"), "run_id": status.get("run_id"),
            "approval": approval or status.get("approval")}

    def approve_receipt(
        self, *, task_id: str, execution_generation: int, request_id: str, choice: str, grant: str
    ) -> Mapping[str, Any] | None:
        """Resolve approval for the exact durable remote run."""
        record = self._receipt(task_id, execution_generation)
        if record is None:
            return None
        request_id = str(request_id or "").strip()
        if not request_id:
            raise PeerRunsHTTPError("an exact approval request_id is required")
        self._require_room_grant(grant)
        body = {"choice": choice, "request_id": request_id}
        if self.proof_install_id is not None:
            status = self._poll_receipt(record, grant=grant)
            prompt = next((p for p in status.get("pending_controls", [])
                           if p.get("kind") == "approval" and p.get("prompt_id") == request_id), None)
            if prompt is None:
                raise PeerRunsHTTPError("the exact peer approval is no longer pending")
            body["execution_generation"] = prompt["execution_generation"]
        return self._post_run_action(record, "approval", body=body, grant=grant)

    def _post_run_action(
        self, record: Mapping[str, Any], action: str, *, body: dict[str, Any], grant: str
    ) -> dict[str, Any]:
        result = self._request(
            _run_path(record, action), method="POST", body=body, room_grant=grant)
        self._status_cache.pop(str(record["run_id"]), None)
        return result

    def stop(self, *, dispatch: Mapping[str, Any], grant: str) -> Mapping[str, Any] | None:
        checked = HostedMemberDispatch.from_mapping(dispatch)
        self.bind_room_scope(**{f: getattr(checked, f) for f in _RECEIPT_SCOPE_FIELDS})
        return self.stop_receipt(
            task_id=checked.task_id, execution_generation=checked.execution_generation, grant=grant)

    def stop_receipt(
        self, *, task_id: str, execution_generation: int, grant: str) -> Mapping[str, Any] | None:
        """Stop the exact durable remote run after a home restart."""
        record = self._receipt(task_id, execution_generation)
        if record is None:
            return None
        result = self._post_run_action(
            record, "stop", body={}, grant=self._require_room_grant(grant))
        if result.get("status") in _TERMINAL_RUN_STATES:
            self._terminal_receipts.add((str(task_id), int(execution_generation)))
        return result

    def _recover_output_receipt(self, checked, *, grant):
        """Prove an already accepted exact attempt without ever POSTing a Run."""
        from gateway.hosted_room_peer_output import dispatch_digest
        from gateway.platforms.api_server_run_scope import room_run_scope_key
        if self.proof_install_id != checked.target_install_id:
            raise PeerRunsHTTPError('peer output receipt requires exact target proof', ambiguous=True)
        run_id = 'run_' + hashlib.sha256((room_run_scope_key(checked.as_mapping()) + '\0' +
            f"room:{checked.task_id}:{checked.execution_generation}").encode()).hexdigest()[:32]
        try:
            status = self._request('/v1/runs/' + run_id, room_grant=grant)
        except PeerRunsHTTPError as exc:
            # A missing/read-failed receipt is not proof that this attempt never ran.
            raise PeerRunsHTTPError(str(exc), ambiguous=True, retryable=exc.retryable,
                                   status_code=exc.status_code, error_code=exc.error_code) from exc
        if (status.get('run_id') != run_id or status.get('status') not in _KNOWN_RUN_STATES
                or status.get('peer_output_dispatch_digest') != dispatch_digest(checked)):
            raise PeerRunsHTTPError('peer output accepted dispatch changed or is unavailable', ambiguous=True)
        receipt = {**{field: getattr(checked, field) for field in _RECEIPT_SCOPE_FIELDS},
            'task_id': checked.task_id, 'execution_generation': checked.execution_generation,
            'run_id': run_id, 'session_id': self._session_id(checked, grant=grant)}
        if self.receipt_db_path is not None:
            from gateway import hosted_rooms
            hosted_rooms.upsert_remote_run_receipt(self.receipt_db_path, record=receipt)
        self._runs[(checked.task_id, checked.execution_generation)] = receipt
        return receipt

    def output_request(self, *, scope, manifest_digest, operation, grant, **fields):
        self.bind_room_scope(**{field: scope[field] for field in _RECEIPT_SCOPE_FIELDS})
        receipt = self._receipt(scope['task_id'], scope['execution_generation'])
        if receipt is None:
            # Observation recovery never creates a Run. Canonical accepted state
            # can outlive both Home's receipt and the target's HTTP receipt.
            from tui_gateway.hosted_room_peer_output import stored_consent
            from gateway.hosted_room_peer_output import output_scope
            consent = stored_consent(self.receipt_db_path, scope) if self.receipt_db_path else None
            if consent is None or consent.get('dispatch') is None:
                raise PeerRunsHTTPError("peer output Run receipt unavailable")
            checked = HostedMemberDispatch.from_mapping(consent['dispatch'])
            if output_scope(checked).as_mapping() != scope:
                raise PeerRunsHTTPError("peer output persisted dispatch changed")
            receipt = self._recover_output_receipt(checked, grant=grant)
        return self._request(_run_path(receipt, 'artifacts', operation), method='POST', room_grant=grant,
            body={'artifact_scope': scope, 'manifest_digest': manifest_digest, **fields})

    def issue_invitation(
        self, *, room_id: str, home_install_id: str, authority_gateway_id: str,
        authority_epoch: int, member_id: str, grant_id: str, ttl_seconds: float = 3600,
        status_ttl_seconds: float | None = None) -> Mapping[str, Any]:
        """Ask the target gateway to mint a scoped room-member grant."""
        if not self.api_key:
            raise PeerRunsHTTPError("issuing an invitation requires the target gateway API key")
        return self._request(
            "/v1/room-members/invitations", method="POST",
            body={
                "room_id": room_id, "home_install_id": home_install_id,
                "authority_gateway_id": authority_gateway_id, "authority_epoch": authority_epoch,
                "member_id": member_id, "grant_id": grant_id, "ttl_seconds": ttl_seconds,
                **({} if status_ttl_seconds is None else {
                    "status_ttl_seconds": status_ttl_seconds})})

    def refresh_grant(
        self, *, grant: str, ttl_seconds: float = 24 * 60 * 60,
        capability_digest: str | None = None, execution_policy_digest: str | None = None,
        on_issued=None,
    ) -> Mapping[str, Any]:
        """Renew dispatch access only while its frozen authority is unchanged."""
        refreshed = self._scoped_post(
            "/v1/room-members/grants/refresh", grant, body={"ttl_seconds": ttl_seconds})
        replacement = str(refreshed.get("grant") or "")
        if not replacement:
            raise PeerRunsHTTPError("peer returned no refreshed room grant")
        try:
            if on_issued is not None:
                on_issued(replacement)
            # Route publication still requires a live scoped probe; custody precedes it.
            probe = self.probe(grant=replacement)
            error = digest_reauthorization_error(
                GatewayRoomCatalog.from_mapping(probe.get("catalog")),
                capability_digest=capability_digest, execution_policy_digest=execution_policy_digest)
            if error is not None:
                raise error
        except Exception:
            # A replacement nobody will use is retired, not left live until it expires.
            try:
                self.revoke_grant_exact(grant=replacement)
            except Exception:
                logger.warning("Could not retire an unused refreshed room grant")
            raise
        return {**refreshed, "catalog": probe.get("catalog")}

    def cleanup_issuance(self, *, grant: str, request_id: str) -> Mapping[str, Any]:
        """Retire one unredeemed refresh result without receiving its dispatch bearer."""
        return self._acknowledged(self._scoped_post(
            "/v1/room-members/grants/cleanup-issuance", grant, body={'request_id': request_id}))

    def revoke_grant(self, *, grant: str) -> Mapping[str, Any]:
        """Revoke this grant's exact room/home/target/profile scope."""
        return self._acknowledged(self._scoped_post("/v1/room-members/grants/revoke", grant, body={}))

    def revoke_grant_exact(self, *, grant: str) -> Mapping[str, Any]:
        """Retire this one grant, never the replacement that shares its scope."""
        return self._acknowledged(self._scoped_post("/v1/room-members/grants/revoke-exact", grant, body={}))

    @staticmethod
    def _acknowledged(result: Mapping[str, Any]) -> Mapping[str, Any]:
        if result.get("revoked") is not True:
            raise PeerRunsHTTPError("peer did not acknowledge the grant revocation", retryable=True)
        return result

    def probe(self, *, grant: str, features: str | None = None) -> Mapping[str, Any]:
        """Verify gateway reachability and the live scoped capability catalog."""
        return self._request(
            "/v1/room-members/capabilities", room_grant=self._require_room_grant(grant),
            headers={"Hermes-Room-Features": features} if features else None)

    def replicate_page(
        self, *, grant: str, room_id: str, room_name: str, members: list[dict[str, Any]],
        page: dict[str, Any]) -> Mapping[str, Any]:
        """Send one history page with a ``replicate`` grant, never broad API auth.

        Pages are bounded in UTF-8; escaping non-ASCII text would multiply their wire size.
        """
        return self._request(
            "/v1/room-members/replica", method="POST", room_grant=self._require_room_grant(grant),
            ensure_ascii=False,
            body={"room_id": room_id, "room_name": room_name, "members": members, "page": page})

    def replicate_work_records(self, *, grant: str, record: dict[str, Any]) -> Mapping[str, Any]:
        """Send one whole task-evidence record with a ``work_records`` grant."""
        return self._request(
            "/v1/room-members/work-records", method="POST", room_grant=self._require_room_grant(grant),
            ensure_ascii=False, body={"record": record})

    def retire_replica(self, notice) -> Mapping[str, Any]:
        """Deliver the exact signed enrollment retirement through pinned installation proof."""
        if re.search(r"/p/[^/]+$", urllib.parse.urlsplit(self.base_url).path):
            raise PeerRunsHTTPError("retirement requires the installation endpoint")
        if self.proof_install_id != notice.target_install_id:
            raise PeerRunsHTTPError('copy retirement requires pinned installation proof')
        return self._request(
            "/v1/group-replicas/retire", method="POST",
            body={'notice': notice.payload(), 'signature': notice.value},
            room_grant=self._require_room_grant(notice.proof_grant))

    def _scoped_post(self, path: str, grant: str, *, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            path, method="POST", body=body, room_grant=self._require_room_grant(grant))

    @staticmethod
    def _require_room_grant(grant: str) -> str:
        """Prevent scoped operations from falling back to broad Bearer auth."""
        value = str(grant or "")
        if not value or value in {"compat", "compatibility-only"}:
            raise PeerRunsHTTPError("a scoped room grant is required")
        return value

    def _check_auth_probe_cooldown(self, key: tuple[str, str, str]) -> None:
        with self._auth_probe_lock:
            rejected = self._auth_probe_rejections.get(key)
            if rejected is not None and self.clock() >= rejected[0]:
                del self._auth_probe_rejections[key]
                rejected = None
        if rejected is not None:
            _, message, status_code, error_code = rejected
            raise PeerRunsHTTPError(message, status_code=status_code, error_code=error_code)

    def _remember_auth_probe_rejection(self, key: tuple[str, str, str], failure: PeerRunsHTTPError) -> None:
        with self._auth_probe_lock:
            self._auth_probe_rejections[key] = (
                self.clock() + _AUTH_PROBE_COOLDOWN_SECONDS, str(failure), failure.status_code, failure.error_code)
            self._auth_probe_rejections.move_to_end(key)
            while len(self._auth_probe_rejections) > _MAX_AUTH_PROBE_REJECTIONS:
                self._auth_probe_rejections.popitem(last=False)


def _open_roomlink_url(request: urllib.request.Request, *, timeout: float, reject_redirects: bool = False):
    """Open with the installed transport policy; within a renewal, refuse every redirect."""
    from hermes_cli import urllib_security
    if not reject_redirects:
        return urllib_security.open_credentialed_url(request, timeout=timeout)

    def refusing_opener(_redirect_handler):
        policy = urllib_security._secure_opener_from_installed_policy(request.full_url)
        for name, value in getattr(policy, "_hermes_initial_addheaders", ()):
            if not request.has_header(name):
                request.add_header(name, value)
        handlers = [handler for handler in getattr(policy, "handlers", ())
                    if not isinstance(handler, urllib.request.HTTPRedirectHandler)]
        opener = urllib.request.build_opener(*handlers, _RefuseRedirects())
        opener.addheaders = []
        return opener
    return urllib_security.open_credentialed_url(request, timeout=timeout, opener_factory=refusing_opener)


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    handler_order = 100

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)
