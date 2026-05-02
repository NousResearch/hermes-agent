"""TTM control-plane ingress skill (PR-F-H2 of the Hermes alignment plan).

Hermes runtime calls this module to write canonical state back to TTM
during a control-plane run: events, evidence, and approval requests.

Wire contract — ``RUNTIME-ADAPTER-CONTRACT.md §Principal-Scoped Ingress``:

  POST {ingress_base_url}/api/ingress/runtime/{runtime_id}/events
  POST {ingress_base_url}/api/ingress/runtime/{runtime_id}/evidence
  POST {ingress_base_url}/api/ingress/runtime/{runtime_id}/approvals

  Headers (all three):
    Authorization: Bearer <principal_token>
    X-Runtime-Id: hermes
    X-Run-Id: <run_id>

The dispatch receiver (``plugins/ttm-control-plane``, PR-F-H1) injects
the per-run ``principal_token`` + ``ingress_base_url`` into this module
at session start via :func:`bind_run`. Subsequent skill calls resolve
the bound context by ``run_id`` and post to the right TTM instance.

Per ``RUNTIME-PRINCIPAL-CONTRACT.md`` the principal token is a per-run
bearer credential. Plaintext is never logged — only the first 8 chars
plus an ellipsis. On 401 the call raises immediately and does not retry
(the token has been revoked). On 5xx the call retries up to three times
with exponential backoff before raising.
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical event types — RUNTIME-ADAPTER-CONTRACT.md §Events
# ---------------------------------------------------------------------------

EVENT_RUN_DISPATCHED = "run.dispatched"
EVENT_PHASE_ENTERED = "phase.entered"
EVENT_PHASE_COMPLETED = "phase.completed"
EVENT_TASK_UPDATED = "task.updated"
EVENT_EVIDENCE_ADDED = "evidence.added"
EVENT_APPROVAL_REQUESTED = "approval.requested"
EVENT_APPROVAL_GRANTED = "approval.granted"
EVENT_APPROVAL_REJECTED = "approval.rejected"
EVENT_RUNTIME_ERROR = "runtime.error"
EVENT_RUN_CLOSED = "run.closed"

CANONICAL_EVENT_TYPES = frozenset(
    {
        EVENT_RUN_DISPATCHED,
        EVENT_PHASE_ENTERED,
        EVENT_PHASE_COMPLETED,
        EVENT_TASK_UPDATED,
        EVENT_EVIDENCE_ADDED,
        EVENT_APPROVAL_REQUESTED,
        EVENT_APPROVAL_GRANTED,
        EVENT_APPROVAL_REJECTED,
        EVENT_RUNTIME_ERROR,
        EVENT_RUN_CLOSED,
    }
)

DEFAULT_RUNTIME_ID = "hermes"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 0.5

# ---------------------------------------------------------------------------
# Bootstrap env vars — the dispatch-spawn shim sets these on the agent
# process before exec, and :func:`bind_run_from_env` reads them at session
# start. Until PR-F-H1's ``_spawn_headless_session`` is finished, operators
# can also set these manually for local dev/testing of skills that depend
# on TTM ingress.
# ---------------------------------------------------------------------------

ENV_RUN_ID = "TTM_RUN_ID"
ENV_PRINCIPAL_TOKEN = "TTM_PRINCIPAL_TOKEN"
ENV_INGRESS_BASE_URL = "TTM_INGRESS_BASE_URL"
ENV_RUNTIME_ID = "TTM_RUNTIME_ID"
ENV_SCOPE_EPOCH = "TTM_SCOPE_EPOCH"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IngressError(Exception):
    """Base class for TTM ingress failures."""


class IngressNotBoundError(IngressError):
    """Raised when an ingress call is made for a run_id with no bound context."""


class IngressAuthError(IngressError):
    """Raised on 401 — the principal token was rejected. Do not retry."""

    def __init__(self, run_id: str, body: str = "") -> None:
        super().__init__(f"principal_token_rejected for run_id={run_id}")
        self.run_id = run_id
        self.body = body


class IngressClientError(IngressError):
    """Raised on a non-401 4xx response — the request was malformed."""

    def __init__(self, run_id: str, status_code: int, body: str = "") -> None:
        super().__init__(
            f"ingress request rejected status={status_code} run_id={run_id}"
        )
        self.run_id = run_id
        self.status_code = status_code
        self.body = body


class IngressServerError(IngressError):
    """Raised when retries are exhausted on a 5xx response or transport error."""

    def __init__(
        self,
        run_id: str,
        attempts: int,
        last_status: int | None = None,
        last_error: str = "",
    ) -> None:
        super().__init__(
            f"ingress server error after {attempts} attempts run_id={run_id} "
            f"last_status={last_status} last_error={last_error}"
        )
        self.run_id = run_id
        self.attempts = attempts
        self.last_status = last_status
        self.last_error = last_error


# ---------------------------------------------------------------------------
# Per-run binding registry
# ---------------------------------------------------------------------------


@dataclass
class _IngressBinding:
    """Live ingress context for one TTM control-plane run."""

    run_id: str
    principal_token: str
    ingress_base_url: str
    runtime_id: str = DEFAULT_RUNTIME_ID
    scope_epoch: int = 1
    bound_at: float = field(default_factory=time.time)


def _token_present(token: str) -> str:
    """Return a fixed marker indicating whether a principal token is bound.

    Logging even a truncated prefix risks leakage in long-running
    production logs (post-mortem from operator review). Emit only a
    boolean signal — the caller already has run_id for correlation.
    """
    return "set" if token else "unset"


def _humanize_event(event_type: str) -> str:
    """Best-effort summary fallback when the caller does not provide one."""
    return event_type.replace(".", " ").replace("_", " ")


# ---------------------------------------------------------------------------
# TtmIngress — public surface
# ---------------------------------------------------------------------------


class TtmIngress:
    """Per-process registry + HTTP wire for TTM control-plane ingress.

    Thread-safe. Bind a run with :meth:`bind_run`, then call
    :meth:`post_event`, :meth:`post_evidence`, or :meth:`request_approval`
    by ``run_id``. Use :meth:`unbind_run` at run close to drop the token
    from process memory.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        client_factory: Callable[[], httpx.Client] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._lock = threading.Lock()
        self._by_run: dict[str, _IngressBinding] = {}
        self._timeout = timeout_seconds
        self._max_retries = max(1, int(max_retries))
        self._retry_base_delay = retry_base_delay
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=timeout_seconds)
        )
        self._sleep = sleep

    # -- binding management --------------------------------------------------

    def bind_run(
        self,
        run_id: str,
        principal_token: str,
        ingress_base_url: str,
        *,
        runtime_id: str = DEFAULT_RUNTIME_ID,
        initial_scope_epoch: int = 1,
    ) -> None:
        """Register the ingress context for ``run_id``.

        Called by the dispatch receiver / headless agent bootstrap once
        per run. Subsequent ingress calls in the run resolve their token
        and base URL from this binding. Re-binding the same run_id is a
        no-op unless ``principal_token`` differs (rebinds replace the
        token; this is what TTM's ``POST /control-plane/{run_id}/rebind``
        triggers after a scope expansion).
        """
        if not run_id:
            raise ValueError("run_id is required")
        if not principal_token:
            raise ValueError("principal_token is required")
        if not ingress_base_url:
            raise ValueError("ingress_base_url is required")

        with self._lock:
            existing = self._by_run.get(run_id)
            if existing is None:
                self._by_run[run_id] = _IngressBinding(
                    run_id=run_id,
                    principal_token=principal_token,
                    ingress_base_url=ingress_base_url.rstrip("/"),
                    runtime_id=runtime_id,
                    scope_epoch=int(initial_scope_epoch),
                )
                logger.debug(
                    "ttm_ingress.bind_run run_id=%s base=%s token=%s",
                    run_id,
                    ingress_base_url,
                    _token_present(principal_token),
                )
                return
            existing.principal_token = principal_token
            existing.ingress_base_url = ingress_base_url.rstrip("/")
            existing.runtime_id = runtime_id
            existing.bound_at = time.time()
            logger.debug(
                "ttm_ingress.rebind_run run_id=%s base=%s token=%s",
                run_id,
                ingress_base_url,
                _token_present(principal_token),
            )

    def bind_run_from_env(
        self,
        env: Mapping[str, str] | None = None,
    ) -> str | None:
        """Bind the active run from bootstrap env vars set by the spawn shim.

        Reads ``TTM_RUN_ID`` + ``TTM_PRINCIPAL_TOKEN`` + ``TTM_INGRESS_BASE_URL``
        (and optional ``TTM_RUNTIME_ID`` / ``TTM_SCOPE_EPOCH``) from ``env``
        (defaults to :mod:`os.environ`) and registers the binding. Returns
        the bound ``run_id`` on success, or ``None`` if any required var is
        missing — callers can use that as a "not running under TTM control"
        signal.

        This is the contract between PR-F-H1's spawn path and the agent
        process: the spawn shim writes the per-run dispatch context into
        env vars, then exec's the agent; the agent's H3+ skills call this
        once at session start and stop carrying the token themselves.
        """
        source = env if env is not None else os.environ
        run_id = source.get(ENV_RUN_ID)
        token = source.get(ENV_PRINCIPAL_TOKEN)
        base = source.get(ENV_INGRESS_BASE_URL)
        if not (run_id and token and base):
            logger.debug(
                "ttm_ingress.bind_run_from_env.skipped missing=%s",
                [
                    name
                    for name, val in (
                        (ENV_RUN_ID, run_id),
                        (ENV_PRINCIPAL_TOKEN, token),
                        (ENV_INGRESS_BASE_URL, base),
                    )
                    if not val
                ],
            )
            return None
        runtime_id = source.get(ENV_RUNTIME_ID) or DEFAULT_RUNTIME_ID
        scope_raw = source.get(ENV_SCOPE_EPOCH, "1").strip() or "1"
        try:
            scope_epoch = int(scope_raw)
        except ValueError as exc:
            raise ValueError(
                f"{ENV_SCOPE_EPOCH}={scope_raw!r} is not an integer"
            ) from exc
        self.bind_run(
            run_id,
            token,
            base,
            runtime_id=runtime_id,
            initial_scope_epoch=scope_epoch,
        )
        return run_id

    def unbind_run(self, run_id: str) -> None:
        """Drop the bound principal_token for ``run_id`` from memory."""
        with self._lock:
            self._by_run.pop(run_id, None)
        logger.debug("ttm_ingress.unbind_run run_id=%s", run_id)

    def update_scope_epoch(self, run_id: str, scope_epoch: int) -> None:
        """Track the current scope_epoch so callers do not have to pass it."""
        with self._lock:
            binding = self._by_run.get(run_id)
            if binding is None:
                raise IngressNotBoundError(f"run_id={run_id} is not bound")
            binding.scope_epoch = int(scope_epoch)

    def is_bound(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._by_run

    def _binding(self, run_id: str) -> _IngressBinding:
        with self._lock:
            binding = self._by_run.get(run_id)
        if binding is None:
            raise IngressNotBoundError(
                f"run_id={run_id} is not bound — call bind_run() at session start"
            )
        return binding

    # -- ingress operations --------------------------------------------------

    def post_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        scope_epoch: int | None = None,
        summary: str | None = None,
        actor_type: str = "runtime",
        actor_id: str | None = None,
    ) -> str:
        """Append a canonical event record. Returns the new ``event_id``.

        ``event_type`` SHOULD be drawn from :data:`CANONICAL_EVENT_TYPES`;
        TTM accepts arbitrary strings but operator dashboards only render
        the canonical set. Non-canonical types log a WARNING locally so
        the caller notices in dev.

        ``actor_type`` defaults to ``"runtime"`` and ``actor_id`` defaults
        to the bound runtime_id (e.g. ``"hermes"``) — the normal case for
        agent-issued events. Set ``actor_type="human"`` (and leave
        ``actor_id=None``) for operator-issued events; TTM rejects an
        ``actor_id`` on human events because identity comes from auth.
        """
        if event_type not in CANONICAL_EVENT_TYPES:
            logger.warning(
                "ttm_ingress.post_event.non_canonical event_type=%s run_id=%s "
                "(allowed: %s)",
                event_type,
                run_id,
                sorted(CANONICAL_EVENT_TYPES),
            )

        binding = self._binding(run_id)
        body: dict[str, Any] = {
            "event_type": event_type,
            "actor_type": actor_type,
            "expected_scope_epoch": (
                int(scope_epoch) if scope_epoch is not None else binding.scope_epoch
            ),
            "summary": summary or _humanize_event(event_type),
            "payload": dict(payload or {}),
        }
        if actor_type == "human":
            # TTM enforces actor_id is omitted for human actors.
            body["actor_id"] = None
        else:
            body["actor_id"] = actor_id or binding.runtime_id

        response_body = self._post(
            binding,
            "events",
            body,
            op="post_event",
            extra_log={"event_type": event_type},
        )
        event_id = str(response_body.get("event_id", ""))
        if not event_id:
            logger.warning(
                "ttm_ingress.post_event.missing_event_id run_id=%s body_keys=%s",
                run_id,
                list(response_body.keys()),
            )
        return event_id

    def post_evidence(
        self,
        run_id: str,
        kind: str,
        subject: str,
        content_hash: str,
        storage_ref: str,
        source_event_id: str,
        verdict: str,
        *,
        verification_status: str = "passed",
        scope_epoch: int | None = None,
        evidence_id: str | None = None,
    ) -> str:
        """Append an evidence item linked to a prior event. Returns ``evidence_id``.

        TTM requires evidence to be content-addressed (sha256 of the
        artifact in ``content_hash``) and bound to a ``source_event_id``
        that already exists in the run — typically an event posted via
        :meth:`post_event` immediately before the artifact was produced.
        """
        binding = self._binding(run_id)
        body: dict[str, Any] = {
            "evidence_id": evidence_id or str(uuid.uuid4()),
            "kind": kind,
            "subject": subject,
            "content_hash": content_hash.lower(),
            "storage_ref": storage_ref,
            "expected_scope_epoch": (
                int(scope_epoch) if scope_epoch is not None else binding.scope_epoch
            ),
            "source_event_id": source_event_id,
            "verdict": verdict,
            "verification_status": verification_status,
        }
        response_body = self._post(
            binding,
            "evidence",
            body,
            op="post_evidence",
            extra_log={"kind": kind},
        )
        return str(response_body.get("evidence_id", body["evidence_id"]))

    def request_approval(
        self,
        run_id: str,
        approval_type: str,
        summary: str,
        *,
        scope_epoch: int | None = None,
        notes_ref: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Open an approval gate as ``status='requested'``. Returns ``approval_id``.

        ``approval_type`` is the gate name (e.g. ``"scope"``,
        ``"contract_lock"``); TTM treats it as the gate identifier.
        ``summary`` is the human-facing description of what the operator
        is being asked to approve.
        """
        binding = self._binding(run_id)
        body: dict[str, Any] = {
            "approval_type": approval_type,
            "expected_scope_epoch": (
                int(scope_epoch) if scope_epoch is not None else binding.scope_epoch
            ),
            "summary": summary,
            "notes_ref": notes_ref,
            "payload": dict(payload or {}),
        }
        response_body = self._post(
            binding,
            "approvals",
            body,
            op="request_approval",
            extra_log={"approval_type": approval_type},
        )
        return str(response_body.get("approval_id", ""))

    # -- ingress read --------------------------------------------------------

    def get_run_state(self, run_id: str) -> dict[str, Any]:
        """Return the current run state snapshot from TTM.

        Calls ``GET {ingress_base_url}/api/ingress/runtime/{runtime_id}/runs/{run_id}/state``
        with the same auth headers as the write routes. On success the response
        ``scope_epoch`` is auto-applied to the binding so subsequent write
        calls use the fresh epoch without the caller having to do it manually.

        Raises :exc:`IngressNotBoundError` if ``bind_run`` has not been called,
        :exc:`IngressAuthError` on 401 (token revoked — do not retry),
        :exc:`IngressClientError` on non-401 4xx, :exc:`IngressServerError`
        after retries exhausted.
        """
        binding = self._binding(run_id)
        response_body = self._request(
            binding,
            "GET",
            f"runs/{run_id}/state",
            op="get_run_state",
        )
        scope_epoch = response_body.get("scope_epoch")
        if scope_epoch is not None:
            try:
                self.update_scope_epoch(run_id, int(scope_epoch))
            except (ValueError, IngressNotBoundError):
                pass
        return response_body

    # -- HTTP wire -----------------------------------------------------------

    def _request(
        self,
        binding: _IngressBinding,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        op: str,
        extra_log: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one ingress request with retry/backoff + structured logging."""
        url = (
            f"{binding.ingress_base_url}/api/ingress/runtime/"
            f"{binding.runtime_id}/{path}"
        )
        headers: dict[str, str] = {
            "Authorization": f"Bearer {binding.principal_token}",
            "X-Runtime-Id": binding.runtime_id,
            "X-Run-Id": binding.run_id,
        }
        if method == "POST":
            headers["Content-Type"] = "application/json"
        log_ctx = {
            "op": op,
            "run_id": binding.run_id,
            "url": url,
            "token": _token_present(binding.principal_token),
            **(extra_log or {}),
        }

        last_status: int | None = None
        last_error: str = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._client_factory() as client:
                    if method == "GET":
                        response = client.get(url, headers=headers)
                    else:
                        response = client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last_error = repr(exc)
                logger.warning(
                    "ttm_ingress.%s.transport_error attempt=%s/%s ctx=%s err=%s",
                    op,
                    attempt,
                    self._max_retries,
                    log_ctx,
                    exc,
                )
                if attempt == self._max_retries:
                    raise IngressServerError(
                        binding.run_id, attempt, None, last_error
                    ) from exc
                self._sleep(self._retry_base_delay * (2 ** (attempt - 1)))
                continue

            status_code = response.status_code
            last_status = status_code

            if 200 <= status_code < 300:
                logger.debug(
                    "ttm_ingress.%s.ok status=%s ctx=%s",
                    op,
                    status_code,
                    log_ctx,
                )
                try:
                    return response.json()
                except ValueError:
                    return {}

            body_preview = response.text[:200] if response.text else ""

            if status_code == 401:
                logger.warning(
                    "ttm_ingress.%s.principal_token_rejected status=401 ctx=%s body=%s",
                    op,
                    log_ctx,
                    body_preview,
                )
                raise IngressAuthError(binding.run_id, body=body_preview)

            if 400 <= status_code < 500:
                logger.warning(
                    "ttm_ingress.%s.client_error status=%s ctx=%s body=%s",
                    op,
                    status_code,
                    log_ctx,
                    body_preview,
                )
                raise IngressClientError(
                    binding.run_id, status_code, body=body_preview
                )

            # 5xx — retry with exponential backoff.
            last_error = body_preview
            logger.warning(
                "ttm_ingress.%s.server_error attempt=%s/%s status=%s ctx=%s body=%s",
                op,
                attempt,
                self._max_retries,
                status_code,
                log_ctx,
                body_preview,
            )
            if attempt == self._max_retries:
                raise IngressServerError(
                    binding.run_id, attempt, last_status, last_error
                )
            self._sleep(self._retry_base_delay * (2 ** (attempt - 1)))

        # Loop falls through only if max_retries==0, which __init__ guards.
        raise IngressServerError(binding.run_id, self._max_retries, last_status, last_error)

    def _post(
        self,
        binding: _IngressBinding,
        path: str,
        body: dict[str, Any],
        *,
        op: str,
        extra_log: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(binding, "POST", path, body, op=op, extra_log=extra_log)


# ---------------------------------------------------------------------------
# Module-level singleton + thin functional API
# ---------------------------------------------------------------------------


_default_ingress: TtmIngress | None = None
_default_lock = threading.Lock()


def get_default() -> TtmIngress:
    """Return the per-process default :class:`TtmIngress` instance."""
    global _default_ingress
    if _default_ingress is None:
        with _default_lock:
            if _default_ingress is None:
                _default_ingress = TtmIngress()
    return _default_ingress


def bind_run(
    run_id: str,
    principal_token: str,
    ingress_base_url: str,
    *,
    runtime_id: str = DEFAULT_RUNTIME_ID,
    initial_scope_epoch: int = 1,
) -> None:
    """Bind the per-run ingress context on the default instance."""
    get_default().bind_run(
        run_id,
        principal_token,
        ingress_base_url,
        runtime_id=runtime_id,
        initial_scope_epoch=initial_scope_epoch,
    )


def bind_run_from_env(env: Mapping[str, str] | None = None) -> str | None:
    """Bind the default instance from bootstrap env vars (spawn-shim contract)."""
    return get_default().bind_run_from_env(env)


def unbind_run(run_id: str) -> None:
    get_default().unbind_run(run_id)


def update_scope_epoch(run_id: str, scope_epoch: int) -> None:
    get_default().update_scope_epoch(run_id, scope_epoch)


def post_event(
    run_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    scope_epoch: int | None = None,
    summary: str | None = None,
    actor_type: str = "runtime",
    actor_id: str | None = None,
) -> str:
    return get_default().post_event(
        run_id,
        event_type,
        payload,
        scope_epoch=scope_epoch,
        summary=summary,
        actor_type=actor_type,
        actor_id=actor_id,
    )


def post_evidence(
    run_id: str,
    kind: str,
    subject: str,
    content_hash: str,
    storage_ref: str,
    source_event_id: str,
    verdict: str,
    *,
    verification_status: str = "passed",
    scope_epoch: int | None = None,
    evidence_id: str | None = None,
) -> str:
    return get_default().post_evidence(
        run_id,
        kind,
        subject,
        content_hash,
        storage_ref,
        source_event_id,
        verdict,
        verification_status=verification_status,
        scope_epoch=scope_epoch,
        evidence_id=evidence_id,
    )


def request_approval(
    run_id: str,
    approval_type: str,
    summary: str,
    *,
    scope_epoch: int | None = None,
    notes_ref: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    return get_default().request_approval(
        run_id,
        approval_type,
        summary,
        scope_epoch=scope_epoch,
        notes_ref=notes_ref,
        payload=payload,
    )


def get_run_state(run_id: str) -> dict[str, Any]:
    """Return the current run state snapshot from TTM (auto-updates scope_epoch)."""
    return get_default().get_run_state(run_id)


__all__ = [
    "CANONICAL_EVENT_TYPES",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BASE_DELAY",
    "DEFAULT_RUNTIME_ID",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENV_INGRESS_BASE_URL",
    "ENV_PRINCIPAL_TOKEN",
    "ENV_RUNTIME_ID",
    "ENV_RUN_ID",
    "ENV_SCOPE_EPOCH",
    "EVENT_APPROVAL_GRANTED",
    "EVENT_APPROVAL_REJECTED",
    "EVENT_APPROVAL_REQUESTED",
    "EVENT_EVIDENCE_ADDED",
    "EVENT_PHASE_COMPLETED",
    "EVENT_PHASE_ENTERED",
    "EVENT_RUNTIME_ERROR",
    "EVENT_RUN_CLOSED",
    "EVENT_RUN_DISPATCHED",
    "EVENT_TASK_UPDATED",
    "IngressAuthError",
    "IngressClientError",
    "IngressError",
    "IngressNotBoundError",
    "IngressServerError",
    "TtmIngress",
    "bind_run",
    "bind_run_from_env",
    "get_default",
    "get_run_state",
    "post_event",
    "post_evidence",
    "request_approval",
    "unbind_run",
    "update_scope_epoch",
]
