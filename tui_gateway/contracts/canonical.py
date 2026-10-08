"""Canonical ``hermes-gateway-v1`` methods that only the session authority serves.

``gateway/session_controls.py::AuthorityConnection.dispatch`` answers these on the authenticated
native socket; the TUI sidecar (``server._methods``) never registers them, so they live in
``CANONICAL_METHODS`` rather than ``METHODS``. Shared names with different authority semantics are overridden in
``canonical_sessions`` / ``canonical_projections``; unchanged names reuse ``METHODS``. The dispatcher refuses an unknown or missing key with ``4001 invalid_params`` before the
handler runs; value checks stay in the handlers, which own their domain reasons.
"""

from __future__ import annotations

from typing import Literal

from .base import JsonValue, Params, Result
from .common import OpenModel
from .groups_bot_relay import RoomParams, RoomTaskReceipt
from .registry import canonical_method

# ── admissions ────────────────────────────────────────────────────────────────────────────────


class CanonicalSessionRef(Result):
    profile_id: str
    session_id: str


class AdmissionStatus(Result):
    """``gateway/session_contract.py::AdmissionReceipt`` as ``dataclasses.asdict`` sends it."""

    admission_id: str
    ref: CanonicalSessionRef
    sequence: int
    status: Literal["queued", "started", "unknown", "terminal"]
    outcome: str | None
    authority_epoch: int
    execution_generation: int | None


class PromptReceiptParams(Params):
    session_id: str
    admission_id: str
    include_result: bool | None = None


class PromptReceiptResult(AdmissionStatus):
    """``result`` / ``usage``: the structured turn result committed at settlement, only with
    ``include_result`` on a terminal admission that saved one."""

    result: JsonValue | None = None
    usage: dict[str, JsonValue] | None = None


canonical_method("prompt.receipt", params=PromptReceiptParams, result=PromptReceiptResult,
                 doc="Current receipt of one admission the caller submitted (or controls).")


class PromptCancelParams(Params):
    session_id: str
    admission_id: str


canonical_method("prompt.cancel", params=PromptCancelParams, result=AdmissionStatus,
                 doc="Retire a still-queued admission; a started one is unaffected.")


class PromptResolveUnknownParams(Params):
    session_id: str
    admission_id: str
    execution_generation: int


canonical_method("prompt.resolve_unknown", params=PromptResolveUnknownParams, result=AdmissionStatus,
                 doc="Acknowledge a turn lost across an owner restart; the FIFO behind it resumes. "
                     "Never requeues the lost input.")


# ── session ───────────────────────────────────────────────────────────────────────────────────


class SessionMutateParams(Params):
    """``operation`` + ``payload`` are validated by ``hermes_state_mutations.validate_action``;
    ``expected_generation`` is required for delete/rewind/reset/branch/model/compress."""

    session_id: str
    request_id: str
    expected_revision: int
    operation: str
    payload: dict[str, JsonValue]
    expected_generation: int | None = None


class SessionMutateResult(OpenModel):
    """The committed mutation receipt plus the operation's projection (``title``, ``archived``,
    ``branched_session_id``, compress ``status``/``lines``, …). ``revision`` is absent only on a
    read-only compress preview."""

    session_id: str
    operation: str
    revision: int | None = None


canonical_method("session.mutate", params=SessionMutateParams, result=SessionMutateResult,
                 doc="Revision-fenced, retry-idempotent session edit (rename, archive, sidebar, branch, "
                     "delete, rewind, reset, model, compress, import).")


class CanonicalSessionInfoParams(Params):
    session_id: str


class CanonicalSessionInfo(OpenModel):
    """``gateway/session_local.py::local_session_info``."""

    source: str
    model: str | None = None
    lazy: bool
    profile_id: str
    desktop_protocol: str
    profile_name: str
    cwd: str | None = None
    launch_request: dict[str, JsonValue] | None = None


canonical_method("session.info", params=CanonicalSessionInfoParams, result=CanonicalSessionInfo,
                 doc="Frozen launch policy projection of one local session.")


class SessionDetachParams(Params):
    session_id: str
    subscription_id: str


class SessionDetachResult(Result):
    session_id: str
    subscription_id: str
    detached: bool


canonical_method("session.detach", params=SessionDetachParams, result=SessionDetachResult,
                 doc="Drop this connection's subscription; a stale subscription id answers detached=false.")


class RuntimeDescribeParams(Params):
    pass


class SessionCreateDescriptor(Result):
    sources: list[str]
    parameters: list[str]


class RuntimeDescribeResult(Result):
    instance_id: str
    profile_id: str
    authority_epoch: int
    capabilities: list[str]
    session_create: SessionCreateDescriptor


canonical_method("runtime.describe", params=RuntimeDescribeParams, result=RuntimeDescribeResult,
                 doc="Owner identity and the canonical capabilities a client may rely on.")


class ClarifyRespondParams(Params):
    session_id: str
    execution_generation: int
    prompt_id: str
    answer: str


class PromptResponseResult(Result):
    status: Literal["resolved", "already_resolved"]
    prompt_id: str


canonical_method("clarify.respond", params=ClarifyRespondParams, result=PromptResponseResult,
                 doc="Generation-fenced answer to a pending clarify prompt (empty answer = skipped).")


# ── cron / kanban / a2a producers ─────────────────────────────────────────────────────────────


class CronRunParams(Params):
    job_id: str
    request_id: str
    extra_prompt: str | None


class CronSubmitResult(Result):
    session_id: str
    admission_id: str


canonical_method("cron.submit", params=CronRunParams, result=CronSubmitResult,
                 doc="Admit one cron firing into the owning profile's durable FIFO.")


class CronStatusResult(OpenModel):
    """``result`` is the ``run_job`` tuple once terminal; ``recover`` adds the frozen ``job`` and
    answers ``status='missing'`` for a firing that was never admitted."""

    status: str
    result: list[JsonValue] | None
    job_flags: dict[str, JsonValue] | None = None
    job: dict[str, JsonValue] | None = None


canonical_method("cron.recover", params=CronRunParams, result=CronStatusResult,
                 doc="Re-observe a firing after a scheduler restart (same request identity).")


class CronAdmissionParams(Params):
    session_id: str
    admission_id: str


canonical_method("cron.status", params=CronAdmissionParams, result=CronStatusResult,
                 doc="Status and, once terminal, the result of one cron admission.")


class CronCancelResult(Result):
    ok: bool


canonical_method("cron.cancel", params=CronAdmissionParams, result=CronCancelResult,
                 doc="Cancel a queued cron admission or latch cancellation on a started one.")


class KanbanRunParams(Params):
    board: str
    task_id: str
    run_id: int
    claim_lock: str
    db: str | None = None


class KanbanRunResult(Result):
    session_id: str
    receipt: AdmissionStatus


canonical_method("kanban.run", params=KanbanRunParams, result=KanbanRunResult,
                 doc="Native-owner only: admit the dispatcher's current claim on a kanban task.")


class A2aForwardParams(Params):
    agent: str
    tenant: str
    peer: str
    context_id: str
    input_id: str
    text: str


class A2aForwardResult(AdmissionStatus):
    session_id: str
    result: JsonValue | None


canonical_method("a2a.forward", params=A2aForwardParams, result=A2aForwardResult,
                 doc="Forward one A2A input into the conversation its identity tuple names.")


# ── managed workers ───────────────────────────────────────────────────────────────────────────


class WorkerScopeParams(Params):
    """The producer claim every worker verb proves (``gateway/session_worker.py::_SCOPE``)."""

    profile_id: str
    session_id: str
    execution_id: str
    generation: int
    pid: int
    birth: float
    secret: str


class WorkerRegisterParams(WorkerScopeParams):
    kind: str


class WorkerExecution(OpenModel):
    """A ``worker_executions`` row without its adoption digest."""

    execution_id: str
    session_id: str
    kind: str
    owner_epoch: int
    generation: int
    status: str
    last_sequence: int


canonical_method("worker.register", params=WorkerRegisterParams, result=WorkerExecution,
                 doc="Register a compute worker execution on an idle session.")
canonical_method("worker.adopt", params=WorkerScopeParams, result=WorkerExecution,
                 doc="Adopt a registered execution after verifying the live producer claim.")


class WorkerPersistParams(WorkerScopeParams):
    epoch: int
    sequence: int
    operation: str
    payload: dict[str, JsonValue]


class WorkerPersistResult(OpenModel):
    """The operation's durable receipt (``message_id`` for an append, delegation results, …)."""


canonical_method("worker.persist", params=WorkerPersistParams, result=WorkerPersistResult,
                 doc="Sequence-fenced typed persistence write from an adopted worker.")


# ── hosted rooms (canonical-only verbs) ───────────────────────────────────────────────────────


class GroupsAttachmentUploadParams(RoomParams):
    upload_id: str
    kind: str
    name: str
    mime: str
    data_base64: str


class RoomAttachment(OpenModel):
    attachment_id: str
    kind: str
    name: str
    size: int
    mime: str
    sha256: str
    state: str
    created_at: float
    idempotent: bool
    event_id: str | None = None


canonical_method("groups.attachment.upload", params=GroupsAttachmentUploadParams, result=RoomAttachment,
                 doc="Store one attachment for a later groups.send manifest (idempotent per upload_id).")


class GroupsAttachmentDownloadParams(RoomParams):
    event_id: str
    attachment_id: str


class RoomAttachmentBytes(RoomAttachment):
    data_base64: str


canonical_method("groups.attachment.download", params=GroupsAttachmentDownloadParams,
                 result=RoomAttachmentBytes, doc="Read one committed attachment for a live room viewer.")


class GroupsDiscardParams(RoomParams):
    member_id: str
    task_id: str
    execution_generation: int


class GroupsDiscardResult(Result):
    discarded: bool = True
    task: RoomTaskReceipt


canonical_method("groups.discard", params=GroupsDiscardParams, result=GroupsDiscardResult,
                 doc="Discard one indeterminate room task after explicit user confirmation.")


# Shared-name overrides load after the authority-only value shapes above.
from . import canonical_sessions, canonical_projections  # noqa: F401,E402
