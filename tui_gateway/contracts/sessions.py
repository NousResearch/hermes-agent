"""Session lifecycle contracts (``tui_gateway/methods_session.py``): create / resume / activate /
close, the live-session snapshot those share, history + compression + undo, mid-turn corrections,
listing/browsing stored rows, spawn-tree snapshots, event replay and the stateless one-shot LLM call.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import (OpenModel, PendingApproval, ProfileParams, SessionLiveInfo, SessionParams, TranscriptMessage,
                     Usage)
from .connectors_operation import ConnectionRequestPayload
from .registry import method


# ── shared live-session snapshot ──────────────────────────────────────────────────────────────


class OpenRequestEntry(Result):
    """One unanswered server→client request (``server_requests.Request.snapshot``); the reconnecting
    client re-delivers it to its request handlers."""

    id: str
    method: str
    params: dict[str, JsonValue]


class InflightTurn(Result):
    """``session_auto_continue._inflight_snapshot``: the live (or retained failed) turn a reconnecting
    client rebuilds its bubbles from."""

    assistant: str = ""
    streaming: bool = False
    user: str = ""
    display_kind: str | None = None
    display_metadata: dict[str, JsonValue] | None = None
    corrections: list[str] | None = None
    correction_offsets: list[int] | None = None
    error: str | None = None
    status: str | None = None
    recoverable: bool | None = None
    error_surface: dict[str, JsonValue] | None = None


class QueuedPrompt(Result):
    user: str


class TodoState(Result):
    """``tool_progress._normalize_todo_state``: the authoritative todo snapshot."""

    todos: list[dict[str, JsonValue]]
    revision: int


class AutoContinue(Result):
    """A crash-interrupted turn was scheduled to continue right after this resume."""

    attempt: int
    interrupted_at: float


class LiveSessionStatus(WireEnum):
    idle = "idle"
    starting = "starting"
    waiting = "waiting"
    working = "working"
    streaming = "streaming"  # lazy watch window whose child run is active
    resuming = "resuming"  # deferred hydration in flight


class LiveSessionSnapshot(Result):
    """Union of ``server._live_session_payload``, ``methods_session._resume_response`` and the
    live-unpersisted resume path; keys only some paths produce are optional."""

    session_id: str
    message_count: int
    messages: list[TranscriptMessage]
    info: SessionLiveInfo
    stored_session_id: str | None = None
    resumed: str | None = None
    session_key: str | None = None
    messages_omitted: bool | None = None
    hydrating: bool | None = None
    running: bool | None = None
    turn_started_at: float | None = None
    started_at: float | None = None
    status: str | None = None  # a LiveSessionStatus value
    inflight: InflightTurn | None = None
    queued: QueuedPrompt | None = None
    pending_approval: PendingApproval | None = None
    open_requests: list[OpenRequestEntry] | None = None
    # The open connection operation (``tools/connectors/live.current``) as its ``connection.request``
    # payload: the card restores with the server's deadline after a reconnect or restart.
    pending_connection: ConnectionRequestPayload | None = None
    todo_state: TodoState | None = None
    auto_continue: AutoContinue | None = None


# ── session.create ────────────────────────────────────────────────────────────────────────────


class SeedMessage(Params):
    """One create-time transcript row (``session_history._coerce_seed_history``); ``text`` is the
    legacy alias of ``content``; only ``display_kind: "hidden"`` is accepted from the wire. Clients
    forward stored rows verbatim (``_row_id``, ``timestamp``, …) and the coercer drops what it does
    not use, so the row stays open."""

    model_config = Params.model_config | {"extra": "allow"}

    role: str
    content: str | None = None
    text: str | None = None
    display_kind: str | None = None


class SessionCreateParams(ProfileParams):
    cols: int | None = None
    source: str | None = None
    cwd: str | None = None
    messages: list[SeedMessage] | None = None
    parent_session_id: str | None = None
    title: str | None = None
    model: str | None = None
    provider: str | None = None
    reasoning_effort: str | None = None
    fast: bool | None = None  # presence is the contract: omitted inherits, true pins priority, false pins normal
    close_on_disconnect: bool = False
    hidden: bool = False
    room_plumbing: bool = False
    follow_profile_config: bool = False


class SessionCreateResult(Result):
    session_id: str
    stored_session_id: str
    message_count: int
    messages: list[TranscriptMessage]
    info: SessionLiveInfo


method("session.create", params=SessionCreateParams, result=SessionCreateResult,
       doc="Mint a live session (agent builds after the reply); a DB row appears on the first prompt unless seeded.")


# ── session.resume / activate ─────────────────────────────────────────────────────────────────


class SessionResumeParams(SessionParams):
    """``session_id`` is the STORED id (or an exact title); the reply's ``session_id`` is the runtime id."""

    cols: int | None = None
    source: str | None = None
    lazy: bool = False
    defer_history: bool = False
    omit_messages: bool = False
    eager_build: bool = False
    close_on_disconnect: bool = False


class SessionResumeResult(LiveSessionSnapshot):
    pass


method("session.resume", params=SessionResumeParams, result=SessionResumeResult,
       doc="Attach to a stored session: reuse it if live here, else lazy / deferred / cold / eager rebuild.")


class SessionActivateParams(SessionParams):
    cols: int | None = None  # sent by the desktop; the handler keeps the session's current width
    omit_messages: bool = False


class SessionActivateResult(LiveSessionSnapshot):
    pass


method("session.activate", params=SessionActivateParams, result=SessionActivateResult,
       doc="Attach the frontend to a live session without closing the previously focused one.")


# ── listing ───────────────────────────────────────────────────────────────────────────────────


class SessionListParams(ProfileParams):
    title: str | None = None  # exact-title lookup (title as identity); windowless
    limit: int | None = None
    include_hidden: bool = False


class SessionListRow(Result):
    """``methods_session._session_row_summary``; ``resolved_id`` only on a title lookup that followed a
    compression lineage to its tip."""

    id: str
    resolved_id: str | None = None
    title: str = ""
    preview: str = ""
    started_at: float = 0
    message_count: int = 0
    source: str = ""


class SessionListResult(Result):
    sessions: list[SessionListRow]


method("session.list", params=SessionListParams, result=SessionListResult,
       doc="Human-facing stored sessions, most recent first (sub-agent / kanban sources denied).")


class SessionMostRecentParams(ProfileParams):
    pass


class SessionMostRecentResult(Result):
    session_id: str | None
    title: str | None = None
    started_at: float | None = None
    source: str | None = None


method("session.most_recent", params=SessionMostRecentParams, result=SessionMostRecentResult,
       doc="Most recent human-facing session; errors fold into a null session_id.")


class SessionActiveListParams(ProfileParams):
    current_session_id: str | None = None


class SessionActiveItem(Result):
    """``server._session_live_item``."""

    current: bool
    id: str
    last_active: float
    message_count: int
    model: str
    preview: str
    session_key: str
    started_at: float
    status: LiveSessionStatus
    title: str


class SessionActiveListResult(Result):
    sessions: list[SessionActiveItem]


method("session.active_list", params=SessionActiveListParams, result=SessionActiveListResult,
       doc="Live sessions in this process, insertion order (not a DB browser).")


# ── stored-row mutation ───────────────────────────────────────────────────────────────────────


class SessionDeleteParams(SessionParams):
    """``session_id`` is the STORED id."""


class SessionDeleteResult(Result):
    deleted: str


method("session.delete", params=SessionDeleteParams, result=SessionDeleteResult,
       doc="Delete a stored session + transcripts; refused while it is live here.")


class SessionTitleParams(SessionParams):
    title: str | None = None  # absent: read; present: set (non-empty)


class SessionTitleResult(Result):
    title: str
    session_key: str | None = None
    pending: bool | None = None  # True: no row yet, applied on the first turn


method("session.title", params=SessionTitleParams, result=SessionTitleResult,
       doc="Read or set a live session's title; a title set before the row exists is queued.")


class SessionSetHiddenParams(Params):
    """``session_id`` is a live runtime id first, else a stored id / key / title."""

    session_id: str
    hidden: bool = True
    profile: str | None = None


class SessionSetHiddenResult(Result):
    hidden: bool
    session_key: str


method("session.set_hidden", params=SessionSetHiddenParams, result=SessionSetHiddenResult,
       doc="Set/clear hidden (out of the default list, still resumable by its owner) on a session + lineage.")


class SessionWorkspaceMoveParams(ProfileParams):
    session_key: str
    cwd: str


class SessionWorkspaceMoveResult(Result):
    cwd: str
    branch: str | None = None
    git_repo_root: str | None = None


method("session.workspace.move", params=SessionWorkspaceMoveParams, result=SessionWorkspaceMoveResult,
       doc="Re-home a stored session's workspace; git identity is replaced and a live agent follows.")


class SessionCwdSetParams(SessionParams):
    cwd: str


class SessionCwdSetResult(SessionLiveInfo):
    """The refreshed ``session.info`` view (full agent view, or the lazy shape)."""


method("session.cwd.set", params=SessionCwdSetParams, result=SessionCwdSetResult,
       doc="Change a live, idle session's working directory.")


# ── live-session lifecycle ────────────────────────────────────────────────────────────────────


class SessionCloseParams(SessionParams):
    pass


class SessionCloseResult(Result):
    closed: bool  # False when the runtime id was already gone


method("session.close", params=SessionCloseParams, result=SessionCloseResult,
       doc="Tear down a live session (its stored row stays resumable).")


class SessionBranchParams(SessionParams):
    name: str | None = None
    count: int | None = None  # keep only the first N rows of the source history


class SessionBranchResult(Result):
    session_id: str
    stored_session_id: str
    title: str
    parent: str
    message_count: int
    messages: list[TranscriptMessage]
    info: SessionLiveInfo


method("session.branch", params=SessionBranchParams, result=SessionBranchResult,
       doc="Fork a live session into a new stored child that shares the parent's history so far.")


class SessionUndoParams(SessionParams):
    pass


class SessionUndoResult(Result):
    removed: int


method("session.undo", params=SessionUndoParams, result=SessionUndoResult,
       doc="Drop the last user turn (and everything after it) from an idle session.")


class SessionSaveParams(SessionParams):
    pass


class SessionSaveResult(Result):
    """Under turn isolation the compute host's result passes through verbatim."""

    file: str | None = None
    model_config = Result.model_config | {"extra": "allow"}


method("session.save", params=SessionSaveParams, result=SessionSaveResult,
       doc="Export the transcript to ~/.hermes/sessions/saved (classic /save).")


class SessionStatusParams(SessionParams):
    pass


class SessionStatusResult(Result):
    output: str


method("session.status", params=SessionStatusParams, result=SessionStatusResult,
       doc="Rendered /status text for the session.")


class SessionHistoryParams(SessionParams):
    pass


class SessionHistoryResult(Result):
    count: int
    messages: list[TranscriptMessage]


method("session.history", params=SessionHistoryParams, result=SessionHistoryResult,
       doc="The durable display transcript (ancestors included, row ids attached).")


class SessionUsageParams(SessionParams):
    pass


class SessionUsageResult(Usage):
    credits_lines: list[str] | None = None


method("session.usage", params=SessionUsageParams, result=SessionUsageResult,
       doc="Token / context / cost counters for the session (+ Nous credit lines when available).")


class SessionContextBreakdownParams(SessionParams):
    pass


class ContextCategory(Result):
    color: str
    id: str
    label: str
    tokens: int


class ContextFileSource(Result):
    """One row of ``agent.context_file_sources.list_context_file_sources``."""

    label: str
    path: str
    chars: int
    est_tokens: int
    loaded: bool
    status: str


class SessionContextBreakdownResult(Result):
    """``agent.context_breakdown.compute_session_context_breakdown`` (empty categories before the agent builds)
    plus the per-file context manifest (empty until the agent exists)."""

    categories: list[ContextCategory]
    context_max: int
    context_percent: int
    context_used: int
    estimated_total: int
    context_estimated: bool
    context_source: str
    model: str
    context_files: list[ContextFileSource] = []


method("session.context_breakdown", params=SessionContextBreakdownParams, result=SessionContextBreakdownResult,
       doc="Cursor-style split of the context window by category.")


# ── compression ───────────────────────────────────────────────────────────────────────────────


class CompressionSummary(OpenModel):
    """``agent.manual_compression_feedback.summarize_manual_compression``."""

    noop: bool = False
    aborted: bool = False
    refused_would_grow: bool | None = None
    fallback_used: bool | None = None
    headline: str = ""
    token_line: str = ""
    note: str | None = None


class SessionCompressParams(SessionParams):
    focus_topic: str | None = None


class SessionCompressResult(Result):
    """In-process: the before/after summary + replacement transcript. Compute host: its result passes
    through (hence open) with ``turn_isolation``; a lock held elsewhere answers ``compressed: false``."""

    status: str | None = None  # compressed | aborted | pending
    removed: int | None = None
    before_messages: int | None = None
    after_messages: int | None = None
    before_tokens: int | None = None
    after_tokens: int | None = None
    summary: CompressionSummary | None = None
    usage: Usage | None = None
    info: SessionLiveInfo | None = None
    messages: list[TranscriptMessage] | None = None
    compressed: bool | None = None
    lock_held: bool | None = None
    message: str | None = None
    turn_isolation: bool | None = None
    host_ack: dict[str, JsonValue] | None = None
    model_config = Result.model_config | {"extra": "allow"}


method("session.compress", params=SessionCompressParams, result=SessionCompressResult,
       doc="Manual /compress of an idle session, optionally focused on a topic.")


# ── interrupt / steer / redirect ──────────────────────────────────────────────────────────────


class SessionInterruptParams(SessionParams):
    expected_hosted_task_id: str | None = None  # only interrupt if this hosted task is the running one


class InterruptStatus(WireEnum):
    interrupted = "interrupted"
    not_interrupted = "not_interrupted"


class SessionInterruptResult(Result):
    status: InterruptStatus
    interrupted: bool | None = None
    turn_isolation: bool | None = None


method("session.interrupt", params=SessionInterruptParams, result=SessionInterruptResult,
       doc="Stop the running turn (and streaming TTS); retires the crash-recovery marker.")


class CorrectionStatus(WireEnum):
    queued = "queued"
    redirected = "redirected"
    rejected = "rejected"


class SessionCorrectionParams(SessionParams):
    text: str


class SessionCorrectionResult(Result):
    status: CorrectionStatus
    text: str


method("session.steer", params=SessionCorrectionParams, result=SessionCorrectionResult,
       doc="Inject text into the next tool result without interrupting the turn.")
method("session.redirect", params=SessionCorrectionParams, result=SessionCorrectionResult,
       doc="Redirect the active turn (queued for the next turn while the agent is still building).")


# ── spawn trees ───────────────────────────────────────────────────────────────────────────────


class SpawnTreeSaveParams(ProfileParams):
    subagents: list[dict[str, JsonValue]]
    session_id: str | None = None  # stored key; "default" when absent
    started_at: float | None = None
    finished_at: float | None = None
    label: str | None = None


class SpawnTreeSaveResult(Result):
    path: str
    session_id: str


method("spawn_tree.save", params=SpawnTreeSaveParams, result=SpawnTreeSaveResult,
       doc="Persist a finished delegation tree snapshot under the session's spawn-trees dir.")


class SpawnTreeListParams(ProfileParams):
    session_id: str | None = None
    cross_session: bool = False
    limit: int | None = None


class SpawnTreeEntry(OpenModel):
    """Index row (``server._append_spawn_tree_index``) or a legacy file scan."""

    path: str
    session_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    label: str = ""
    count: int = 0


class SpawnTreeListResult(Result):
    entries: list[SpawnTreeEntry]


method("spawn_tree.list", params=SpawnTreeListParams, result=SpawnTreeListResult,
       doc="Saved spawn-tree snapshots, newest first.")


class SpawnTreeLoadParams(ProfileParams):
    path: str


class SpawnTreeLoadResult(Result):
    """The snapshot file as written by ``spawn_tree.save`` (open: the file is the contract)."""

    session_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    label: str | None = None
    subagents: list[dict[str, JsonValue]] = Field(default_factory=list)
    model_config = Result.model_config | {"extra": "allow"}


method("spawn_tree.load", params=SpawnTreeLoadParams, result=SpawnTreeLoadResult,
       doc="Read one saved spawn-tree snapshot (path must be under the spawn-trees root).")


# ── terminal / event replay ───────────────────────────────────────────────────────────────────


class TerminalResizeParams(SessionParams):
    cols: int | None = None


class TerminalResizeResult(Result):
    cols: int


method("terminal.resize", params=TerminalResizeParams, result=TerminalResizeResult,
       doc="Record the client's column width for server-side rendering.")


class SessionEventsSinceParams(SessionParams):
    last_seen: int | None = None


class SessionEventsSinceResult(Result):
    events: list[dict[str, JsonValue]]  # recorded event frames' ``params`` objects
    latest_seq: int
    truncated: bool
    count: int
    epoch: str
    open_requests: list[OpenRequestEntry]


method("session.events.since", params=SessionEventsSinceParams, result=SessionEventsSinceResult,
       doc="Replay events after a seq watermark on WS reconnect; truncated means refetch state.")


class SessionEventsStatsParams(ProfileParams):
    pass


class SessionEventsStatsResult(Result):
    """``event_replay.replay_stats``."""

    sessions: int
    events: int
    bytes: int
    max_per_session: int
    max_bytes_per_session: int
    max_bytes_process: int


method("session.events.stats", params=SessionEventsStatsParams, result=SessionEventsStatsResult,
       doc="Replay-buffer occupancy telemetry (ops/debug).")


# ── typed plugin continuation ────────────────────────────────────────────────────────────────


class DesktopContinuationDestination(Result):
    profile_name: str
    session_id: str


class SessionContinuationReadinessParams(ProfileParams):
    plugin_id: str | None = None
    # Invalid or absent destinations produce a fail-closed readiness result.
    destination: JsonValue = None


class SessionContinuationReadinessResult(Result):
    ready: bool
    status: str
    generation: str | None = None
    native_session_id: str | None = None
    destination: DesktopContinuationDestination | None = None
    busy: bool | None = None


method("session.continuation.readiness", params=SessionContinuationReadinessParams,
       result=SessionContinuationReadinessResult,
       doc="Probe an authorized plugin's exact connected native continuation destination.")


# ── one-shot LLM ──────────────────────────────────────────────────────────────────────────────


class LlmOneshotParams(ProfileParams):
    """Needs a ``template`` or ``instructions`` / ``input``; a live ``session_id`` lends its model."""

    template: str | None = None
    instructions: str | None = None
    input: str | None = None
    variables: dict[str, JsonValue] | None = None
    task: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    session_id: str | None = None


class LlmOneshotResult(Result):
    text: str


method("llm.oneshot", params=LlmOneshotParams, result=LlmOneshotResult,
       doc="Stateless one-shot LLM completion (titles, ideas) on the session's or the task backend.")
