"""Notification payloads: every ``event`` frame the gateway emits (``params.payload``).

One ``Payload`` model per event name, registered with ``event(...)``; ``request.cancel`` lives in
``server_requests.py`` next to the requests it withdraws. Each model's docstring names the Python
emitter it was typed from. Fields the emitter always sets are required; fields it conditionally omits
use ``X | None = None`` so serialization writes a wire ``null``.
"""

from __future__ import annotations

from .base import JsonValue, Payload, WireEnum
from .common import MessageReaction, SessionLiveInfo, SubagentStatus, Usage
from .config_free_tier_control import SessionControlSnapshot
from .registry import event


# ── gateway lifecycle ─────────────────────────────────────────────────────────────────────────


class SkinPayload(Payload):
    """``change_watcher.resolve_skin`` returns ``{}`` when the skin engine fails."""

    name: str | None = None
    colors: dict[str, str] | None = None
    light_colors: dict[str, str] | None = None
    dark_colors: dict[str, str] | None = None
    branding: dict[str, str] | None = None
    banner_logo: str | None = None
    banner_hero: str | None = None
    tool_prefix: str | None = None
    help_header: str | None = None


class GatewayReadyPayload(Payload):
    """``entry.py`` / ``ws.py`` first frame; stdio omits ``heartbeat``."""

    skin: SkinPayload
    change_events: bool
    replay_epoch: str
    heartbeat: bool | None = None


event("gateway.ready", GatewayReadyPayload,
      doc="First frame of a connection: the resolved skin, the change-event capability and the replay epoch.")
event("skin.changed", SkinPayload,
      doc="The active skin moved (name switch or live colour edit); repaint from this palette.")


class SetupReadyPayload(Payload):
    """``hermes_cli/free_tier_bootstrap.py::SetupRecord.as_payload``."""

    provider_configured: bool
    inference_provider: str
    free_tier: bool
    has_identity: bool
    other_providers: bool
    error: str = ""
    # Present only when the free-tier mint did not happen (``anon_auth.MintFailure.as_payload``).
    error_code: str | None = None
    retryable: bool | None = None
    retry_after: int | None = None
    finished_at: float


event("setup.ready", SetupReadyPayload,
      doc="The free-tier bootstrap finished (broadcast); the desktop's setup gate reads the record.")


class ErrorPayload(Payload):
    """Every ``_emit("error", …)`` site sets exactly ``message``."""

    message: str


event("error", ErrorPayload, doc="A session-level failure outside a turn (agent init, model switch, compression, resume).")


class NoticePayload(Payload):
    """``tui_gateway/model_switch.py`` capability-refresh notice."""

    message: str


event("notice", NoticePayload, doc="Informational one-liner for the session (capabilities refreshed).")


# ── turn stream ───────────────────────────────────────────────────────────────────────────────


event("message.start", None, doc="A turn began streaming; no payload.")


class StreamDeltaPayload(Payload):
    """``prompt_turn._invoke_agent._stream`` (message.delta: ``text`` + optional ``rendered``),
    ``agent_callbacks._agent_cbs`` (reasoning.delta / thinking.delta), ``tool_progress._progress_reasoning``
    (reasoning.available). ``verbose`` rides only when the session's verbose reasoning mode is on."""

    text: str
    rendered: str | None = None
    verbose: bool | None = None


event("message.delta", StreamDeltaPayload, doc="One streamed chunk of the assistant reply.")
event("reasoning.delta", StreamDeltaPayload, doc="One streamed chunk of the model's reasoning.")
event("reasoning.available", StreamDeltaPayload, doc="A completed reasoning block (non-streaming providers).")
event("thinking.delta", StreamDeltaPayload, doc="Legacy thinking-text chunk (thinking_callback).")


class MessageInterimPayload(Payload):
    """``prompt_turn._interim_assistant_cb`` / ``agent_callbacks`` interim_assistant_callback."""

    text: str
    already_streamed: bool


event("message.interim", MessageInterimPayload,
      doc="Interim assistant commentary (text beside tool calls) sealed as its own segment.")


class TurnStatus(WireEnum):
    """``prompt_turn._result_status``."""

    complete = "complete"
    error = "error"
    interrupted = "interrupted"


class ErrorSurface(Payload):
    """``agent/error_surface.py::_surface`` — advisory failure classification."""

    layer: str
    code: str
    retryable: bool
    provider: str | None = None
    model: str | None = None
    auth_kind: str | None = None
    provider_label: str | None = None


class NotificationLevel(WireEnum):
    """Levels emitted by ``agent.credits_tracker.AgentNotice``."""

    info = "info"
    warn = "warn"
    error = "error"
    success = "success"


class NotificationKind(WireEnum):
    """Notice display modes emitted by the credits tracker and startup path."""

    sticky = "sticky"
    ttl = "ttl"
    agent = "agent"


class BillingBlock(Payload):
    """``agent/billing_links.py::BillingBlock.to_dict`` (+ ``unverified`` from conversation_loop)."""

    provider: str
    provider_label: str
    model: str
    billing_url: str | None
    is_nous: bool
    message: str
    unverified: bool | None = None


class MessageCompletePayload(Payload):
    """``prompt_turn`` terminal payload and its error/mirrored variants."""

    text: JsonValue  # Producer preserves non-string model output verbatim.
    usage: Usage | None = None
    status: TurnStatus | None = None
    reasoning: str | None = None
    warning: str | None = None
    response_previewed: bool | None = None
    billing: BillingBlock | None = None
    failure_reason: str | None = None
    rendered: str | None = None
    error: str | None = None
    recoverable: bool | None = None
    error_surface: ErrorSurface | None = None
    partial: bool | None = None


event("message.complete", MessageCompletePayload, doc="The turn ended: final text, usage and outcome.")


class StatusUpdateKind(WireEnum):
    """Kinds written by ``server._status_update`` and direct emitters."""

    status = "status"
    lifecycle = "lifecycle"
    compacting = "compacting"
    compacted = "compacted"
    goal = "goal"
    loop = "loop"
    process = "process"
    heartbeat = "heartbeat"
    ready = "ready"
    compressing = "compressing"
    warn = "warn"


class StatusUpdatePayload(Payload):
    """``server._status_update`` and the direct emitters."""

    kind: StatusUpdateKind
    text: str


event("status.update", StatusUpdatePayload, doc="Transient status line (kind: status, lifecycle, compacting, goal, loop, heartbeat, process, …).")


class SessionUsagePayload(Payload):
    """``server._start_usage_ticker``."""

    usage: Usage


event("session.usage", SessionUsagePayload, doc="Mid-turn usage tick; message.complete carries the authoritative final usage.")


class SessionTitlePayload(Payload):
    """``prompt_turn._invoke_agent`` ``_on_session_title`` hook."""

    session_id: str
    title: str


event("session.title", SessionTitlePayload, doc="Auto-titling renamed the session (``session_id`` is the stored key).")


class ReactionPayload(Payload):
    """``agent_callbacks`` reaction_callback."""

    kind: str


event("reaction", ReactionPayload, doc="Affection reaction detected in the user's message (hearts etc.).")


class ReviewSummaryPayload(Payload):
    """``server`` background_review_callback."""

    text: str


event("review.summary", ReviewSummaryPayload, doc="Background review of the last turn finished.")


# ── tools ─────────────────────────────────────────────────────────────────────────────────────


class ToolStartPayload(Payload):
    """``tool_progress._on_tool_start`` and ``agent_callbacks._mirror_subagent_to_child``."""

    tool_id: str
    name: str
    context: str | None = None
    # WHY arbitrary tool schemas own their argument shapes.
    args: dict[str, JsonValue] | None = None
    args_text: str | None = None
    preview: str | None = None


event("tool.start", ToolStartPayload, doc="A tool call began (stable id + full args).")


class TodoStatus(WireEnum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class TodoItem(Payload):
    """One normalized row from ``tool_progress._normalize_todo_state``."""

    id: str
    content: str
    status: TodoStatus
    parent: str | None = None


class ToolCompletePayload(Payload):
    """``tool_progress._on_tool_complete``; todo tools add a normalized snapshot."""

    tool_id: str
    name: str
    # WHY arbitrary tool schemas own their argument shapes.
    args: dict[str, JsonValue] | None = None
    duration_s: float | None = None
    # The producer writes either a parsed tool JSON value or its original JSON string.
    result: JsonValue | None = None
    summary: str | None = None
    result_text: str | None = None
    inline_diff: str | None = None
    todos: list[TodoItem] | None = None
    revision: int | None = None


event("tool.complete", ToolCompletePayload, doc="A tool call finished: parsed result, summary, optional diff / todo snapshot.")


class ToolGeneratingPayload(Payload):
    """``agent_callbacks`` tool_gen_callback."""

    name: str


event("tool.generating", ToolGeneratingPayload, doc="The model is emitting a tool call's arguments.")


class ToolOutputRiskLevel(WireEnum):
    low = "low"
    high = "high"


class ToolOutputRiskPayload(Payload):
    """``tool_progress._progress_output_risk``."""

    tool_id: str
    name: str
    risk: ToolOutputRiskLevel
    findings: list[str]
    redacted: bool


event("tool.output_risk", ToolOutputRiskPayload, doc="Tool output was classified as risky (prompt-injection / secret findings).")


class TodoUpdatedPayload(Payload):
    """``tool_progress._normalize_todo_state`` — full task snapshot."""

    todos: list[TodoItem]
    revision: int


event("todo.updated", TodoUpdatedPayload, doc="Full todo snapshot after a todo tool ran.")


# ── notifications ─────────────────────────────────────────────────────────────────────────────


class NotificationShowPayload(Payload):
    """``AgentNotice`` and the startup notice both set every display field."""

    text: str
    level: NotificationLevel
    kind: NotificationKind
    ttl_ms: int | None
    key: str | None
    id: str | None


class NotificationClearPayload(Payload):
    key: str


event("notification.show", NotificationShowPayload, doc="Show / replace a keyed out-of-band notice (toast or status bar).")
event("notification.clear", NotificationClearPayload, doc="Withdraw the notice with this key.")


class TipShowPayload(Payload):
    """``tools/tip_tool.py``."""

    selector: str
    text: str
    title: str | None = None
    side: str | None = None


event("tip.show", TipShowPayload, doc="Point at a desktop element with a one-line tip bubble.")


# ── session lifecycle ─────────────────────────────────────────────────────────────────────────


class SessionInfoPayload(Payload, SessionLiveInfo):
    """Payload counterpart to the method-result model; shared fields stay single-sourced."""


event("session.info", SessionInfoPayload,
      doc="Live session settings snapshot (``server._session_info``); method results use SessionLiveInfo.")


class ResumePhaseStatus(WireEnum):
    loading = "loading"
    complete = "complete"
    failed = "failed"


class SessionResumeProgressPayload(Payload):
    """``server._hydrate_resume_history``."""

    phase: str
    status: ResumePhaseStatus
    message_count: int | None = None
    message: str | None = None


event("session.resume_progress", SessionResumeProgressPayload, doc="Deferred resume hydration progress.")


class SessionReclaimReason(WireEnum):
    idle_timeout = "idle_timeout"
    lru_evict = "lru_evict"
    ws_orphan_reap = "ws_orphan_reap"


class SessionReclaimedPayload(Payload):
    """``session_lifecycle._announce_session_reclaimed`` (broadcast)."""

    session_id: str
    stored_session_id: str
    reason: SessionReclaimReason


event("session.reclaimed", SessionReclaimedPayload, doc="The backend reclaimed a live session out from under its clients.")


class SessionControlUpdatePayload(Payload):
    control: SessionControlSnapshot


event("session.control.update", SessionControlUpdatePayload, doc="Persisted goal / loop / heartbeat state changed.")


class BillingStepUpVerificationPayload(Payload):
    """``methods_session`` billing.step_up on_verification."""

    verification_url: str
    user_code: str


event("billing.step_up.verification", BillingStepUpVerificationPayload,
      doc="Device-flow URL + code for the billing scope step-up; the client opens the browser.")


# ── side agents (methods_prompt._spawn_side_agent) ────────────────────────────────────────────


class BackgroundCompletePayload(Payload):
    """``methods_prompt._spawn_side_agent`` background completion."""

    task_id: str
    text: str


class BtwCompletePayload(Payload):
    """``methods_prompt._spawn_side_agent`` BTW completion includes its question."""

    task_id: str
    question: str
    text: str


class PreviewRestartCompletePayload(Payload):
    """``methods_prompt._spawn_side_agent`` preview-restart completion."""

    task_id: str
    text: str


event("background.complete", BackgroundCompletePayload, doc="A /background side agent finished.")
event("btw.complete", BtwCompletePayload, doc="A /btw side question was answered.")
event("preview.restart.complete", PreviewRestartCompletePayload, doc="The hidden preview-restart agent finished.")


class PreviewRestartProgressLevel(WireEnum):
    info = "info"
    error = "error"


class PreviewRestartProgressPayload(Payload):
    """``methods_prompt`` restart body + ``agent_callbacks._preview_restart_callbacks``."""

    task_id: str
    text: str
    level: PreviewRestartProgressLevel | None = None


event("preview.restart.progress", PreviewRestartProgressPayload, doc="Progress line from the preview-restart agent.")


# ── subagents (tool_progress._progress_subagent) ──────────────────────────────────────────────


class SubagentOutputTailEntry(Payload):
    """``tools/delegate_tool_results.py::_extract_output_tail`` row."""

    tool: str
    preview: str
    is_error: bool


class SubagentEventPayload(Payload):
    """``tool_progress._progress_subagent`` — every ``subagent.*`` frame; identity fields are optional
    because older emitters omit them and the TUI spawn tree falls back to flat rendering."""

    goal: str
    task_count: int
    task_index: int
    subagent_id: str | None = None
    parent_id: str | None = None
    child_session_id: str | None = None
    delegation_id: str | None = None
    depth: int | None = None
    model: str | None = None
    tool_count: int | None = None
    toolsets: list[str] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    api_calls: int | None = None
    files_read: list[str] | None = None
    files_written: list[str] | None = None
    output_tail: list[SubagentOutputTailEntry] | None = None
    tool_name: str | None = None
    text: str | None = None
    status: SubagentStatus | None = None
    summary: str | None = None
    duration_seconds: float | None = None
    tool_preview: str | None = None


for _name, _doc in (
    ("subagent.spawn_requested", "delegate_task accepted a child goal (before the child starts)."),
    ("subagent.start", "A delegated child started running."),
    ("subagent.progress", "Batched tool-name progress from a child."),
    ("subagent.thinking", "A child's reasoning chunk."),
    ("subagent.tool", "A child called a tool."),
    ("subagent.complete", "A child finished (status + observability rollup)."),
):
    event(_name, SubagentEventPayload, doc=_doc)


# ── MoA ───────────────────────────────────────────────────────────────────────────────────────


class MoaReferencePayload(Payload):
    """``tool_progress._progress_moa_reference`` omits counters when unavailable."""

    label: str
    text: str
    index: int | None = None
    count: int | None = None


class MoaAggregatingPayload(Payload):
    aggregator: str


class MoaProgressPayload(Payload):
    label: str
    refs_done: int
    refs_total: int


class MoaPhasePayload(Payload):
    """``tool_progress._progress_moa_phase`` adds counters and aggregator conditionally."""

    phase: str
    refs_done: int | None = None
    refs_total: int | None = None
    aggregator: str | None = None


event("moa.reference", MoaReferencePayload, doc="One MoA reference model's output.")
event("moa.aggregating", MoaAggregatingPayload, doc="The MoA aggregator started.")
event("moa.progress", MoaProgressPayload, doc="MoA reference fan-out progress (n/total).")
event("moa.phase", MoaPhasePayload, doc="MoA phase transition (currently only ``aggregator``).")


# ── desktop GUI (tools/desktop_ui emitters) ──────────────────────────────────────────────────


class PreviewOpenPayload(Payload):
    """``tools/open_preview_tool.py`` always sets the normalized URL and label."""

    url: str
    label: str


class PreviewClosePayload(Payload):
    """Preview close producers always set URL; ``""`` closes every tab."""

    url: str


event("preview.open", PreviewOpenPayload, doc="Open a URL / file in the desktop preview pane.")
event("preview.close", PreviewClosePayload, doc="Close the preview pane or one tab.")


class LayoutApplyPayload(Payload):
    """``tools/apply_layout_tool.py`` sends only the normalized preset."""

    preset: str


class PaneRevealPayload(Payload):
    """``tools/focus_pane_tool.py`` sends only the selected pane."""

    pane: str


event("layout.apply", LayoutApplyPayload, doc="Apply a named desktop layout preset.")
event("pane.reveal", PaneRevealPayload, doc="Focus / reveal a named desktop pane.")


class MessageReactionPayload(Payload):
    """``tools/react_to_message_tool.py``."""

    row_id: int
    reactions: list[MessageReaction]
    role: str


event("message.reaction", MessageReactionPayload, doc="The agent reacted to a message; paint it live.")


class TerminalOutputPayload(Payload):
    """``session_notifications`` process_registry.on_output."""

    process_id: str
    chunk: str


class TerminalClosePayload(Payload):
    process_id: str


event("agent.terminal.output", TerminalOutputPayload, doc="Output chunk from an agent-owned background process.")
event("terminal.close", TerminalClosePayload, doc="An agent-owned background process closed.")


class BrowserProgressLevel(WireEnum):
    """Levels passed to ``methods_browser.announce``."""

    info = "info"
    error = "error"


class BrowserProgressPayload(Payload):
    """``methods_browser.announce`` emits connection progress."""

    message: str
    level: BrowserProgressLevel


event("browser.progress", BrowserProgressPayload, doc="Browser (CDP) connect / install progress line.")


# ── browser controller (gateway/browser_control_broker frames re-enveloped as events) ─────────


class BrowserControllerCommandPayload(Payload):
    """``browser_control_broker.dispatch`` always includes controller routing keys."""

    command_id: str
    action: str
    arguments: dict[str, JsonValue]
    controller_id: str | None
    browser_profile_id: str | None
    tool_call_id: str | None


class BrowserControllerCancelPayload(Payload):
    """``browser_control_broker._cancel_frame`` includes ``tool_call_id``, possibly null."""

    command_id: str
    tool_call_id: str | None


event("browser.controller.command", BrowserControllerCommandPayload, doc="Dispatch one browser action to the attached controller.")
event("browser.controller.cancel", BrowserControllerCancelPayload, doc="Withdraw a pending controller command.")


# ── voice ─────────────────────────────────────────────────────────────────────────────────────


event("voice.interrupted", None, doc="Barge-in: the spoken interjection interrupted the turn; no payload.")


class VoiceStatus(WireEnum):
    idle = "idle"
    listening = "listening"
    transcribing = "transcribing"


class VoiceStatusPayload(Payload):
    """``hermes_cli.voice`` invokes ``methods_voice._vr_on_status`` with these states."""

    state: VoiceStatus


class VoiceTranscriptPayload(Payload):
    """``methods_voice._vr_transcript`` / ``_deliver_fd_transcript`` / typed stop phrase in methods_prompt."""

    text: str | None = None
    stop_phrase: bool | None = None
    typed: bool | None = None
    no_speech_limit: bool | None = None


class WakeDetectedPayload(Payload):
    """``methods_voice`` wake detector ``_on_detect``."""

    phrase: str
    profile: str | None = None
    start_new_session: bool


event("voice.status", VoiceStatusPayload, doc="Voice recorder state changed.")
event("voice.transcript", VoiceTranscriptPayload, doc="A voice capture produced text (or a stop phrase / silence limit).")
event("wake.detected", WakeDetectedPayload, doc="A wake phrase fired.")


# ── pets ──────────────────────────────────────────────────────────────────────────────────────


class PetChangedPayload(Payload):
    """``change_watcher._pet_changed_payload`` omits metadata when pets are disabled."""

    enabled: bool
    slug: str | None = None
    displayName: str | None = None  # noqa: N815 - wire key
    scale: float | None = None
    spritesheetRevision: str | None = None  # noqa: N815 - wire key


class PetGenerateProgressPayload(Payload):
    """``methods_session`` emits an init row, then draft rows without ``None`` values."""

    token: str
    count: int
    index: int | None = None
    dataUri: str | None = None  # noqa: N815 - wire key


class PetHatchProgressPayload(Payload):
    """``methods_session`` emits either detail or parsed row fields."""

    event: str
    detail: str | None = None
    state: str | None = None
    done: str | None = None
    total: str | None = None


event("pet.changed", PetChangedPayload, doc="The active pet / its spritesheet changed (watcher).")
event("pet.generate.progress", PetGenerateProgressPayload, doc="Pet base-draft generation progress.")
event("pet.hatch.progress", PetHatchProgressPayload, doc="Pet hatch (row drawing) progress.")


# ── change watcher signals (payload is {} today) ─────────────────────────────────────────────


class ChangeSignalPayload(Payload):
    """``change_watcher._CHANGE_WATCHES`` sends ``{}`` for these signals."""


event("cron.changed", ChangeSignalPayload, doc="cron/jobs.json moved; refetch the cron list.")
event("sessions.changed", ChangeSignalPayload, doc="state.db moved; refetch the session list.")
event("platforms.changed", ChangeSignalPayload, doc="gateway_state.json moved; refetch platform status.")
event("pairing.changed", ChangeSignalPayload, doc="Pairing state moved; refetch pairing.")
event("bot_relay.outbox.pending", ChangeSignalPayload, doc="A bot-relay outbox envelope is queued; drain it.")


__all__ = [
    "BackgroundCompletePayload", "BillingBlock", "BillingStepUpVerificationPayload", "BrowserControllerCancelPayload",
    "BrowserControllerCommandPayload", "BrowserProgressLevel", "BrowserProgressPayload", "BtwCompletePayload",
    "ChangeSignalPayload", "ErrorPayload", "ErrorSurface", "GatewayReadyPayload", "LayoutApplyPayload",
    "MessageCompletePayload", "MessageInterimPayload", "MessageReaction", "MessageReactionPayload",
    "MoaAggregatingPayload", "MoaPhasePayload", "MoaProgressPayload", "MoaReferencePayload", "NoticePayload",
    "NotificationClearPayload", "NotificationKind", "NotificationLevel", "NotificationShowPayload",
    "PaneRevealPayload", "PetChangedPayload", "PetGenerateProgressPayload", "PetHatchProgressPayload",
    "PreviewClosePayload", "PreviewOpenPayload", "PreviewRestartCompletePayload", "PreviewRestartProgressLevel",
    "PreviewRestartProgressPayload", "ReactionPayload", "ResumePhaseStatus", "ReviewSummaryPayload", "SessionControlSnapshot",
    "SessionControlUpdatePayload", "SessionInfoPayload", "SessionReclaimReason", "SessionReclaimedPayload", "SessionResumeProgressPayload",
    "SessionTitlePayload", "SessionUsagePayload", "SetupReadyPayload", "SkinPayload", "StatusUpdatePayload", "StreamDeltaPayload",
    "SubagentEventPayload", "SubagentOutputTailEntry", "TerminalClosePayload", "TerminalOutputPayload",
    "StatusUpdateKind", "TipShowPayload", "TodoItem", "TodoStatus", "TodoUpdatedPayload", "ToolCompletePayload",
    "ToolGeneratingPayload", "ToolOutputRiskLevel", "ToolOutputRiskPayload", "ToolStartPayload", "TurnStatus",
    "VoiceStatus", "VoiceStatusPayload", "VoiceTranscriptPayload", "WakeDetectedPayload",
]
