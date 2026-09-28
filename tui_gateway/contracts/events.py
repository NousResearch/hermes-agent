"""Notification payloads: every ``event`` frame the gateway emits (``params.payload``).

One ``Payload`` model per event name, registered with ``event(...)``; ``request.cancel`` lives in
``server_requests.py`` next to the requests it withdraws. Each model's docstring names the Python
emitter it was typed from — that emitter is the source of truth, the hand-written TS in
``apps/shared/src/gateway-events.ts`` was only the map. Fields the emitter always sets are required;
anything conditional is ``X | None = None``.

A handful of payloads stay ``extra="allow"`` on purpose: their closed shape is owned by another
module (the skin engine, the pet store, the goal/loop/heartbeat state files, the free-tier bootstrap
record) or they are watcher signals whose payload is ``{}`` today and may grow. Everything else is
closed, so a drifted emitter fails the suite (``registry.check_payload`` raises under
``HERMES_TEST_ISOLATION``).
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Payload, WireEnum
from .common import MessageReaction, SessionLiveInfo, SubagentStatus, ToolLabel, ToolLabelKind, Usage
from .config_free_tier_control import SessionControlSnapshot
from .registry import event


class OpenPayload(Payload):
    """A payload whose known keys are typed but whose closed set is owned elsewhere."""

    model_config = Payload.model_config | {"extra": "allow"}


# ── gateway lifecycle ─────────────────────────────────────────────────────────────────────────


class SkinPayload(OpenPayload):
    """``tui_gateway/change_watcher.py::resolve_skin`` — the resolved active skin (``HermesSkin``).
    ``{}`` when the skin engine failed to load. Colour maps are token → colour string."""

    name: str = ""
    description: str = ""
    colors: dict[str, str] = Field(default_factory=dict)
    light_colors: dict[str, str] = Field(default_factory=dict)
    dark_colors: dict[str, str] = Field(default_factory=dict)
    branding: dict[str, str] = Field(default_factory=dict)
    banner_logo: str = ""
    banner_hero: str = ""
    tool_prefix: str = ""
    help_header: str = ""


class GatewayReadyPayload(Payload):
    """``tui_gateway/entry.py`` (stdio) / ``tui_gateway/ws.py`` (WebSocket) first frame."""

    skin: SkinPayload
    change_events: bool
    replay_epoch: str
    heartbeat: bool | None = None  # WebSocket transport only


event("gateway.ready", GatewayReadyPayload,
      doc="First frame of a connection: the resolved skin, the change-event capability and the replay epoch.")
event("skin.changed", SkinPayload,
      doc="The active skin moved (name switch or live colour edit); repaint from this palette.")


class SetupReadyPayload(OpenPayload):
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
    """``agent/error_surface.py::_surface`` — advisory {layer, code, retryable} (+ identity, + auth hint,
    + ``resets_at`` epoch seconds when the provider named when its limit lifts)."""

    layer: str
    code: str
    retryable: bool
    provider: str | None = None
    model: str | None = None
    resets_at: float | None = None
    model_config = Payload.model_config | {"extra": "allow"}


class BillingBlock(Payload):
    """``agent/billing_links.py::BillingBlock.to_dict`` (+ ``unverified`` from conversation_loop)."""

    provider: str
    provider_label: str
    model: str
    billing_url: str | None
    is_nous: bool
    message: str
    unverified: bool | None = None


class PersistedTurn(Payload):
    """Committed SQLite row addresses for the agent's current-turn suffix. Missing ids are
    unproven, never negative acknowledgements. ``complete`` permits retiring the whole local
    turn only when the original turn boundary, every row and final body are still accounted
    for; compaction, redirects and partial writes conservatively leave it false. Row ids are
    scoped to the owning profile's store, as in ``SessionMessage.row_id``."""

    row_ids: list[int]
    complete: bool
    user_row_id: int | None = None
    final_assistant_row_id: int | None = None


class MessageCompletePayload(Payload):
    """``prompt_turn._complete_turn_payload`` / ``session_auto_continue._emit_terminal_turn_error`` /
    ``agent_callbacks._mirror_subagent_to_child`` (child watch mirror: ``text`` only) /
    ``compute_host_bridge`` (``text`` + ``status``)."""

    text: str | JsonValue = ""
    usage: Usage | None = None
    status: TurnStatus | None = None
    reasoning: str | None = None
    warning: str | None = None
    response_previewed: bool | None = None
    response_transformed: bool | None = None
    billing: BillingBlock | None = None
    failure_reason: str | None = None
    rendered: str | None = None
    error: str | None = None
    recoverable: bool | None = None
    error_surface: ErrorSurface | None = None
    partial: bool | None = None
    persisted_turn: PersistedTurn | None = None


event("message.complete", MessageCompletePayload, doc="The turn ended: final text, usage and outcome.")


class StatusUpdatePayload(Payload):
    """``server._status_update`` and the direct emitters (goal / loop / heartbeat / process)."""

    kind: str
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
    """``tool_progress._on_tool_start`` (+ ``agent_callbacks._mirror_subagent_to_child`` rows with
    ``preview`` and empty ``args``). ``todos``/``revision`` are NOT set by the emitter; kept optional
    because tool.start rows may pass through connector redaction unchanged."""

    tool_id: str
    name: str
    context: str | None = None
    args: dict[str, JsonValue] | None = None
    args_text: str | None = None
    preview: str | None = None
    labels: list[ToolLabel] | None = None


event("tool.start", ToolStartPayload, doc="A tool call began (stable id + full args).")


class ToolCompletePayload(Payload):
    """``tool_progress._on_tool_complete``; ``todos``/``revision`` merged in for the todo tools."""

    tool_id: str
    name: str
    args: dict[str, JsonValue] | None = None  # mirrored child rows / room relays omit it
    duration_s: float | None = None
    result: JsonValue = None
    summary: str | None = None
    result_text: str | None = None
    inline_diff: str | None = None
    todos: list[JsonValue] | None = None
    revision: int | None = None
    labels: list[ToolLabel] | None = None


event("tool.complete", ToolCompletePayload, doc="A tool call finished: parsed result, summary, optional diff / todo snapshot.")


class ToolGeneratingPayload(Payload):
    """``agent_callbacks`` tool_gen_callback."""

    name: str


event("tool.generating", ToolGeneratingPayload, doc="The model is emitting a tool call's arguments.")


class ToolOutputRiskPayload(Payload):
    """``tool_progress._progress_output_risk``."""

    tool_id: str
    name: str
    risk: str
    findings: list[str]
    redacted: bool


event("tool.output_risk", ToolOutputRiskPayload, doc="Tool output was classified as risky (prompt-injection / secret findings).")


class TodoUpdatedPayload(Payload):
    """``tool_progress._normalize_todo_state`` — full task snapshot."""

    todos: list[JsonValue]
    revision: int


event("todo.updated", TodoUpdatedPayload, doc="Full todo snapshot after a todo tool ran.")


# ── notifications ─────────────────────────────────────────────────────────────────────────────


class NotificationShowPayload(Payload):
    """``agent/credits_tracker.py::AgentNotice`` via notice_callback, and ``server._await_agent_ready``'s
    slow-build notice. ``level``: info | warn | error | success; ``kind``: sticky | ttl | agent."""

    text: str
    level: str
    kind: str
    ttl_ms: int | None = None
    key: str | None = None
    id: str | None = None


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


# SessionLiveInfo is a Result (it is also the ``info`` of create/resume/activate); the registry only
# needs ``model_validate`` and the generator renders one TS type either way.
event("session.info", SessionLiveInfo,  # type: ignore[arg-type]
      doc="Live session settings snapshot (``server._session_info``); also the ``info`` of create/resume/activate.")


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


class SessionReclaimedPayload(Payload):
    """``session_lifecycle._announce_session_reclaimed`` (broadcast)."""

    session_id: str
    stored_session_id: str
    reason: str  # idle_timeout | lru_evict | ws_orphan_reap


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


class SideAgentCompletePayload(Payload):
    """``methods_prompt._spawn_side_agent``: ``{task_id, **extra, text}``; btw adds ``question``."""

    task_id: str
    text: str
    question: str | None = None


event("background.complete", SideAgentCompletePayload, doc="A /background side agent finished.")
event("btw.complete", SideAgentCompletePayload, doc="A /btw side question was answered.")
event("preview.restart.complete", SideAgentCompletePayload, doc="The hidden preview-restart agent finished.")


class PreviewRestartProgressPayload(Payload):
    """``methods_prompt`` restart body + ``agent_callbacks._preview_restart_callbacks``."""

    task_id: str
    text: str
    level: str | None = None


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
    """``tool_progress._progress_moa_reference``."""

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
    """``tools/open_preview_tool.py``."""

    url: str
    label: str = ""


class PreviewClosePayload(Payload):
    """``tools/preview_tool.py`` / ``tools/close_preview_tool.py``; ``url`` '' closes every tab."""

    url: str = ""


event("preview.open", PreviewOpenPayload, doc="Open a URL / file in the desktop preview pane.")
event("preview.close", PreviewClosePayload, doc="Close the preview pane or one tab.")


class LayoutApplyPayload(OpenPayload):
    """``tools/apply_layout_tool.py``."""

    preset: str


class PaneRevealPayload(OpenPayload):
    """``tools/focus_pane_tool.py``."""

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


class BrowserProgressPayload(Payload):
    """``methods_browser`` announce(); ``level``: info | warn | error."""

    message: str
    level: str


event("browser.progress", BrowserProgressPayload, doc="Browser (CDP) connect / install progress line.")


# ── browser controller (gateway/browser_control_broker frames re-enveloped as events) ─────────


class BrowserControllerCommandPayload(Payload):
    """``gateway/browser_control_broker.py`` FRAME_COMMAND params."""

    command_id: str
    action: str
    arguments: dict[str, JsonValue]
    controller_id: str | None = None
    browser_profile_id: str | None = None
    tool_call_id: str | None = None


class BrowserControllerCancelPayload(Payload):
    """``gateway/browser_control_broker.py::_cancel_frame``."""

    command_id: str
    tool_call_id: str | None = None


event("browser.controller.command", BrowserControllerCommandPayload, doc="Dispatch one browser action to the attached controller.")
event("browser.controller.cancel", BrowserControllerCancelPayload, doc="Withdraw a pending controller command.")


# ── voice ─────────────────────────────────────────────────────────────────────────────────────


event("voice.interrupted", None, doc="Barge-in: the spoken interjection interrupted the turn; no payload.")


class VoiceStatusPayload(Payload):
    """``methods_voice._vr_on_status``; states come from the recorder (idle / listening / transcribing …)."""

    state: str


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


class PetChangedPayload(OpenPayload):
    """``change_watcher._pet_changed_payload`` — ``pet.info.meta``-shaped; ``{enabled: false}`` when off."""

    enabled: bool
    slug: str | None = None
    displayName: str | None = None  # noqa: N815 - wire key
    scale: float | None = None
    spritesheetRevision: str | None = None  # noqa: N815 - wire key


class PetGenerateProgressPayload(OpenPayload):
    """``methods_session`` pet.generate: token-only init frame, then one per draft."""

    token: str
    count: int
    index: int | None = None
    dataUri: str | None = None  # noqa: N815 - wire key


class PetHatchProgressPayload(OpenPayload):
    """``methods_session`` pet.hatch ``_on_progress``: ``{event, detail}`` or the parsed row form."""

    event: str
    detail: str | None = None
    state: str | None = None
    done: str | None = None
    total: str | None = None


event("pet.changed", PetChangedPayload, doc="The active pet / its spritesheet changed (watcher).")
event("pet.generate.progress", PetGenerateProgressPayload, doc="Pet base-draft generation progress.")
event("pet.hatch.progress", PetHatchProgressPayload, doc="Pet hatch (row drawing) progress.")


# ── change watcher signals (payload is {} today) ─────────────────────────────────────────────


class ChangeSignalPayload(OpenPayload):
    """``change_watcher._CHANGE_WATCHES`` payload fn — ``{}`` for every watch except pet.changed."""


event("cron.changed", ChangeSignalPayload, doc="cron/jobs.json moved; refetch the cron list.")
event("sessions.changed", ChangeSignalPayload, doc="state.db moved; refetch the session list.")
event("platforms.changed", ChangeSignalPayload, doc="gateway_state.json moved; refetch platform status.")
event("projects.changed", ChangeSignalPayload, doc="projects.db moved; refetch the project list + tree.")
event("pairing.changed", ChangeSignalPayload, doc="Pairing state moved; refetch pairing.")
event("bot_relay.outbox.pending", ChangeSignalPayload, doc="A bot-relay outbox envelope is queued; drain it.")


__all__ = [
    "BillingBlock", "BillingStepUpVerificationPayload", "BrowserControllerCancelPayload",
    "BrowserControllerCommandPayload", "BrowserProgressPayload", "ChangeSignalPayload", "ErrorPayload",
    "ErrorSurface", "GatewayReadyPayload", "LayoutApplyPayload", "MessageCompletePayload",
    "MessageInterimPayload", "MessageReaction", "MessageReactionPayload", "MoaAggregatingPayload",
    "MoaPhasePayload", "MoaProgressPayload", "MoaReferencePayload", "NoticePayload",
    "NotificationClearPayload", "NotificationShowPayload", "OpenPayload", "PaneRevealPayload",
    "PetChangedPayload", "PetGenerateProgressPayload", "PetHatchProgressPayload", "PreviewClosePayload",
    "PreviewOpenPayload", "PreviewRestartProgressPayload", "ReactionPayload", "ResumePhaseStatus",
    "ReviewSummaryPayload", "SessionControlSnapshot", "SessionControlUpdatePayload",
    "SessionReclaimedPayload", "SessionResumeProgressPayload", "SessionTitlePayload", "SessionUsagePayload",
    "SetupReadyPayload", "SideAgentCompletePayload", "SkinPayload", "StatusUpdatePayload",
    "StreamDeltaPayload", "SubagentEventPayload", "SubagentOutputTailEntry", "TerminalClosePayload",
    "TerminalOutputPayload", "TipShowPayload", "TodoUpdatedPayload", "ToolCompletePayload",
    "ToolGeneratingPayload", "ToolLabel", "ToolLabelKind", "ToolOutputRiskPayload", "ToolStartPayload",
    "TurnStatus", "VoiceStatusPayload",
    "VoiceTranscriptPayload", "WakeDetectedPayload",
]
