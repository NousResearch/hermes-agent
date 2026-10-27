"""Shared method names whose canonical session wire differs from standalone serve.

Reuse upstream authority-only value shapes; these overrides do not widen METHODS.
"""
from typing import Literal

from pydantic import RootModel

from .base import JsonValue, Params, Result
from .common import ProfileParams, TranscriptMessage
from .connectors_operation import ConnectionRequestPayload
from .sessions import OpenRequestEntry, TodoState
from .canonical import AdmissionStatus, CanonicalSessionRef, PromptResponseResult, SessionInventoryFields
from .registry import canonical_method


class CanonicalSessionParams(ProfileParams):
    session_id: str


class CanonicalCreateParams(ProfileParams):
    request_id: str | None = None
    source: Literal['cli', 'tui', 'gui', 'acp', 'tool', 'oneshot'] = 'cli'
    cwd: str | None = None
    # The sessions-row ``cwd`` when it differs from the execution ``cwd`` (the classic CLI records
    # main's stamp: the launch dir, ``""`` = none); ``cwd`` stays where the tools run.
    row_cwd: str | None = None
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    toolsets: list[str] | None = None
    reasoning: str | None = None
    max_turns: int | str | None = None
    ignore_rules: bool = False
    yolo: bool = False
    safe_mode: bool = False
    ignore_user_config: bool = False
    skills: list[str] | None = None
    checkpoints: bool = False
    accept_hooks: bool = False
    pass_session_id: bool = False
    editor: dict[str, JsonValue] | None = None
    title: str | None = None
    hidden: bool = False
    follow_profile_config: bool | None = None
    service_tier: str | None = None
    cwd_explicit: bool | None = None
    # Wall-clock budget per run in seconds (the classic CLI's `--run-budget`).
    run_budget: float | None = None
    # The classic CLI's `-v`: the session agent's status lines are not quieted.
    verbose: bool = False
    # A shell-launched client's allowlisted environment (HERMES_* behaviour values, toolchain PATH/venv),
    # applied to this session only; never echoed in ``launch_request``.
    client_env: dict[str, str] | None = None


class CanonicalResumeParams(ProfileParams):
    session_id: str = ''
    title: str | None = None
    # Bare `-c` / `--resume latest`: the owner resolves the most recent session of this surface
    # family (``workspace`` = the caller's git root / cwd, tried first).
    latest: Literal['cli', 'tui'] | None = None
    workspace: str | None = None
    source: str | None = None
    editor: dict[str, JsonValue] | None = None
    # Re-supplies a launch-only key a restart revoked; never durable, must match the launch's.
    api_key: str | None = None
    defer_history: bool = False
    omit_messages: bool = False
    cols: int | None = None
    # A renderer (the Ink TUI) asks for main's display projection of the same rows
    # (``tui_gateway.session_history._history_to_messages``) as ``display_messages``.
    display_projection: bool = False


class CanonicalSessionHandle(Result):
    ref: CanonicalSessionRef
    instance_id: str
    authority_epoch: int
    revision: int
    execution_generation: int
    execution_state: Literal['idle', 'running', 'waiting', 'unknown', 'terminal']


class CanonicalPendingAdmission(AdmissionStatus):
    input_id: str
    text: str
    # The queued-row verbs the owner serves for this row: ``steer`` (``prompt.steer_queued``) and
    # ``replace`` (``prompt.replace_queued``). Absent from an owner that predates them.
    controls: list[Literal['steer', 'replace']] | None = None
    # A queued admission in transfer to the running turn (``prompt.steer_queued``): listed ``started``
    # until that turn settles it, but it returns to ``queued`` if its text never reached the turn, so a
    # client keeps it at its queued rank. Absent from an owner that predates it.
    steering: bool | None = None


class CanonicalPendingPrompt(Result):
    kind: Literal['approval', 'clarify']
    prompt_id: str
    execution_generation: int
    choices: list[str]
    command: str | None = None
    description: str | None = None
    edit: dict[str, JsonValue] | None = None
    question: str | None = None
    multi_select: bool | None = None
    # Local viewer turns: the seconds the gateway waits before it gives up on this prompt.
    timeout_s: int | None = None


class CanonicalSnapshotInfo(SessionInventoryFields):
    desktop_protocol: Literal['hermes-gateway-v1']
    source: str | None = None
    model: str | None = None
    lazy: bool | None = None
    profile_id: str | None = None
    profile_name: str | None = None
    cwd: str | None = None
    edit_approval_policy: Literal['ask', 'workspace_session', 'session'] | None = None
    launch_request: dict[str, JsonValue] | None = None
    reasoning_effort: str | None = None
    reasoning_effort_wire: str | None = None
    service_tier: str | None = None
    fast: bool | None = None
    title: str | None = None
    started_at: float | None = None
    last_active_at: float | None = None
    yolo: bool | None = None
    api_mode: str | None = None
    tool_count: int | None = None
    credential_hint: str | None = None
    # ``session.info``'s session-start notice: ``session.create`` returns the same projection.
    config_warning: str | None = None


class CanonicalSnapshot(Result):
    session_id: str
    stored_session_id: str
    messages: list[TranscriptMessage]
    # Present only when ``display_projection`` was asked: the rows as main's TUI rendered them.
    display_messages: list[TranscriptMessage] | None = None
    message_count: int
    running: bool
    authority_epoch: int
    replay_epoch: str
    last_sequence: int
    subscription_id: str
    revision: int
    execution_generation: int
    pending: list[CanonicalPendingAdmission]
    prompts: list[CanonicalPendingPrompt]
    info: CanonicalSnapshotInfo
    # The open ``manage_connections`` card, so a resume restores it while the tool waits.
    pending_connection: ConnectionRequestPayload | None = None
    # The session's todo snapshot (its transcript's newest todo result); null when it has none.
    todo_state: TodoState | None = None
    # The owner's server→client requests still unanswered (``session_server_requests.open_requests``).
    open_requests: list[OpenRequestEntry] | None = None


class CanonicalListParams(ProfileParams):
    scope: Literal['live', 'stored'] = 'live'
    limit: int = 200
    title: str | None = None
    include_hidden: bool = False


class CanonicalSessionListRow(Result):
    session_id: str
    id: str
    title: str
    source: str | None
    started_at: float | None
    message_count: int
    running: bool
    model: str | None = None
    resolved_id: str | None = None
    root_title: str | None = None


class CanonicalListResult(Result):
    sessions: list[CanonicalSessionListRow]
    scope: Literal['stored', 'live']


class CanonicalAttachment(Params):
    path: str
    mime: str


class CanonicalTurnAuthor(Params):
    id: str | None = None
    name: str | None = None
    is_bot: bool = False
    origin: str | None = None


class CanonicalModelOverride(Params):
    model: str
    provider: str | None = None
    # ``/model B --once --reasoning X``: this admission alone runs level X. The owner refuses a level
    # ``parse_reasoning_effort`` cannot read (``gateway/session_cli_turn.py``).
    reasoning: str | None = None


class CanonicalSubmitParams(CanonicalSessionParams):
    text: str
    submission_id: str | None = None
    input_id: str | None = None
    queued: bool = False
    attachments: list[CanonicalAttachment] | None = None
    finite: bool = False
    unattended: bool = False
    surface: str | None = None
    voice_context: str | None = None
    interrupted: bool = False
    voice_turn: bool = False
    # Desktop composer presentation, committed per admission (``display_v1``): a hidden widget
    # intent persists ``display_kind=hidden`` (never a user bubble); a large paste's preview is
    # titler input only, never model input.
    display_kind: Literal['hidden'] | None = None
    title_preview: str | None = None
    # Classic CLI per-turn inputs, committed per admission (``cli_turn_v1``): who wrote the turn
    # (memory attribution only), the model-facing text when it differs from the persisted ``text``
    # (voice prefix, model-switch / skills-reload notes) and a one-turn ``/model --once`` override.
    turn_author: CanonicalTurnAuthor | None = None
    api_text: str | None = None
    model_override: CanonicalModelOverride | None = None
    # The submitter answers the clarify tool's whole batch in one ``clarify`` server request (main's
    # batch panel); without it the turn's viewers get one ``clarify.request`` card per question.
    clarify_batch: bool = False


class CanonicalVoiceStopped(Result):
    """A bare stop phrase typed while the session's voice chat is on: the chat ended, nothing was
    admitted (TUI-26)."""

    voice_stopped: Literal[True]


class CanonicalSubmitResult(RootModel):
    """``prompt.submit``: the admission receipt, or ``{voice_stopped: true}``."""

    root: AdmissionStatus | CanonicalVoiceStopped


class CanonicalSteerQueuedParams(CanonicalSessionParams):
    admission_id: str


class CanonicalReplaceQueuedParams(CanonicalSessionParams):
    """The edited prompt: admitted as a NEW admission (``submission_id``) at the FIFO tail."""

    admission_id: str
    submission_id: str
    text: str
    attachments: list[CanonicalAttachment] | None = None
    display_kind: Literal['hidden'] | None = None
    title_preview: str | None = None


class CanonicalControlParams(CanonicalSessionParams):
    execution_generation: int


class CanonicalCorrectionParams(CanonicalControlParams):
    text: str


class CanonicalCorrectionAck(Result):
    status: Literal['queued', 'redirected', 'rejected']
    text: str
    execution_generation: int
    authority_epoch: int
    # Only on a 'rejected' no agent ever saw (none was live): safe to admit as the next prompt. A plain
    # 'rejected' may follow delivery (a Codex steer whose ACK timed out), so clients never re-send it.
    delivered: Literal[False] | None = None


class CanonicalCorrectionResult(RootModel):
    """``session.steer`` / ``session.redirect``: the correction verdict, or ``{voice_stopped: true}``
    when the busy-time text was a bare stop phrase that ended the LOCAL voice chat."""

    root: CanonicalCorrectionAck | CanonicalVoiceStopped


class CanonicalApprovalParams(CanonicalControlParams):
    prompt_id: str
    choice: str


class CanonicalEventsParams(CanonicalSessionParams):
    last_sequence: int = 0
    last_seen: int = 0
    replay_epoch: str | None = None


class CanonicalReplayEvent(Result):
    type: str
    session_id: str
    payload: dict[str, JsonValue]
    replay_epoch: str
    seq: int
    authority_epoch: int | None = None
    execution_generation: int | None = None
    admission_id: str | None = None


class CanonicalEventsResult(Result):
    events: list[CanonicalReplayEvent]
    latest_seq: int
    last_sequence: int
    epoch: str
    replay_epoch: str
    truncated: bool
    snapshot_required: bool
    count: int
    open_requests: list[OpenRequestEntry] | None = None


for _name, _params, _result in (
    ('session.create', CanonicalCreateParams, CanonicalSnapshot),
    ('session.resume', CanonicalResumeParams, CanonicalSnapshot),
    ('session.list', CanonicalListParams, CanonicalListResult),
    ('session.interrupt', CanonicalControlParams, CanonicalSessionHandle),
    ('session.steer', CanonicalCorrectionParams, CanonicalCorrectionResult),
    ('session.redirect', CanonicalCorrectionParams, CanonicalCorrectionResult),
    ('session.events.since', CanonicalEventsParams, CanonicalEventsResult),
    ('prompt.submit', CanonicalSubmitParams, CanonicalSubmitResult),
    ('approval.respond', CanonicalApprovalParams, PromptResponseResult),
):
    canonical_method(_name, params=_params, result=_result)

canonical_method('prompt.steer_queued', params=CanonicalSteerQueuedParams, result=AdmissionStatus,
                 doc="Hand a still-queued admission's text to the running turn's agent as a steer. "
                     "The admission reads 'started' with that turn (pending snapshots mark it "
                     "'steering') and never runs on its own; it settles terminal 'steered' when the "
                     "turn completes with its text, else 'interrupted' (its completion frame says "
                     "'cancelled', as an interrupted turn's does); a text the turn never consumed "
                     "goes back to its queue place. "
                     "Refusals: admission_started, session_idle, steer_unavailable (no agent to take "
                     "it: the admission stays queued), unsupported_control, not_found.")
canonical_method('prompt.replace_queued', params=CanonicalReplaceQueuedParams, result=AdmissionStatus,
                 doc="Cancel a still-queued admission and admit the edited prompt as a new admission "
                     "at the back of the FIFO, in one write bound to the replaced admission. "
                     "Refusals: admission_started, admission_conflict (the submission_id replaced "
                     "another admission), queue_full, unsupported_control, not_found.")
