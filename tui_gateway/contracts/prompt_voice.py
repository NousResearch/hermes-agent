"""Prompt submission, attachments, side agents, approvals, batch-clarify locks (``methods_prompt.py``)
and voice / wake-word control (``methods_voice.py``).

``profile`` rides on every method here: the desktop routes session-scoped calls through
``requestForOwnedSession`` / ``requestForBot`` / ``requestGatewayForProfile``, which stamp it.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import OpenModel, PendingApproval, SessionParams
from .registry import method

# ── prompt.submit ─────────────────────────────────────────────────────────────────────────────


class ClientSurface(WireEnum):
    """Where the user typed this turn; anything else is treated as the plain app window."""

    hud = "hud"
    voice_live = "voice-live"


class PromptSubmitParams(SessionParams):
    """``text`` is normally a string; the relay / hosted paths may hand a structured (parts list)
    payload, and the busy path renders it. Truncation (rewind / edit / regenerate) needs explicit
    consent: ``confirm_truncate`` plus one durable target (``truncate_before_row_id`` preferred,
    ``truncate_before_message_id``, or the legacy ``truncate_before_user_ordinal``)."""

    text: JsonValue = ""
    display_kind: str | None = None  # only "hidden" is honoured; anything else renders as a user row
    interrupted: bool | None = None  # client-side barge-in: the turn's model message carries the note
    queued: bool | None = None  # client queue drain — the busy path must hold it, never redirect/steer
    # Optional, untrusted JSON evidence. Malformed evidence is discarded by Desktop admission;
    # it must not invalidate an otherwise valid submitted message.
    input_provenance: JsonValue | None = None
    surface: str | None = None  # a ClientSurface value; unknown values clear the surface
    voice_context: str | None = None  # recent spoken transcript, model input only (voice-live)
    # Desktop-generated large-paste preview (first ~1000 chars); TITLE input only, never the model turn.
    title_preview: str | None = None
    truncate_before_user_ordinal: int | None = None
    truncate_before_row_id: int | None = None
    truncate_before_message_id: str | None = None
    confirm_truncate: bool | None = None
    confirm_empty_truncate: bool | None = None
    rebind_survivor_row_ids: list[int] | None = None
    # In-process only: injected by the hosted-room / bot-relay handlers, never accepted from a
    # client (a client dict for ``_turn_author`` answers 4124). Excluded from the rendered wire.
    hosted_task: JsonValue | None = Field(default=None, exclude=True, alias="_hosted_task")
    turn_author: JsonValue | None = Field(default=None, exclude=True, alias="_turn_author")
    hosted_terminal_callback: JsonValue | None = Field(
        default=None, exclude=True, alias="_hosted_terminal_callback")


class PromptSubmitStatus(WireEnum):
    streaming = "streaming"
    queued = "queued"
    steered = "steered"
    redirected = "redirected"


class PromptSubmitResult(Result):
    """``status`` is absent only on the typed-stop-phrase reply (``voice_stopped``). After a
    truncation the survivor row ids let the client rebind its cached ``rowId``s (``None`` map
    entries: drop the cached id). ``turn_isolation`` marks a compute-host dispatch."""

    status: PromptSubmitStatus | None = None
    voice_stopped: bool | None = None
    # The row written for THIS accepted input, captured before the worker can consume it.
    # Absent on queued/steered/redirected inputs and whenever persistence is not yet proven.
    user_row_id: int | None = None
    survivor_user_row_ids: list[int | None] | None = None
    survivor_row_id_map: dict[str, int | None] | None = None
    turn_isolation: bool | None = None


method("prompt.submit", params=PromptSubmitParams, result=PromptSubmitResult,
       doc="Send a user turn to a live session; busy sessions queue / steer / redirect instead of refusing.")


# ── attachments ───────────────────────────────────────────────────────────────────────────────


class ImageMeta(Result):
    """``tui_gateway/server.py::_image_meta`` — dimensions when PIL can open the file."""

    name: str | None = None
    width: int | None = None
    height: int | None = None
    token_estimate: int | None = None


class AttachedImageResult(ImageMeta):
    """``methods_prompt.py::_attached_image_result``: the image is queued for the next turn."""

    attached: bool
    path: str | None = None
    count: int | None = None
    remainder: str | None = None
    text: str | None = None
    bytes: int | None = None
    message: str | None = None  # clipboard.paste: why nothing was attached


class ClipboardPasteParams(SessionParams):
    pass


method("clipboard.paste", params=ClipboardPasteParams, result=AttachedImageResult,
       doc="Save the host clipboard image into the session and queue it for the next turn.")


class ImageAttachParams(SessionParams):
    path: str  # a host path, optionally followed by remainder text (drop syntax accepted)


method("image.attach", params=ImageAttachParams, result=AttachedImageResult,
       doc="Queue a gateway-visible image file for the next turn.")


class ImageAttachBytesParams(SessionParams):
    """``content_base64`` (or the ``data`` alias) carries the bytes; ``filename`` / ``ext`` only hint
    the extension — magic bytes decide."""

    content_base64: str | None = None
    data: str | None = None
    filename: str | None = None
    ext: str | None = None


method("image.attach_bytes", params=ImageAttachBytesParams, result=AttachedImageResult,
       doc="Queue an image uploaded as base64 (remote client); reply mirrors image.attach.")


class PdfAttachParams(SessionParams):
    """Host ``path`` or base64 ``content_base64`` / ``data``; ``first_page`` / ``last_page`` bound the
    render (per-call page cap enforced server-side)."""

    path: str | None = None
    content_base64: str | None = None
    data: str | None = None
    filename: str | None = None
    first_page: int | None = None
    last_page: int | None = None


class PdfPage(ImageMeta):
    path: str
    page: int


class PdfAttachResult(Result):
    attached: bool
    filename: str
    pages_attached: int
    pages: list[PdfPage]
    count: int
    text: str


method("pdf.attach", params=PdfAttachParams, result=PdfAttachResult,
       doc="Render a PDF's pages to PNG and queue them as images for the next turn.")


class FileAttachParams(SessionParams):
    """``path`` when the file is gateway-visible, else ``data_url`` carries the bytes; ``name`` labels
    an uploaded file."""

    path: str | None = None
    data_url: str | None = None
    name: str | None = None


class FileAttachResult(Result):
    attached: bool
    name: str
    path: str
    ref_path: str
    ref_text: str
    uploaded: bool


method("file.attach", params=FileAttachParams, result=FileAttachResult,
       doc="Stage a non-image file into the session workspace and hand back its @file: ref.")


class ImageDetachParams(SessionParams):
    path: str


class ImageDetachResult(Result):
    detached: bool
    count: int


method("image.detach", params=ImageDetachParams, result=ImageDetachResult,
       doc="Drop a queued image before the turn is sent.")


class InputDetectDropParams(SessionParams):
    text: str | None = None


class InputDetectDropResult(ImageMeta):
    """``matched: false`` alone when the text is not a drop; an image drop is queued immediately
    (``is_image`` + ``count``), a file drop only yields the ``text`` to insert."""

    matched: bool
    is_image: bool | None = None
    path: str | None = None
    count: int | None = None
    text: str | None = None


method("input.detect_drop", params=InputDetectDropParams, result=InputDetectDropResult,
       doc="Recognise a terminal file drop pasted into the composer and turn it into an attachment.")


# ── side agents ───────────────────────────────────────────────────────────────────────────────


class SideAgentParams(SessionParams):
    text: str


class TaskIdResult(Result):
    """The side agent runs detached; its answer lands on the parent session as an event carrying
    this ``task_id``."""

    task_id: str


method("prompt.background", params=SideAgentParams, result=TaskIdResult,
       doc="Run a task on a fresh agent in the background; the answer arrives as background.complete.")
method("prompt.btw", params=SideAgentParams, result=TaskIdResult,
       doc="Side question over a snapshot of the live conversation; the answer arrives as btw.complete.")


class PreviewRestartParams(SessionParams):
    url: str
    cwd: str | None = None
    context: str | None = None  # the preview pane's console output


method("preview.restart", params=PreviewRestartParams, result=TaskIdResult,
       doc="Spawn a hidden agent that brings the desktop preview's dev server back up.")


# ── batch clarify locks / proxied request answers ─────────────────────────────────────────────


class ClarifyLockParams(Params):
    request_id: str
    question_id: str
    answer: JsonValue = ""  # non-string answers are JSON-encoded server-side
    profile: str | None = None


class ClarifyLockStatus(WireEnum):
    ok = "ok"
    expired = "expired"


class ClarifyLockResult(Result):
    """``remaining`` lists the qids still unanswered; the lock that empties it resolves the request.
    ``expired``: the wait already ended (timeout / cancel) — not an error."""

    status: ClarifyLockStatus
    remaining: list[str] | None = None


method("clarify.lock", params=ClarifyLockParams, result=ClarifyLockResult,
       doc="Lock one answer of a batch clarify request (editable until every question is locked).")


class RequestAnswerParams(Params):
    id: str  # the open server→client request id
    result: dict[str, JsonValue]
    profile: str | None = None


class RequestAnswerResult(Result):
    status: ClarifyLockStatus


method("request.answer", params=RequestAnswerParams, result=RequestAnswerResult,
       doc="Answer an open server→client request from a client that never received the frame.")


# ── approvals ─────────────────────────────────────────────────────────────────────────────────


class ApprovalPendingParams(SessionParams):
    pass


class ApprovalPendingResult(Result):
    approvals: list[PendingApproval]


method("approval.pending", params=ApprovalPendingParams, result=ApprovalPendingResult,
       doc="Replay the approvals still waiting on this session (reconnect / polling).")


class ApprovalReceivedParams(SessionParams):
    request_id: str


class ApprovalReceivedResult(Result):
    acknowledged: bool


method("approval.received", params=ApprovalReceivedParams, result=ApprovalReceivedResult,
       doc="Tell the backend the card is on screen, so its timeout clock starts.")


class ApprovalRespondParams(SessionParams):
    """``choice`` is one of the offered ``approval`` choices (once / session / always / deny); ``all``
    resolves every pending approval, ``request_id`` a specific one, neither the oldest."""

    choice: str | None = None  # default "deny"
    all: bool | None = None
    request_id: str | None = None


class ApprovalRespondResult(Result):
    resolved: int  # how many pending approvals this decision unblocked


method("approval.respond", params=ApprovalRespondParams, result=ApprovalRespondResult,
       doc="Deliver the user's decision on a dangerous command (falls back to durable identity on a stale sid).")


# ── voice ─────────────────────────────────────────────────────────────────────────────────────


class VoiceToggleAction(WireEnum):
    status = "status"
    on = "on"
    off = "off"
    tts = "tts"


class VoiceToggleParams(Params):
    action: VoiceToggleAction = VoiceToggleAction.status
    profile: str | None = None


class VoiceToggleResult(Result):
    """``methods_voice.py::_voice_status_payload`` (+ the requirements probe on ``status``, the spoken
    stop hint on ``on``)."""

    enabled: bool
    record_key: str
    tts: bool
    stop_hint: str | None = None
    available: bool | None = None
    audio_available: bool | None = None
    stt_available: bool | None = None
    details: str | None = None


method("voice.toggle", params=VoiceToggleParams, result=VoiceToggleResult,
       doc="/voice parity: report, flip voice mode on/off, or toggle speech output.")


class VoiceRecordAction(WireEnum):
    start = "start"
    stop = "stop"


class VoiceRecordParams(Params):
    action: VoiceRecordAction = VoiceRecordAction.start
    session_id: str | None = None  # where voice.transcript / voice.status events are addressed
    profile: str | None = None


class VoiceRecordStatus(WireEnum):
    recording = "recording"
    stopped = "stopped"
    busy = "busy"


class VoiceRecordResult(Result):
    status: VoiceRecordStatus
    reason: str | None = None  # "wake_owned" when another surface holds the mic


method("voice.record", params=VoiceRecordParams, result=VoiceRecordResult,
       doc="VAD-bounded push-to-talk; the transcript arrives as a voice.transcript event.")


class VoiceTtsParams(Params):
    text: str
    profile: str | None = None


class VoiceTtsResult(Result):
    status: str  # "speaking"


method("voice.tts", params=VoiceTtsParams, result=VoiceTtsResult,
       doc="Speak text through the backend TTS engine (barge-in aware).")


# ── wake word ─────────────────────────────────────────────────────────────────────────────────


class WakeStartParams(Params):
    """``surface`` names the caller ("tui" | "gui"); ``persist`` is the explicit gesture that also
    flips ``wake_word.enabled`` on; ``client_capture`` asks for PCM streamed via wake.feed."""

    surface: str | None = None
    persist: bool | None = None
    client_capture: bool | None = None
    session_id: str | None = None  # session the wake.detected event is addressed to
    profile: str | None = None


class WakeStartResult(Result):
    """``started: false`` carries ``reason`` (unavailable / disabled / disabled_for_surface / owned);
    ``sample_rate`` / ``frame_length`` describe the PCM frames the armed detector expects."""

    started: bool
    reason: str | None = None
    hint: str | None = None
    phrase: str | None = None
    provider: str | None = None
    owner_surface: str | None = None
    enabled_persisted: bool | None = None
    capture: str | None = None
    sample_rate: int | None = None
    frame_length: int | None = None


method("wake.start", params=WakeStartParams, result=WakeStartResult,
       doc="Arm the wake-word listener for the calling surface; refusals explain why.")


class WakeStopParams(Params):
    persist: bool | None = None
    profile: str | None = None


class WakeOwnerResult(Result):
    """``methods_voice.py::_owner_result`` — ``reason: "not_owner"`` when the caller does not hold
    the listener."""

    reason: str | None = None


class WakeStopResult(WakeOwnerResult):
    stopped: bool
    disabled_persisted: bool


method("wake.stop", params=WakeStopParams, result=WakeStopResult,
       doc="Stop this surface's listener; persist also writes wake_word.enabled: false.")


class WakeControlParams(Params):
    profile: str | None = None


class WakePauseResult(WakeOwnerResult):
    paused: bool


method("wake.pause", params=WakeControlParams, result=WakePauseResult,
       doc="Release the mic (e.g. while the desktop's browser captures audio).")


class WakeResumeResult(WakeOwnerResult):
    resumed: bool


method("wake.resume", params=WakeControlParams, result=WakeResumeResult,
       doc="Reclaim the mic after a pause; no-op if the listener isn't armed.")


class WakeStatusParams(Params):
    surface: str | None = None
    client_capture: bool | None = None
    profile: str | None = None


class WakeInputDevice(OpenModel):
    """``tools/wake_word.py::_describe_input_device`` — PortAudio diagnostics for the configured mic."""

    selector: int | str | None = None
    name: str | None = None
    error: str | None = None
    max_input_channels: int | None = None
    default_samplerate: float | None = None
    hostapi_index: int | None = None
    hostapi: str | None = None


class WakeStatusResult(Result):
    """``enabled`` is config truth; ``listening`` is this caller's armed detector; ``audio_silent``
    means armed but deaf (see ``hint``)."""

    listening: bool
    owned_by_caller: bool
    owner_surface: str | None = None
    phrase: str
    provider: str
    configured_surface: str
    input_device: WakeInputDevice
    available: bool
    hint: str
    enabled: bool
    audio_silent: bool
    capture: str
    local_input_available: bool
    sample_rate: int
    frame_length: int


method("wake.status", params=WakeStatusParams, result=WakeStatusResult,
       doc="Everything a client needs to draw the wake-word state and decide whether to (re)arm.")


class WakeFeedParams(Params):
    """``pcm`` (or the ``pcm_b64`` alias): base64 int16 mono little-endian, 16 kHz only."""

    pcm: str | None = None
    pcm_b64: str | None = None
    sample_rate: int | None = None
    profile: str | None = None


class WakeFeedResult(WakeOwnerResult):
    fed: bool  # false with reason "empty" / "not_owner"


method("wake.feed", params=WakeFeedParams, result=WakeFeedResult,
       doc="Push client-captured PCM into the armed detector (mic-less remote backends).")
