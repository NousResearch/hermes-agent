"""Server→client requests: the backend asks the renderer a question (``server_requests.send``).

Every entry is one request method: the ``params`` the frame carries (``session_id`` is added by
the transport and declared on the shared base) and the ``result`` the client answers with. The
``request.cancel`` event that withdraws an open request lives here too.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tools.tour_presets import TourPreset

from .base import JsonValue, Params, Payload, Result, WireEnum
from .liveness import DeviceCapability
from .registry import event, server_request


class ServerRequestParams(Params):
    """Every request frame's params start with the session the question belongs to."""

    session_id: str


class ValueResult(Result):
    """The answer to any one-string prompt (sudo, secret, vault prompts, desktop bridges):
    ``''`` means skipped / declined."""

    value: str


# ── clarify ───────────────────────────────────────────────────────────────────────────────────


class ClarifyQuestion(Params):
    qid: str
    question: str
    choices: list[str] | None = None
    multi_select: bool = False


class ClarifyRequestParams(ServerRequestParams):
    """``answers`` rides only on a reconnect replay (locks the server already accepted; null = skipped)."""

    questions: list[ClarifyQuestion]
    answers: dict[str, str | None] | None = None


class ClarifyResult(Result):
    """``{answers}`` for the whole set (early locks go through the ``clarify.lock`` RPC); a response
    without ``answers`` is cancel-all. ``outcome`` (optional): how the viewer's own panel ended, as
    main's classic CLI reports it (``timed_out`` keeps the answers locked before its deadline)."""

    answers: dict[str, str | None] | None = None
    outcome: Literal["submitted", "timed_out", "cancelled"] | None = None


server_request("clarify", params=ClarifyRequestParams, result=ClarifyResult,
               doc="The clarify tool: ask the user 1-5 questions.")



# ── setup_beat (first-run onboarding) ─────────────────────────────────────────────────────────


class SetupBeat(WireEnum):
    name = "name"
    theme = "theme"
    layout = "layout"
    connectors = "connectors"
    plugins = "plugins"
    tour = "tour"
    build = "build"


class SetupOption(Params):
    id: str
    label: str


class SetupNameCard(Params):
    suggested: str | None = None  # the OS account's name: shown on the card, never sent to the model


class SetupThemeCard(Params):
    picker: str


class SetupOptionsCard(Params):
    options: list[SetupOption]


class SetupConnectorsCard(Params):
    suggested: list[str]


class SetupPluginsCard(Params):
    plugins: list[str]


class SetupBeatRequestParams(ServerRequestParams):
    """``card`` is the beat's data (``agent/onboarding_flow.py::BEATS``): name, theme, connectors and
    plugins have their own shape; layout, tour and build carry ``options``. ``tool_call_id`` is the
    setup_beat call the card belongs to: only that transcript row draws it."""

    beat: SetupBeat
    question: str | None = None
    tool_call_id: str | None = None
    card: SetupNameCard | SetupThemeCard | SetupOptionsCard | SetupConnectorsCard | SetupPluginsCard


class SetupNameAnswer(Result):
    name: str


class SetupThemeAnswer(Result):
    accent: str | None = None
    mode: str | None = None


class SetupLayoutAnswer(Result):
    layout: str


class SetupConnectorsAnswer(Result):
    connected: list[str]
    newly_connected: list[str] = Field(default_factory=list)


class SetupPluginsAnswer(Result):
    installed: list[str]
    label: str | None = None  # what the card showed, for the transcript; never recorded


class SetupTourAnswer(Result):
    tour: str
    label: str | None = None


class SetupBuildAnswer(Result):
    build: str
    label: str | None = None


class SetupBeatResult(Result):
    """``answer`` holds the shown beat's picks; ids must be one of the card's options (the backend
    rejects any other answer). ``said`` is composer text typed while the name card waited: the name.
    ``wrote_instead``: the user sent a message instead of answering; the backend ends the turn and the
    client sends that message as the next one. A response with none of these, or none within 300 s,
    closes the card unanswered."""

    answer: (SetupNameAnswer | SetupThemeAnswer | SetupLayoutAnswer | SetupConnectorsAnswer | SetupPluginsAnswer
             | SetupTourAnswer | SetupBuildAnswer | None) = None
    said: str | None = None
    wrote_instead: bool = False


server_request("setup_beat", params=SetupBeatRequestParams, result=SetupBeatResult,
               doc="The onboarding chat's card for one beat; blocks setup_beat until answered (300 s).")


# ── approval ──────────────────────────────────────────────────────────────────────────────────


class ApprovalChoice(WireEnum):
    once = "once"
    session = "session"
    always = "always"
    deny = "deny"


class ApprovalRequestParams(ServerRequestParams):
    """``tui_gateway/server.py::_approval_request_payload`` — the command is redacted server-side."""

    request_id: str
    command: str = ""
    description: str = ""
    choices: list[ApprovalChoice] = Field(default_factory=list)
    allow_permanent: bool | None = None
    allow_session: bool | None = None
    smart_denied: bool | None = None
    tool_name: str | None = None
    session_id_hint: str | None = Field(default=None, alias="gateway_session_id")
    # The approval queue entry carries tool-specific context the card may render; the closed set
    # of keys is owned by tools/approval.py, so it stays open here.
    model_config = Params.model_config | {"extra": "allow"}


class ApprovalResult(Result):
    choice: ApprovalChoice
    all: bool | None = None


server_request("approval", params=ApprovalRequestParams, result=ApprovalResult,
               doc="A dangerous command awaits the user's decision.")


# ── one-string prompts ────────────────────────────────────────────────────────────────────────


class EmptyRequestParams(ServerRequestParams):
    pass


class SudoRequestParams(ServerRequestParams):
    """Original command, redacted server-side before any password-injection rewrite."""

    command: str = ""


server_request("sudo", params=SudoRequestParams, result=ValueResult,
               doc="Masked sudo password for the terminal tool.")


class SecretRequestParams(ServerRequestParams):
    env_var: str
    prompt: str
    metadata: dict[str, JsonValue] | None = None


server_request("secret", params=SecretRequestParams, result=ValueResult,
               doc="Masked value for a named env var (skills / setup flows).")


class VaultUnlockRequestParams(ServerRequestParams):
    backend: str
    display_name: str


server_request("vault.unlock_prompt", params=VaultUnlockRequestParams, result=ValueResult,
               doc="Master password to unlock an external password manager for this session.")


class VaultSaveLoginRequestParams(ServerRequestParams):
    origin: str
    site: str


server_request("vault.save_login", params=VaultSaveLoginRequestParams, result=ValueResult,
               doc="Save a login for a site the agent is about to fill; the answer is JSON {identifier, password}.")


class VaultCodeRequestParams(ServerRequestParams):
    site: str | None = None
    hint: str | None = None


server_request("vault.code", params=VaultCodeRequestParams, result=ValueResult,
               doc="A one-time / 2FA code the user reads from their device.")


# ── desktop GUI bridges ───────────────────────────────────────────────────────────────────────


class ReadRangeRequestParams(ServerRequestParams):
    start: int | None = None
    count: int | None = None


server_request("terminal.read", params=ReadRangeRequestParams, result=ValueResult,
               doc="Read the visible in-app terminal buffer (JSON text answer).")
server_request("preview.read", params=ReadRangeRequestParams, result=ValueResult,
               doc="Read the in-app browser preview's text (JSON text answer).")
server_request("window.read", params=EmptyRequestParams, result=ValueResult,
               doc="Enumerate the native window below the app (JSON text answer).")


class PreviewActRequestParams(ServerRequestParams):
    """``tools/drive_preview_tool.py`` and ``tools/annotate_preview_tool.py`` field sets."""

    action: str
    ref: str | None = None
    selector: str | None = None
    text: str | None = None
    key: str | None = None
    submit: bool | None = None
    full: bool | None = None
    to: str | None = None
    amount: int | None = None
    max: int | None = None
    allow_shortcut: bool | None = None


server_request("preview.act", params=PreviewActRequestParams, result=ValueResult,
               doc="Click / type / scroll / annotate inside the in-app browser preview.")


class TourStep(Params):
    selector: str | None = None
    title: str | None = None
    text: str | None = None
    side: str | None = None
    model_config = Params.model_config | {"extra": "allow"}


class TourRequestParams(ServerRequestParams):
    """``tools/tour_tool.py`` field set."""

    action: str
    surface: str | None = None
    selector: str | None = None
    title: str | None = None
    text: str | None = None
    side: str | None = None
    steps: list[TourStep] | None = None
    step_index: int | None = None
    preset: TourPreset | None = None


server_request("tour", params=TourRequestParams, result=ValueResult,
               doc="Drive a guided tour highlight in the desktop renderer.")


class ThemeMode(WireEnum):
    light = "light"
    dark = "dark"
    system = "system"


class ThemePicker(WireEnum):
    accent = "accent"
    layout = "layout"


class ThemeApplyRequestParams(ServerRequestParams):
    """``tools/set_theme_tool.py`` values; accent is a swatch id, ``theme`` or ``#rrggbb``."""

    accent: str | None = None
    mode: ThemeMode | None = None
    layout: str | None = None


class ThemePickRequestParams(ServerRequestParams):
    picker: ThemePicker


class ThemeValues(Params):
    """What the renderer applied. ``accent`` is a swatch id, ``custom:#rrggbb``, or null for the theme's own."""

    accent: str | None = None
    mode: ThemeMode | None = None
    layout: str | None = None


class ThemePickResult(Result):
    """``picked`` once the user chose (already applied); ``cancelled`` when the card was dismissed. A response
    with neither is a cancel."""

    picked: ThemeValues | None = None
    said: str | None = None
    cancelled: bool | None = None


server_request("theme.apply", params=ThemeApplyRequestParams, result=ValueResult,
               doc="Apply set_theme values in the window showing the session (JSON text answer).")
server_request("theme.pick", params=ThemePickRequestParams, result=ThemePickResult,
               doc="Show an accent or layout picker card in the chat and wait for the pick.")


# ── handheld device bridges ───────────────────────────────────────────────────────────────────


class DeviceRequestOptions(Params):
    """Capability-specific knobs the client reads (``{quality: "high"}`` for a camera frame, …).
    Open by design: the capability vocabulary is closed, its options are not."""

    model_config = Params.model_config | {"extra": "allow"}


class DeviceRequestParams(ServerRequestParams):
    """``capability`` must be one the client advertised in ``client.capabilities``; the backend
    refuses the request at once when it did not, so an agent never stalls on a phone-less client."""

    capability: DeviceCapability
    options: DeviceRequestOptions | None = None


class DeviceCapture(Params):
    """One binary body a capability produced (a photo, a screen frame, a recording).

    Deliberately a separate field from ``value``: a capture is megabytes, and anything that reaches
    the model's tool result is re-sent on every later request of the conversation. The backend
    materialises ``data`` into the profile's media cache and the tool result carries the PATH
    (``vision_analyze`` reads it like any screenshot), so a frame costs a few hundred bytes of
    context instead of a whole base64 image (#11911).
    """

    #: ``image/jpeg``, ``image/png``, ``image/webp``, ``audio/wav``, ``audio/mpeg``, ``audio/webm``;
    #: anything else is refused by the backend rather than guessed at.
    mime_type: str
    #: The capture, base64 (the wire is JSON). Never echoed into a tool result.
    data: str
    #: Optional client-side label shown in the handle (``"rear camera"``, ``"screen"``).
    label: str | None = None


class DeviceRequestResult(Result):
    """What the capability answered. ``denied`` is the user's refusal — a distinct outcome from an
    absent capability, so a tool can report it instead of surfacing a transport failure.

    ``value`` is the JSON text a non-binary capability produced (a location fix, a notification
    list, a picked file's metadata); it must stay small — it is a tool result. Binary bodies belong
    in ``captures``, which never reach the conversation as text.
    """

    value: str = ""
    denied: bool = False
    reason: str | None = None
    captures: list[DeviceCapture] = Field(default_factory=list)


server_request("device.request", params=DeviceRequestParams, result=DeviceRequestResult,
               doc="Ask the client's device for one advertised capability (camera frame, location fix, …).")


# ── withdrawal ────────────────────────────────────────────────────────────────────────────────


class RequestCancelReason(WireEnum):
    timeout = "timeout"
    interrupted = "interrupted"
    shutdown = "shutdown"
    resolved = "resolved"
    session_closed = "session_closed"


class RequestCancelPayload(Payload):
    id: str
    method: str
    reason: str  # a RequestCancelReason value; callers in tools/approval may pass their own wording


event("request.cancel", RequestCancelPayload,
      doc="The backend withdrew an open server→client request; clear the matching card only.")
