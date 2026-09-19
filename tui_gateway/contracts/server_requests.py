"""Server→client requests: the backend asks the renderer a question (``server_requests.send``).

Every entry declares the params carried by its request frame (``session_id`` is added by the
transport) and the result returned by the client. ``request.cancel`` is the event-enveloped
withdrawal of an open request.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field

from .base import Params, Payload, Result, WireEnum
from .common import ApprovalChoice
from .registry import event, server_request


class ServerRequestParams(Params):
    """Every request frame belongs to one gateway session."""

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
    multi_select: bool


class ClarifySingle(ServerRequestParams):
    """One clarify question; ``answer: ''`` is the client's explicit skip."""

    kind: Literal["single"]
    question: str
    choices: list[str] | None = None
    multi_select: bool


class ClarifyBatch(ServerRequestParams):
    """Multiple questions. ``answers`` is ``None`` when first sent and carries locked answers only
    on a reconnect replay."""

    kind: Literal["batch"]
    questions: list[ClarifyQuestion]
    answers: dict[str, str] | None = None


# ``server_request`` accepts this alias through the registry's TypeAdapter-backed declaration.
ClarifyRequestParams = Annotated[Union[ClarifySingle, ClarifyBatch], Field(discriminator="kind")]


class ClarifyAnswer(Result):
    """A single answer; ``answer: ''`` is an explicit skip."""

    answer: str


class ClarifyAnswers(Result):
    """The batch answer set. A deadline preserves locked answers with ``timed_out: true``."""

    answers: dict[str, str]
    timed_out: bool = False


ClarifyResult = Union[ClarifyAnswer, ClarifyAnswers]


# ``server_request`` accepts this union alias after A1's TypeAdapter registry change.
server_request("clarify", params=ClarifyRequestParams, result=ClarifyResult,  # type: ignore[arg-type]
               doc="The clarify tool: ask the user one question or a batch.")


# ── approval ──────────────────────────────────────────────────────────────────────────────────


class ApprovalRequestParams(ServerRequestParams):
    """Closed payload from ``tools.approval`` after ``_approval_request_payload`` redacts the
    command and derives ``choices``."""

    request_id: str
    command: str
    description: str
    pattern_key: str | None = None
    pattern_keys: list[str] | None = None
    allow_permanent: bool | None = None
    allow_session: bool | None = None
    smart_denied: bool | None = None
    choices: list[ApprovalChoice]


class ApprovalResult(Result):
    choice: ApprovalChoice
    all: bool = False


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


class SecretMetadata(Params):
    """``tools.skills_tool_setup._capture_required_environment_variables`` supplies these keys."""

    skill_name: str
    help: str | None = None
    required_for: str | None = None


class SecretRequestParams(ServerRequestParams):
    env_var: str
    prompt: str
    metadata: SecretMetadata | None = None


server_request("secret", params=SecretRequestParams, result=ValueResult,
               doc="Masked value for a named env var in a skill setup flow.")


class VaultUnlockRequestParams(ServerRequestParams):
    backend: str
    display_name: str


server_request("vault.unlock_prompt", params=VaultUnlockRequestParams, result=ValueResult,
               doc="Master password to unlock an external password manager for this session.")


class VaultSaveLoginRequestParams(ServerRequestParams):
    origin: str
    site: str


server_request("vault.save_login", params=VaultSaveLoginRequestParams, result=ValueResult,
               doc="Save a login for a site; the value is JSON {identifier, password}.")


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


class PreviewActAction(WireEnum):
    elements = "elements"
    click = "click"
    hover = "hover"
    type = "type"
    scroll = "scroll"
    press = "press"
    strobe = "strobe"
    back = "back"
    forward = "forward"
    reload = "reload"
    pin = "pin"
    hold = "hold"
    unpin = "unpin"


class PreviewScrollTo(WireEnum):
    top = "top"
    bottom = "bottom"


class PreviewActRequestParams(ServerRequestParams):
    """Closed DOM-operation shape from ``tools.drive_preview_tool`` and
    ``tools.annotate_preview_tool``."""

    action: PreviewActAction
    ref: str | None = None
    selector: str | None = None
    text: str | None = None
    key: str | None = None
    submit: bool | None = None
    full: bool | None = None
    to: PreviewScrollTo | None = None
    amount: int | None = None
    max: int | None = None


server_request("preview.act", params=PreviewActRequestParams, result=ValueResult,
               doc="Click, type, scroll, or annotate inside the in-app browser preview.")


class TourAction(WireEnum):
    targets = "targets"
    show = "show"
    start = "start"
    next = "next"
    prev = "prev"
    stop = "stop"


class TourSurface(WireEnum):
    app = "app"
    preview = "preview"


class TourSide(WireEnum):
    top = "top"
    right = "right"
    bottom = "bottom"
    left = "left"


class TourStep(Params):
    """Closed DOM step from ``tools.tour_tool``'s ``_STEP_SCHEMA``."""

    selector: str | None = None
    title: str | None = None
    text: str | None = None
    side: TourSide | None = None


class TourRequestParams(ServerRequestParams):
    """Closed guided-tour DOM operation from ``tools.tour_tool``."""

    # tools/tour_tool.py drops None fields before calling back, so every optional key defaults here.
    action: TourAction
    surface: TourSurface | None = None
    selector: str | None = None
    title: str | None = None
    text: str | None = None
    side: TourSide | None = None
    steps: list[TourStep] | None = None
    step_index: int | None = None


server_request("tour", params=TourRequestParams, result=ValueResult,
               doc="Drive a guided tour highlight in the desktop renderer.")


# ── withdrawal ────────────────────────────────────────────────────────────────────────────────


class WithdrawnRequestMethod(WireEnum):
    approval = "approval"
    clarify = "clarify"
    mcp_setup = "mcp.setup"
    preview_act = "preview.act"
    preview_read = "preview.read"
    secret = "secret"
    sudo = "sudo"
    terminal_read = "terminal.read"
    tour = "tour"
    vault_code = "vault.code"
    vault_save_login = "vault.save_login"
    vault_unlock_prompt = "vault.unlock_prompt"
    window_read = "window.read"


class RequestCancelReason(WireEnum):
    """Every value a producer withdraws a request with: ``server_requests`` (timeout / interrupted /
    shutdown) and the approval queue's ``_drop_entry`` (resolved from another surface, session_closed,
    notify_failed, plus its own timeout / interrupted)."""
    interrupted = "interrupted"
    notify_failed = "notify_failed"
    resolved = "resolved"
    session_closed = "session_closed"
    shutdown = "shutdown"
    timeout = "timeout"


class RequestCancelPayload(Payload):
    id: str
    method: WithdrawnRequestMethod
    reason: RequestCancelReason


event("request.cancel", RequestCancelPayload,
      doc="The backend withdrew an open server→client request; clear the matching card only.")
