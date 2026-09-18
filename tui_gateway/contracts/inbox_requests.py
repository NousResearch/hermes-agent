"""Contracts for ``inbox.requests``: scoped request-detail read layer."""

from __future__ import annotations

from .base import JsonValue, Params, Result
from .common import ProfileParams
from .registry import method


class InboxRequestsParams(ProfileParams):
    session_key: str


class InboxRequestApproval(Result):
    request_id: str = ""
    command: str = ""
    description: str = ""
    choices: list[str] = []
    allow_permanent: bool | None = None
    allow_session: bool | None = None
    smart_denied: bool | None = None
    tool_name: str | None = None


class InboxRequestClarifyQuestion(Result):
    qid: str = ""
    question: str = ""
    choices: list[str] | None = None
    multi_select: bool = False


class InboxRequestClarifyParams(Result):
    question: str | None = None
    choices: list[str] | None = None
    multi_select: bool | None = None
    questions: list[InboxRequestClarifyQuestion] | None = None
    answers: dict[str, str] | None = None


class InboxRequestClarification(Result):
    request_id: str = ""
    kind: str = ""  # "single" or "batch"
    params: InboxRequestClarifyParams = InboxRequestClarifyParams()


class InboxRequestContextMessage(Result):
    role: str = ""  # "user" | "assistant"
    text: str = ""
    timestamp: float | None = None


class InboxRequestContext(Result):
    """Bounded, redacted excerpt of the owning session's recent turns.

    ``available=False`` is a real state, not an empty transcript: the reason names
    why (no displayable rows, or a failed read) so the panel can say "unavailable"
    instead of implying an all-clear.
    """

    available: bool = False
    reason: str | None = None
    messages: list[InboxRequestContextMessage] = []


class InboxExpiredRequest(Result):
    """One request that ended without an answer; what it was for plus how it ended."""

    request_id: str = ""
    kind: str = ""  # "approval"
    command: str = ""
    description: str = ""
    ended_at: float = 0.0
    outcome: str = ""  # "timeout" | "session_closed" | "interrupted" | "notify_failed" | ...


class InboxRequestSessionDetail(Result):
    live_session_ids: list[str] = []
    approvals: list[InboxRequestApproval] = []
    clarifications: list[InboxRequestClarification] = []
    context: InboxRequestContext = InboxRequestContext()
    expired_requests: list[InboxExpiredRequest] = []


class InboxRequestsCoverage(Result):
    profile: str = ""
    session_key: str = ""
    live_session_count: int = 0
    approval_count: int = 0
    clarification_count: int = 0
    context_anchor: str = ""  # "available" | "unavailable: open chat for context" | etc.
    errors: list[str] = []


class InboxRequestsResult(Result):
    sessions: list[InboxRequestSessionDetail] = []
    coverage: InboxRequestsCoverage = InboxRequestsCoverage()


class InboxRedoParams(ProfileParams):
    session_key: str
    request_id: str


class InboxRedoResult(Result):
    redone: bool = False
    session_id: str = ""
    record_cleared: bool = False


class InboxDismissParams(ProfileParams):
    session_key: str
    request_id: str


class InboxDismissResult(Result):
    dismissed: bool = False


method(
    "inbox.requests",
    params=InboxRequestsParams,
    result=InboxRequestsResult,
    doc="Read-only scoped request details for a specific session (approvals + clarifications).",
)

method(
    "inbox.redo",
    params=InboxRedoParams,
    result=InboxRedoResult,
    doc="Re-raise an expired request by asking its live session to attempt the action again.",
)

method(
    "inbox.dismiss",
    params=InboxDismissParams,
    result=InboxDismissResult,
    doc="Drop one expired-request record from the inbox.",
)
