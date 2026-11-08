"""Recoverable conversation closeout, identical on legacy and authority transports."""

from typing import Literal

from pydantic import Field

from .base import Payload, Result
from .common import ProfileParams
from .registry import canonical_method, event, method


class SessionCloseoutParams(ProfileParams):
    session_id: str
    request_id: str = Field(min_length=1, max_length=160)
    intent: Literal['archive', 'new']


class CloseoutOldSession(Result):
    logical_session_id: str
    stored_session_id: str
    runtime_id: str | None = None
    profile: str


class CloseoutDestinations(Result):
    memory: Literal['accepted', 'skipped', 'pending', 'unknown']
    tasks: Literal['reconciled', 'pending', 'unknown']


class CloseoutError(Result):
    code: str
    message: str


class SessionCloseoutResult(Result):
    closeout_id: str
    status: Literal['reviewing', 'preserved', 'failed']
    settled: bool
    old: CloseoutOldSession
    receipt_id: str | None = None
    destinations: CloseoutDestinations
    archived: bool
    next_create_request_id: str
    memory_source_id: str | None = None
    task_ids: list[str] = Field(default_factory=list)
    error: CloseoutError | None = None


class SessionCloseoutUpdated(SessionCloseoutResult, Payload):
    pass


_DOC = ('Preserve one exact conversation, archive it recoverably, and return a durable receipt. '
        'Retries read the same operation; pending online destinations do not imply accepted saves. '
        'New-chat creation uses the returned stable request id and never resets the archived root.')
method('session.closeout', params=SessionCloseoutParams, result=SessionCloseoutResult, doc=_DOC)
canonical_method('session.closeout', params=SessionCloseoutParams, result=SessionCloseoutResult, doc=_DOC)
event('session.closeout.updated', SessionCloseoutUpdated)
