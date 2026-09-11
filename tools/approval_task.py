"""Ephemeral task EVIDENCE from the authenticated Desktop composer contract.

Not an authorization token: a trusted client can forge this just as it can send
approval responses. Never mint from model messages, generic RPC text, or config.
Only the foreground inline Desktop submission currently produces a lease.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re
from threading import Event
from uuid import uuid4

MAX_TASK_CHARS = 8192
_INVALID = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


@dataclass(frozen=True)
class TaskRecord:
    session_key: str
    task_id: str
    raw_text: str


class TaskLease:
    def __init__(self, record: TaskRecord):
        self.record = record
        self._revoked = Event()

    def revoke(self):
        self._revoked.set()

    @property
    def active(self):
        return not self._revoked.is_set()


_task: ContextVar[TaskLease | None] = ContextVar("approval_task_evidence", default=None)


def from_composer(session_key: str, provenance) -> TaskLease | None:
    """Validate the NEW wire contract; caller must enforce transport/admission."""
    if not isinstance(provenance, dict) or set(provenance) != {"kind", "raw_text"}:
        return None
    raw = provenance.get("raw_text")
    if (provenance.get("kind") != "desktop_composer" or not session_key
            or not isinstance(raw, str) or not raw.strip()
            or len(raw) > MAX_TASK_CHARS or _INVALID.search(raw)):
        return None
    return TaskLease(TaskRecord(session_key, uuid4().hex, raw))


def revoke_session_task(session: dict) -> None:
    lease = session.pop("_approval_task_lease", None)
    if isinstance(lease, TaskLease):
        lease.revoke()


def release_task(session: dict, lease: TaskLease | None) -> None:
    """Retire only this generation; never remove a successor's lease."""
    if lease is not None:
        lease.revoke()
        if session.get("_approval_task_lease") is lease:
            session.pop("_approval_task_lease", None)


@contextmanager
def bind_task(lease: TaskLease | None):
    token = _task.set(lease)
    try:
        yield
    finally:
        _task.reset(token)


def task_revoked() -> bool:
    """Copied workers retain the lease, so cancellation is visible before execution."""
    lease = _task.get()
    return lease is not None and not lease.active


def current_task() -> TaskRecord | None:
    from agent.delegation_context import is_delegated_child_context
    from tools.approval_context import get_current_session_key
    lease = _task.get()
    if (lease is None or not lease.active or is_delegated_child_context()
            or lease.record.session_key != get_current_session_key()):
        return None
    return lease.record
