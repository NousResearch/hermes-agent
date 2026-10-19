"""Session-bound client permission for terminal execution, independent of local policy."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
_requester = ContextVar("terminal_client_permission_requester", default=None)
_execution = ContextVar("terminal_client_permission_execution", default=None)


def set_command_permission_requester(requester):
    return _requester.set(requester)


def reset_command_permission_requester(token):
    _requester.reset(token)


@dataclass
class _Execution:
    command: str
    approved: bool = False


def record_command_permission(command: str, choice: str) -> None:
    """Reuse an ACP danger prompt only within this one execution of this command."""
    execution = _execution.get()
    if execution is not None and execution.command == command:
        execution.approved = choice in {"once", "session", "always"}


@contextmanager
def command_permission_scope(command: str):
    """Local guards run first; accepted commands still require the client's decision."""
    requester = _requester.get()
    execution = _Execution(command)
    token = _execution.set(execution)

    def authorize() -> bool:
        if requester is None or execution.approved:
            return True
        try:
            return requester(command) is True
        except Exception:
            logger.debug("Terminal client permission failed", exc_info=True)
            return False

    try:
        yield authorize
    finally:
        _execution.reset(token)
