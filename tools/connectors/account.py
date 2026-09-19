"""Account-owned managed connector operations for the session-less Connectors page."""

from __future__ import annotations

import contextvars
import threading
import uuid
from dataclasses import dataclass, field

from tools.connectors.legs import managed
from tools.connectors.operation import ConnectionOperation, Leg
from tools.connectors.run import drive_operation
from tools.operations import Owner, operations


_PREPARE_WAIT_SECONDS = 31.0
_start_lock = threading.Lock()
_by_connector: dict[tuple[str, str], ConnectionOperation] = {}


@dataclass
class AccountOperationStart:
    operation: ConnectionOperation
    started: bool
    done: threading.Event = field(default_factory=threading.Event)
    failed: bool = False


def _open_leg(name: str, profile: str) -> ConnectionOperation | None:
    operation = _by_connector.get((profile, name))
    return operation if operation is not None and not operation.settled else None


def _matching_open_operation(names: list[str], *, profile: str) -> ConnectionOperation | None:
    matches = [_open_leg(name, profile) for name in names]
    first = next((operation for operation in matches if operation is not None), None)
    if first is None:
        return None
    if any(operation is not first for operation in matches):
        raise ValueError("requested connectors do not share one open operation")
    return first


def find_or_start_operation(
    names: list[str],
    *,
    action: str,
    profile_home: str | None,
) -> AccountOperationStart:
    """Join the exact open managed operation, or atomically reserve and start a new one."""
    profile = Owner.of("", profile_home=profile_home).profile
    with _start_lock:
        if operation := _matching_open_operation(names, profile=profile):
            return AccountOperationStart(operation=operation, started=False)
        session_key = f"account:{uuid.uuid4().hex}"
        operation = ConnectionOperation(
            [Leg(name, "connector", action) for name in names],
            session_key=session_key,
            owner=Owner.of(session_key, profile_home=profile_home),
        )
        started = AccountOperationStart(operation=operation, started=True)
        context = contextvars.copy_context()
        thread = threading.Thread(
            target=lambda: context.run(_run, started, action),
            daemon=True,
            name="connector-account-operation",
        )
        try:
            operations.open(operation, exclusive=True)
            for leg in operation.legs:
                _by_connector[(profile, leg.name)] = operation
            thread.start()
        except Exception:
            operations.close(operation)
            for leg in operation.legs:
                _by_connector.pop((profile, leg.name), None)
            raise
    return started


def _run(start: AccountOperationStart, action: str) -> None:
    operation = start.operation
    try:
        drive_operation(
            operation,
            managed.managed_kind(managed.managed_client(), action, force=False),
            connection_callback=lambda _payload: start.done.set(),
            tick_seconds=managed.WATCH_TICK_SECONDS,
            with_urls_in_result=False,
        )
    except Exception:
        operations.close(operation)
        start.failed = True
    finally:
        profile = operation.owner.profile if operation.owner else None
        for leg in operation.legs:
            _by_connector.pop((profile, leg.name), None)
        start.done.set()


def wait_for_prepare(start: AccountOperationStart) -> bool:
    """Wait only for the initial mint, whose HTTP client has its own bounded timeout."""
    return start.done.wait(_PREPARE_WAIT_SECONDS)
