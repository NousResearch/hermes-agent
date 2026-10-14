from __future__ import annotations

import contextvars
import threading
import uuid
from dataclasses import dataclass, field

from tools.connectors import live
from tools.connectors import managed
from tools.connectors.operation import ConnectionOperation, Target
from tools.connectors.run import drive_operation


_PREPARE_WAIT_SECONDS = 31.0
_start_lock = threading.Lock()


@dataclass
class AccountOperationStart:
    operation: ConnectionOperation
    started: bool
    done: threading.Event = field(default_factory=threading.Event)
    failed: bool = False


def _matching_open_operation(names: list[str], *, alias: str | None, repair_id: str | None,
                             profile_home: str | None) -> ConnectionOperation | None:
    matches = [live.find_target(name, alias=alias, repair_id=repair_id, profile_home=profile_home) for name in names]
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
    alias: str | None = None,
    repair_id: str | None = None,
) -> AccountOperationStart:
    """``alias`` names the one account of ``names[0]`` to connect; ``repair_id`` names the existing
    account a reconnect repairs. The caller passes one name with either."""
    with _start_lock:
        if operation := _matching_open_operation(names, alias=alias, repair_id=repair_id, profile_home=profile_home):
            return AccountOperationStart(operation=operation, started=False)
        operation = ConnectionOperation(
            [Target(name, "connector", action, alias=alias, repair_id=repair_id) for name in names],
            session_key=f"account:{uuid.uuid4().hex}",
        )
        started = AccountOperationStart(operation=operation, started=True)
        context = contextvars.copy_context()
        thread = threading.Thread(
            target=lambda: context.run(_run, started, action),
            daemon=True,
            name="connector-account-operation",
        )
        try:
            live.open(operation)
            thread.start()
        except Exception:
            live.close(operation)
            raise
    return started


def _run(start: AccountOperationStart, action: str) -> None:
    try:
        drive_operation(
            start.operation,
            managed.managed_kind(managed.managed_client(), action, force=False),
            connection_callback=lambda _payload: start.done.set(),
            tick_seconds=managed.WATCH_TICK_SECONDS,
            with_urls_in_result=False,
        )
    except Exception:
        live.close(start.operation)
        start.failed = True
    finally:
        start.done.set()


def wait_for_prepare(start: AccountOperationStart) -> bool:
    return start.done.wait(_PREPARE_WAIT_SECONDS)
