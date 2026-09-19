"""The lifecycle for each ``manage_connections`` operation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tools.connectors.operation import ConnectionOperation, Leg
from tools.connectors.contract import SettleReason
from tools.operations import OperationAlreadyOpen, operations

WATCH_INTERVAL_SECONDS = 5.0

Callback = Callable[[Dict[str, Any]], Optional[str]]


@dataclass
class LegDriver:
    """Hooks that prepare, observe, and describe one operation's legs."""

    prepare: Callable[[ConnectionOperation], None]
    observe: Callable[[ConnectionOperation], None]
    note: str


def run_operation(
    legs: List[Leg],
    kind: LegDriver,
    *,
    session_key: str,
    tool_call_id: Optional[str],
    connection_callback: Optional[Callback],
    tick_seconds: Optional[float] = None,
    with_urls_in_result: bool,
) -> str:
    """Block the tool thread until the operation settles and return JSON."""
    operation = ConnectionOperation(legs, session_key=session_key, tool_call_id=tool_call_id)
    try:
        operations.open(operation, exclusive=True)
    except OperationAlreadyOpen as exc:
        from tools.registry import tool_error

        return tool_error(
            f"a connection operation is already open in this session ({exc.existing.op_id}); it settles "
            "when the user finishes with the card, on Continue, or at its deadline. Do not start another."
        )
    return drive_operation(
        operation,
        kind,
        connection_callback=connection_callback,
        tick_seconds=tick_seconds,
        with_urls_in_result=with_urls_in_result,
    )


def drive_operation(
    operation: ConnectionOperation,
    kind: LegDriver,
    *,
    connection_callback: Optional[Callback],
    tick_seconds: Optional[float] = None,
    with_urls_in_result: bool,
) -> str:
    """Run an already-registered operation to settlement."""
    try:
        kind.prepare(operation)
        operation.settle_if_all_resolved()
        if connection_callback is not None and not operation.settled:
            connection_callback(operation.request_payload())
        _watch(operation, kind, tick_seconds)
    finally:
        operations.close(operation)
    payload = operation.result(with_urls=with_urls_in_result)
    payload["status"] = "settled"
    payload["note"] = kind.note
    return json.dumps(payload, ensure_ascii=False)


def _watch(operation: ConnectionOperation, kind: LegDriver, tick_seconds: Optional[float]) -> None:
    from tools.interrupt import is_interrupted

    tick = WATCH_INTERVAL_SECONDS if tick_seconds is None else tick_seconds
    while not operation.settled:
        if is_interrupted():
            operation.settle(SettleReason.interrupt)
            return
        if time.time() >= operation.deadline_at:
            operation.settle(SettleReason.deadline)
            return
        kind.observe(operation)
        if operation.settled or operation.settle_if_all_resolved():
            return
        operations.wait(operation, timeout=tick)
