"""The one lifecycle every ``manage_connections`` operation follows, whatever the target kind.

``run_operation`` mints the op, registers it in ``live``, lets the kind prepare its targets (a
managed mint, an MCP catalog check), emits the card through the session callback, then loops:
sleep on ``op.wake`` for at most one tick, run the kind's ``observe`` hook, settle when every
target is resolved or the deadline passes. ``connection.respond`` reaches the loop by transitioning
the op through ``live`` and setting ``wake``; ``/stop`` sets the thread interrupt flag."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tools.connectors import live
from tools.connectors.contract import SettleReason
from tools.connectors.operation import ConnectionOperation, Target

WATCH_INTERVAL_SECONDS = 5.0
# The interrupt flag has no wake hook, so the tick sleep is sliced and the flag read each slice.
_WAKE_SLICE_SECONDS = 0.25

Callback = Callable[[Dict[str, Any]], Optional[str]]


@dataclass
class Kind:
    """Per-kind hooks. ``prepare`` runs once before the card; ``observe`` runs every tick and may
    transition targets; ``note`` is the model-facing guidance appended to the settled result."""

    prepare: Callable[[ConnectionOperation], None]
    observe: Callable[[ConnectionOperation], None]
    note: str


def run_operation(
    targets: List[Target],
    kind: Kind,
    *,
    session_key: str,
    tool_call_id: Optional[str],
    connection_callback: Optional[Callback],
    tick_seconds: Optional[float] = None,
    with_urls_in_result: bool,
) -> str:
    """Block the tool thread until the operation settles; return the tool's JSON string."""
    operation = ConnectionOperation(targets, session_key=session_key, tool_call_id=tool_call_id)
    try:
        live.open(operation)
    except live.OperationAlreadyOpen as exc:
        from tools.registry import tool_error

        return tool_error(
            f"a connection operation is already open in this session ({exc.existing.op_id}); it settles "
            "when the user finishes with the card, on Continue, or at its deadline. Do not start another."
        )
    try:
        kind.prepare(operation)
        if connection_callback is not None and not operation.settled:
            connection_callback(operation.request_payload())
        _watch(operation, kind, tick_seconds)
    finally:
        live.close(operation)
    payload = operation.result(with_urls=with_urls_in_result)
    payload["status"] = "settled"
    payload["note"] = kind.note
    return json.dumps(payload, ensure_ascii=False)


def _watch(operation: ConnectionOperation, kind: Kind, tick_seconds: Optional[float]) -> None:
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
        # The card or the clock may have settled the op during the read; its result is frozen.
        if operation.settled or operation.settle_if_all_resolved():
            return
        _sleep_until_wake(operation, tick)


def _sleep_until_wake(operation: ConnectionOperation, tick: float) -> None:
    """Sleep up to one tick, leaving early on ``wake``, the deadline, or the interrupt flag."""
    from tools.interrupt import is_interrupted

    until = min(time.time() + tick, operation.deadline_at)
    while not operation.settled and not is_interrupted():
        remaining = until - time.time()
        if remaining <= 0:
            break
        if operation.wake.wait(min(_WAKE_SLICE_SECONDS, remaining)):
            break
    operation.wake.clear()
