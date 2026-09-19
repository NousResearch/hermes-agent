"""Live operations, one open per session, found by ``op_id``. The RPC layer reads and drives
operations through here; the tool thread that minted one closes it on settle."""

from __future__ import annotations

import threading
from typing import Dict, Optional

from tools.connectors.operation import ConnectionOperation


class OperationAlreadyOpen(RuntimeError):
    def __init__(self, existing: ConnectionOperation):
        super().__init__(f"session {existing.session_key!r} already has operation {existing.op_id} open")
        self.existing = existing


_open: Dict[str, ConnectionOperation] = {}
_lock = threading.Lock()


def open(operation: ConnectionOperation) -> None:  # noqa: A001 - the verb is the API
    with _lock:
        existing = _open.get(operation.session_key)
        if existing is not None and not existing.settled:
            raise OperationAlreadyOpen(existing)
        _open[operation.session_key] = operation


def current(session_key: str) -> Optional[ConnectionOperation]:
    with _lock:
        operation = _open.get(session_key)
    return operation if operation is not None and not operation.settled else None


def get(session_key: str, op_id: str) -> Optional[ConnectionOperation]:
    with _lock:
        operation = _open.get(session_key)
    return operation if operation is not None and operation.op_id == op_id else None


def close(operation: ConnectionOperation) -> None:
    with _lock:
        if _open.get(operation.session_key) is operation:
            del _open[operation.session_key]


def reset_for_tests() -> None:
    with _lock:
        _open.clear()
