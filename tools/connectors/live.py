"""Live operations, one open per (profile, session), found by ``op_id``. The RPC layer reads and
drives operations through here; the tool thread that minted one closes it on settle.

The profile is part of the key: two multiplexed profiles can carry the same timestamp-based
session key, and each side (the tool thread under the turn's profile override, the RPC under the
session's profile scope) resolves it through ``hermes_home_key``."""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from hermes_constants import get_process_hermes_home, hermes_home_key
from tools.connectors.operation import ConnectionOperation


class OperationAlreadyOpen(RuntimeError):
    def __init__(self, existing: ConnectionOperation):
        super().__init__(f"session {existing.session_key!r} already has operation {existing.op_id} open")
        self.existing = existing


_open: Dict[Tuple[str, str], ConnectionOperation] = {}
_lock = threading.Lock()


def _profile_key(profile_home: Optional[str]) -> str:
    """A session record names its profile home only for a non-default profile; the tool thread sees
    the same home through its turn override, and the default profile through the process home."""
    return hermes_home_key(profile_home or get_process_hermes_home())


def _key(session_key: str, profile_home: Optional[str]) -> Tuple[str, str]:
    return _profile_key(profile_home), session_key


def open(operation: ConnectionOperation) -> None:  # noqa: A001 - the verb is the API
    operation.profile_key = hermes_home_key()
    key = (operation.profile_key, operation.session_key)
    with _lock:
        existing = _open.get(key)
        if existing is not None and not existing.settled:
            raise OperationAlreadyOpen(existing)
        _open[key] = operation


def current(session_key: str, *, profile_home: Optional[str] = None) -> Optional[ConnectionOperation]:
    with _lock:
        operation = _open.get(_key(session_key, profile_home))
    return operation if operation is not None and not operation.settled else None


def get(session_key: str, op_id: str, *, profile_home: Optional[str] = None) -> Optional[ConnectionOperation]:
    with _lock:
        operation = _open.get(_key(session_key, profile_home))
    return operation if operation is not None and operation.op_id == op_id else None


def get_by_op_id(op_id: str, *, profile_home: Optional[str] = None) -> Optional[ConnectionOperation]:
    profile_key = _profile_key(profile_home)
    with _lock:
        return next((operation for (key, _), operation in _open.items()
                     if key == profile_key and operation.op_id == op_id and not operation.settled), None)


def find_target(name: str, *, profile_home: Optional[str] = None) -> Optional[ConnectionOperation]:
    profile_key = _profile_key(profile_home)
    with _lock:
        return next((operation for (key, _), operation in _open.items()
                     if key == profile_key and not operation.settled
                     and (target := operation.target(name)) is not None and target.kind == "connector"), None)


def close(operation: ConnectionOperation) -> None:
    key = (operation.profile_key, operation.session_key)
    with _lock:
        if _open.get(key) is operation:
            del _open[key]


def reset_for_tests() -> None:
    with _lock:
        _open.clear()
