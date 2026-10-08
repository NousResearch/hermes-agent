"""Session ``/yolo``: the one toggle / restore / carry contract every surface calls.

The bypass itself is ``tools.approval``'s in-memory per-key set, so every process starts with it empty.
Each surface persists it where its session identity lives: CLI, TUI and Desktop on the session row
(``model_config.yolo_mode``), the messaging gateway on the routing entry, whose key outlives compression
rotations; sessions the gateway owns for CLI/TUI/Desktop clients (``config.set key=yolo``) and their managed
workers use that same routing entry. The surface hands its writer in as ``persist(enabled)``; everything else
is decided here, so the CLI ``/yolo``, the TUI Shift+Tab, the Desktop zap and a Telegram ``/yolo`` behave the same.
"""

import logging
from typing import Any, Callable, Optional

from tools import approval

logger = logging.getLogger(__name__)

Persist = Optional[Callable[[bool], Any]]


def _persist(persist: Persist, enabled: bool, session_key: str) -> None:
    # Best-effort: the live flag is authoritative for this process; a failed write only costs the restore.
    if persist is None:
        return
    try:
        persist(enabled)
    except Exception:
        logger.warning("failed to persist session yolo=%s for %s", enabled, session_key, exc_info=True)


def toggle_session_yolo(session_key: str, enabled: Optional[bool] = None, *, persisted: bool = False,
                        persist: Persist = None) -> bool:
    """Flip (``enabled=None``) or set the bypass for *session_key*; returns the new state.

    *persisted* is the stored flag: after a restart only it is set and the user still sees ON, so a toggle
    turns it OFF. The write runs BEFORE the live flip, so a turn restoring in between never revives a
    bypass that is being switched off.
    """
    if enabled is None:
        enabled = not (persisted or approval.is_session_yolo_enabled(session_key))
    _persist(persist, enabled, session_key)
    (approval.enable_session_yolo if enabled else approval.disable_session_yolo)(session_key)
    return enabled


def restore_session_yolo(session_key: str, persisted: bool) -> bool:
    """Re-arm a persisted bypass in a fresh process; True when this call turned it on.

    Key it on the id approvals are checked under (the stored session id / routing key), never a
    transport-local id. Skipped under the frozen process ``--yolo``, which already bypasses everything.
    """
    if (not session_key or not persisted or approval._YOLO_MODE_FROZEN
            or approval.is_session_yolo_enabled(session_key)):
        return False
    approval.enable_session_yolo(session_key)
    return True


def apply_launch_yolo(session_key: str) -> None:
    """Seed a gateway-owned session's launch ``--yolo`` policy once per session boundary.

    Called every turn of such a route; only the first call (or the first after ``clear_session``) arms it,
    so a later ``/yolo`` off sticks instead of lasting one turn. Skipped under the frozen process ``--yolo``.
    """
    if not session_key or approval._YOLO_MODE_FROZEN:
        return
    with approval._lock:
        if session_key in approval._launch_yolo_applied:
            return
        approval._launch_yolo_applied.add(session_key)
    approval.enable_session_yolo(session_key)


def restore_gateway_yolo(session_key: str, persisted: bool) -> None:
    """Restore the authoritative routing value, including OFF, before frozen launch seeding."""
    if not session_key:
        return
    with approval._lock:
        if session_key in approval._launch_yolo_applied:
            return  # after restoration, the live toggle remains authoritative in this process
        approval._launch_yolo_applied.add(session_key)
        (approval._session_yolo.add if persisted else approval._session_yolo.discard)(session_key)


def transfer_session_yolo(old_key: str, new_key: str) -> None:
    """Move the live bypass when the conversation continues under a new id (compression rotation, /branch).
    The new row records it at creation (``with_session_yolo``), so nothing is written here."""
    if not old_key or not new_key or old_key == new_key or not approval.is_session_yolo_enabled(old_key):
        return
    approval.enable_session_yolo(new_key)
    approval.disable_session_yolo(old_key)


def with_session_yolo(model_config: Optional[dict], session_key: str) -> Optional[dict]:
    """``model_config`` for a session row being created, carrying a live bypass. Rows are created lazily on
    the first turn, so this is where a toggle made before the row existed gets recorded."""
    if not approval.is_session_yolo_enabled(session_key):
        return model_config
    return {**(model_config or {}), "yolo_mode": True}
