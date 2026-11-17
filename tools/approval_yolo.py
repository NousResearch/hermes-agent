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


#: What a refused caller is told when the refusing plugin gave no sentence of its own.
BYPASS_REFUSED = "Turning approvals off was refused by a plugin."


def _bypass_hook_kwargs(surface, scope, session_key, identity, platform, user_id, session_owner) -> dict:
    return {
        "surface": surface, "scope": scope, "session_key": session_key or "",
        "identity": {k: identity.get(k) for k in ("user_id", "provider")} if isinstance(identity, dict) else None,
        "platform": str(getattr(platform, "value", platform) or ""), "user_id": str(user_id or ""),
        "session_owner": str(session_owner or ""),
    }


def _bypass_refusal(results: list, default: str) -> Optional[str]:
    for result in results:
        if isinstance(result, dict) and result.get("action") == "block":
            message = result.get("message")
            return message.strip()[:500] if isinstance(message, str) and message.strip() else default
    return None


def approval_bypass_refusal(*, surface: str, scope: str = "session", session_key: str = "",
                            identity: Optional[dict] = None, platform: Any = None, user_id: Any = None,
                            session_owner: Any = None, default: str = BYPASS_REFUSED) -> Optional[str]:
    """Ask ``pre_approval_bypass`` plugins whether this caller may switch approvals off; the refusal
    sentence (the plugin's, else ``default``), or None to go ahead.

    Every surface asks before it turns a bypass ON or re-arms a stored one, and before anything is
    written; switching approvals back on is never asked. The caller is whoever the surface verified:
    ``identity`` (a connection's server-verified login, passed on as its ``user_id`` and ``provider``),
    a messaging sender (``platform``, ``user_id``) or, when re-arming a stored chat's bypass, that
    chat's stored owner (``session_owner``). Nobody named is the local operator."""
    from hermes_cli.plugins import invoke_hook
    return _bypass_refusal(invoke_hook("pre_approval_bypass", **_bypass_hook_kwargs(
        surface, scope, session_key, identity, platform, user_id, session_owner)), default)


async def approval_bypass_refusal_async(*, surface: str, scope: str = "session", session_key: str = "",
                                        identity: Optional[dict] = None, platform: Any = None, user_id: Any = None,
                                        session_owner: Any = None, default: str = BYPASS_REFUSED) -> Optional[str]:
    """:func:`approval_bypass_refusal` for callers on the gateway loop (``async def`` callbacks are awaited there)."""
    from hermes_cli.plugins import ainvoke_hook
    return _bypass_refusal(await ainvoke_hook("pre_approval_bypass", **_bypass_hook_kwargs(
        surface, scope, session_key, identity, platform, user_id, session_owner)), default)


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


def restore_stored_session_yolo(session_key: str, row: Optional[dict]) -> bool:
    """:func:`restore_session_yolo` from a stored chat's row, while its stored owner (the row's ``user_id``)
    may still switch approvals off (``pre_approval_bypass``); True when this call turned it on."""
    from hermes_state import SessionDB
    stored = SessionDB.session_yolo_enabled(row)
    return restore_session_yolo(session_key, stored and approval_bypass_refusal(
        surface="restore", session_key=session_key, session_owner=(row or {}).get("user_id")) is None)


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


def restore_gateway_yolo(session_key: str, persisted: Optional[bool], *, launch: bool = False) -> None:
    """Restore the authoritative routing value, including OFF, before frozen launch seeding.

    ``persisted`` None means nobody toggled this route: the ``--yolo`` launch policy (*launch*) is seeded
    once per boundary instead, so an untouched entry never reads as a revocation.
    """
    if not session_key:
        return
    if persisted is None:
        if launch:
            apply_launch_yolo(session_key)
        return
    with approval._lock:
        if session_key in approval._launch_yolo_applied:
            return  # after restoration, the live toggle remains authoritative in this process
        approval._launch_yolo_applied.add(session_key)
        (approval._session_yolo.add if persisted else approval._session_yolo.discard)(session_key)


def transfer_session_yolo(old_key: str, new_key: str) -> None:
    """Move the live bypass when the conversation continues under a new id (compression rotation, /branch).
    The new row records it at creation (``with_session_yolo``), so nothing is written here. Every
    continuation site calls this, so it also moves the chat's MCP spend total: a rotation must not hand a
    chat that spent its limit a fresh budget."""
    from tools.mcp_tool_costs import transfer_session
    transfer_session(old_key, new_key)
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
