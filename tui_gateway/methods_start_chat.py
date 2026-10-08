"""``session.start_chat``: the handoff card's Retry of a rejected start_chat call, and the next-turn note that tells
the model about it.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

import json
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

# Two clicks (two windows) must not both find the call unretried and start two chats.
_start_chat_retry_lock = threading.Lock()


@method("session.start_chat")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tui_gateway.start_chat import _rejected, start_chat
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    tool_call_id = str(params.get("tool_call_id") or "")
    if not tool_call_id:
        return _err(rid, 4002, "tool_call_id is required")
    # A running turn owns its own retries (the setup skill retries a profile rejection itself).
    if session.get("running"):
        return _ok(rid, json.loads(_rejected("This chat is still working; retry when its turn ends.", retryable=True)))
    with _start_chat_retry_lock, _session_db(session) as db:
        found = None if db is None else db.tool_row_retry(str(session.get("session_key") or ""), tool_call_id)
        if found is None:
            return _err(rid, 4004, "no saved start_chat result for that tool_call_id")
        row_id, retried = found
        if retried is not None:
            return _ok(rid, retried)
        outcome = json.loads(start_chat(params.get("args") or {}, params["session_id"]))
        if outcome["status"] == "started":
            db.set_tool_row_retry(row_id, outcome)
    return _ok(rid, outcome)


def _pending_tool_retry_notes(session: dict) -> str:
    """Note block for card retries since the last turn (model input only, announced once), or ""."""
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return ""
    try:
        with _session_db(session) as db:
            pending = [] if db is None else db.take_unannounced_tool_retries(session_key)
    except Exception:
        logger.debug("Failed to read pending tool retries", exc_info=True)
        return ""
    return "\n".join(
        f"[The user pressed Retry on your rejected {entry['tool_name']} call. {entry['result'].get('message') or ''}]"
        for entry in pending)


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
