"""``pre_prompt_submit``: the plugin gate for a prompt the user typed in Desktop, the dashboard chat or the Ink TUI.

The messaging gateway's ``pre_gateway_dispatch`` hands plugins the live runner, its adapters and session store;
none of those exist here, so this surface has its own hook with plain values, an ``ask`` callable (one
``clarify`` card on this session) and a small set of directives the host carries out itself. It fires on the
prompt.submit worker thread once the agent is ready and before the turn starts: the session is already busy
(a second submit queues), the user row is already durable, and the thread may block on the user's answer.

Not a ``bind_module`` sibling: it late-imports ``server`` (``server.py`` is past its size cap, and nothing here
needs to close over its globals at import time).
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

HOOK = "pre_prompt_submit"
_MODEL_SCOPES = {"once": "--once", "session": "--session"}


def _server():
    from tui_gateway import server
    return server


def _context_tokens(agent, history: list) -> int:
    """The conversation's prompt size: the last API-reported figure, else a rough estimate (a resumed chat
    has made no call yet in this process, and that is exactly the chat a size-based gate cares about)."""
    reported = int(getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", 0) or 0)
    if reported > 0 or not history:
        return reported
    from agent.model_metadata import estimate_request_tokens_rough
    return int(estimate_request_tokens_rough(
        history, system_prompt=getattr(agent, "_cached_system_prompt", "") or "",
        tools=getattr(agent, "tools", None) or None))


def _last_message_at(history: list) -> float | None:
    """Timestamp of the newest message before this prompt (DB-loaded and flushed messages carry one)."""
    for msg in reversed(history):
        ts = msg.get("timestamp") if isinstance(msg, dict) else None
        if isinstance(ts, (int, float)) and ts > 0:
            return float(ts)
    return None


def _make_ask(sid: str):
    def ask(question: str, choices: list[str], timeout: float | None = None) -> str | None:
        """One single-question clarify card; the chosen label, or None (timeout, cancel, no client)."""
        server = _server()
        from tui_gateway import server_requests
        labels = [str(c) for c in choices or [] if str(c).strip()]
        params = {"questions": [{"qid": "q1", "question": str(question), "choices": labels or None,
                                 "multi_select": False}]}
        wait = server._clarify_timeout_seconds() if timeout is None else timeout
        result = server_requests.send("clarify", sid, params, timeout=wait, qids=["q1"])
        answer = ((result or {}).get("answers") or {}).get("q1")
        return answer if isinstance(answer, str) and answer.strip() else None
    return ask


def _hook_kwargs(sid: str, session: dict, text: str) -> dict[str, Any]:
    server = _server()
    agent = session.get("agent")
    with session["history_lock"]:
        history = list(session.get("history") or [])
    return {
        "session_key": str(session.get("session_key") or ""),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "text": text,
        "platform": server._session_source(session),
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
        "base_url": str(getattr(agent, "base_url", "") or ""),
        "context_tokens": _context_tokens(agent, history),
        "last_message_at": _last_message_at(history),
        "ask": _make_ask(sid),
    }


def _directive(results: list) -> dict | None:
    """First result that is not a plain allow, in registration order (the gateway hook's rule)."""
    for result in results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action == "skip":
            return result
        if action == "allow" and (result.get("model") or result.get("compress") or result.get("reply")):
            return result
    return None


def _notify(sid: str, text: str, level: str = "info") -> None:
    if text and text.strip():
        _server()._emit("notification.show", sid, {
            "text": text.strip(), "level": level, "kind": "ttl", "ttl_ms": 12000,
            "key": f"{HOOK}.reply", "id": f"{HOOK}.reply"})


def _end_without_turn(rid, sid: str, session: dict, reply: str, level: str = "info") -> None:
    """Close the admitted turn with no agent run: the user row stays (as after a cancel before the agent
    was ready), the busy edge closes, and a prompt queued while the hook waited drains."""
    server = _server()
    with session["history_lock"]:
        session["running"] = False
        session["last_active"] = time.time()
        server._clear_inflight_turn(session)
    _notify(sid, reply, level)
    server._emit("session.info", sid, server._session_info(session.get("agent"), session))
    server._drain_queued_prompt(rid, sid, session)


def _apply_model(sid: str, session: dict, spec: Any) -> str | None:
    """Switch through the same path ``/model`` takes; returns an error text, or None on success."""
    if not isinstance(spec, dict) or not spec.get("model"):
        return "the plugin named no model"
    flag = _MODEL_SCOPES.get(str(spec.get("scope") or "session"))
    if flag is None:
        return f"unknown model scope {spec.get('scope')!r}"
    raw = f"{spec['model']} --provider {spec['provider']} {flag}" if spec.get("provider") else f"{spec['model']} {flag}"
    try:
        result = _server()._apply_model_switch(sid, session, raw)
    except Exception as exc:  # health: allow BLE001 -- any switch failure becomes the user's notice, never a dead turn thread
        logger.warning("%s: model switch to %s failed", HOOK, raw, exc_info=True)
        return str(exc) or type(exc).__name__
    if result.get("confirm_required"):
        return (result.get("confirm_message") or "the switch needs confirmation").split("\n", 1)[0]
    return None


def _compress(sid: str, session: dict, text: str) -> str | None:
    """Compact on this (turn) thread, which already owns ``running``; ``/compress``'s own route refuses
    while a turn is admitted. The user row written at submit is retired first (as the queued-prompt drain
    does): compaction adopts every durable row, so it would fold this prompt into the compacted history and
    the turn would then send it a second time. The turn writes the row afresh, into whichever session is
    live after a rotating compaction; a failed compaction writes it back so the transcript keeps it."""
    server = _server()
    from agent.conversation_compression import finalize_context_engine_compression_notification
    agent = session.get("agent")
    retired = _retire_staged_user_row(session)
    try:
        server._compress_session_history(session, None)
    except Exception as exc:  # health: allow BLE001 -- any compaction failure becomes the user's notice; the row is written back
        logger.warning("%s: compaction before the turn failed", HOOK, exc_info=True)
        if not isinstance(exc, server.CompressionLockHeld):
            finalize_context_engine_compression_notification(agent, committed=False)
        if retired:
            server._persist_submit_user_row(session, text, None)
        return "another compaction holds this chat" if isinstance(exc, server.CompressionLockHeld) else (
            str(exc) or type(exc).__name__)
    server._sync_session_key_after_compress(sid, session)
    finalize_context_engine_compression_notification(agent, committed=True)
    server._emit("session.info", sid, server._session_info(agent, session))
    return None


def _retire_staged_user_row(session: dict) -> bool:
    staged = session.pop("_submit_user_row", None)
    if not isinstance(staged, dict) or not isinstance(staged.get("_row_id"), int):
        return False
    server = _server()
    with server._session_db(session) as db:
        if db is None:
            return False
        db.deactivate_message(server._submit_row_owner_key(staged, session), staged["_row_id"])
    return True


def _carry_out(rid, sid: str, session: dict, directive: dict, text: str) -> bool:
    if directive.get("action") == "skip":
        _end_without_turn(rid, sid, session, str(directive.get("reply") or ""))
        return False
    if directive.get("compress") and (err := _compress(sid, session, text)):
        _end_without_turn(rid, sid, session, f"Compaction failed ({err}); your message was not run.", "warn")
        return False
    if directive.get("model") and (err := _apply_model(sid, session, directive["model"])):
        _end_without_turn(rid, sid, session, f"Model switch failed ({err}); your message was not run.", "warn")
        return False
    _notify(sid, str(directive.get("reply") or ""))
    return True


def run_pre_prompt_submit(rid, sid: str, session: dict, text: Any) -> bool:
    """Run the hook for one user prompt; True when the turn should start. Fails open: a raising callback
    or an unreadable result runs the prompt as typed."""
    if not isinstance(text, str) or not text.strip():
        return True
    from hermes_cli.lifecycle import has_hook, invoke_hook
    if not has_hook(HOOK):
        return True
    server = _server()
    tokens = server._set_session_context(str(session.get("session_key") or ""), ui_session_id=sid)
    try:
        with server._session_profile_runtime_scope(session):
            try:
                directive = _directive(invoke_hook(HOOK, **_hook_kwargs(sid, session, text)))
            except Exception:
                logger.warning("%s invocation failed; running the prompt", HOOK, exc_info=True)
                return True
            if session.get("_turn_cancel_requested"):
                _end_without_turn(rid, sid, session, "")
                return False
            return True if directive is None else _carry_out(rid, sid, session, directive, text)
    finally:
        server._clear_session_context(tokens)
