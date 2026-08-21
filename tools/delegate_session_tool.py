"""Persistent external-agent delegation sessions.

`delegate_task` is intentionally a Hermes->Hermes child-agent primitive.  This
module provides the complementary session primitive for external agents whose
native protocol already owns conversation state.  Pi is the first backend:
Hermes keeps one `pi --mode rpc` process/session alive and can send follow-up
turns, steer an active turn, answer Pi extension questions, inspect messages,
and stop/resume the native Pi session without wrapping it in a child AIAgent.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agent.pi_rpc_client import PiRPCClient, pending_question_for_owner
from agent.runtime_cwd import resolve_agent_cwd
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_SESSION_LOCK = threading.RLock()
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_MAX_TEXT = 12_000


def check_delegate_session_requirements() -> bool:
    return bool(shutil.which("pi") or Path.home().joinpath(".local", "bin", "pi").is_file())


def _owner_key(parent_agent: Any) -> str:
    durable = str(getattr(parent_agent, "session_id", "") or "").strip()
    return durable or f"agent:{id(parent_agent)}"


def _bounded(value: Any, maximum: int = _MAX_TEXT) -> str:
    text = str(value or "")
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def _pending_payload(client: PiRPCClient) -> dict[str, Any] | None:
    question = pending_question_for_owner(client)
    if question is None:
        return None
    return {
        "method": question.method,
        "question": _bounded(question.title, 2000),
        "options": list(question.options)[:50],
        "created_at": question.created_at,
    }


def _summary(record: Dict[str, Any], *, include_result: bool = True) -> dict[str, Any]:
    client: PiRPCClient = record["client"]
    out = {
        "session_id": record["session_id"],
        "pi_session_id": record.get("pi_session_id") or record["session_id"],
        "status": record.get("status", "unknown"),
        "cwd": record.get("cwd"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "pending_question": _pending_payload(client),
        "error": record.get("error") or None,
    }
    if include_result and record.get("last_result"):
        result = record["last_result"]
        out["last_result"] = {
            "text": _bounded(result.get("text")),
            "duration_s": result.get("duration_s"),
        }
    return out


def _lookup(session_id: str, parent_agent: Any) -> Dict[str, Any] | None:
    owner = _owner_key(parent_agent)
    with _SESSION_LOCK:
        record = _SESSIONS.get(session_id)
        if record is None or record.get("owner") != owner:
            return None
        return record


def _run_turn(record: Dict[str, Any], message: str, timeout: float) -> None:
    client: PiRPCClient = record["client"]
    with _SESSION_LOCK:
        if record.get("status") == "closed":
            return
        record["status"] = "running"
        record["error"] = ""
        record["updated_at"] = time.time()
    try:
        result = client.run_session_prompt(message, timeout_seconds=timeout)
        state = result.get("state") if isinstance(result, dict) else {}
        with _SESSION_LOCK:
            record["last_result"] = result
            record["pi_session_id"] = (
                state.get("sessionId") if isinstance(state, dict) else None
            ) or record.get("pi_session_id") or record["session_id"]
            if record.get("status") != "closed":
                record["status"] = "idle"
            record["updated_at"] = time.time()
    except Exception as exc:  # noqa: BLE001 - surfaced as bounded session state
        logger.exception("Pi delegate session %s turn failed", record.get("session_id"))
        with _SESSION_LOCK:
            if record.get("status") != "closed":
                record["status"] = "error"
            record["error"] = _bounded(exc, 2000)
            record["updated_at"] = time.time()


def _dispatch_turn(record: Dict[str, Any], message: str, timeout: float) -> None:
    thread = threading.Thread(
        target=_run_turn,
        args=(record, message, timeout),
        name=f"pi-delegate-{record['session_id'][:8]}",
        daemon=True,
    )
    with _SESSION_LOCK:
        record["thread"] = thread
        record["status"] = "running"
        record["updated_at"] = time.time()
    thread.start()


def _initial_prompt(goal: str, context: str | None) -> str:
    policy = (
        "[Delegation policy] You are a coding delegate operating in a persistent "
        "session owned by Hermes. Work directly in the current working tree. Do "
        "not commit or push unless Hermes explicitly asks you to. Ask questions "
        "when a decision is genuinely required; Hermes can answer through the "
        "same delegate session."
    )
    parts = [policy]
    if context and context.strip():
        parts.append("Context from Hermes:\n" + context.strip())
    if goal and goal.strip():
        parts.append("Task:\n" + goal.strip())
    return "\n\n".join(parts)


def delegate_session(
    *,
    action: str = "start",
    session_id: Optional[str] = None,
    goal: Optional[str] = None,
    context: Optional[str] = None,
    message: Optional[str] = None,
    timeout: Optional[int] = None,
    parent_agent: Any = None,
) -> str:
    """Create/control a persistent Pi delegation session."""
    if parent_agent is None:
        return tool_error("delegate_session requires a parent agent context.")

    normalized = (action or "start").strip().lower()
    if normalized not in {"start", "resume", "send", "steer", "status", "messages", "list", "stop"}:
        return tool_error(
            "Unknown action. Use start, resume, send, steer, status, messages, list, or stop."
        )
    effective_timeout = float(max(10, min(int(timeout or 900), 3600)))
    owner = _owner_key(parent_agent)

    if normalized == "list":
        with _SESSION_LOCK:
            rows = [
                _summary(record, include_result=False)
                for record in _SESSIONS.values()
                if record.get("owner") == owner
            ]
        return json.dumps({"success": True, "sessions": rows}, ensure_ascii=False)

    if normalized in {"start", "resume"}:
        requested_id = (session_id or "").strip()
        if normalized == "resume" and not requested_id:
            return tool_error("action='resume' requires session_id.")
        handle = requested_id or str(uuid.uuid4())
        with _SESSION_LOCK:
            existing = _SESSIONS.get(handle)
            if existing is not None:
                if existing.get("owner") != owner:
                    return tool_error("That delegate session belongs to another conversation.")
                return json.dumps({"success": True, "reused": True, **_summary(existing)}, ensure_ascii=False)

        cwd = str(resolve_agent_cwd().resolve())
        client = PiRPCClient(
            persistent_session=True,
            session_id=handle,
            session_name=f"Hermes {handle[:8]}",
            acp_cwd=cwd,
        )
        try:
            state = client.start(timeout=min(30.0, effective_timeout))
        except Exception as exc:  # noqa: BLE001
            client.close()
            return tool_error(f"Could not start Pi delegate session: {_bounded(exc, 1000)}")
        native_id = str(state.get("sessionId") or handle)
        now = time.time()
        record: Dict[str, Any] = {
            "session_id": handle,
            "pi_session_id": native_id,
            "owner": owner,
            "cwd": cwd,
            "client": client,
            "status": "idle",
            "created_at": now,
            "updated_at": now,
            "last_result": None,
            "error": "",
            "thread": None,
        }
        with _SESSION_LOCK:
            _SESSIONS[handle] = record
        if goal and goal.strip():
            _dispatch_turn(record, _initial_prompt(goal, context), effective_timeout)
        return json.dumps({"success": True, "created": True, **_summary(record)}, ensure_ascii=False)

    if not session_id or not session_id.strip():
        return tool_error(f"action='{normalized}' requires session_id.")
    record = _lookup(session_id.strip(), parent_agent)
    if record is None:
        return tool_error("Delegate session not found for this conversation. Use action='resume' to reopen a native Pi session.")
    client: PiRPCClient = record["client"]

    if normalized == "status":
        proc = getattr(client, "_proc", None)
        if record.get("status") not in {"closed", "error"} and proc is not None and proc.poll() is not None:
            with _SESSION_LOCK:
                record["status"] = "error"
                record["error"] = f"Pi RPC process exited with code {proc.returncode}"
                record["updated_at"] = time.time()
        return json.dumps({"success": True, **_summary(record)}, ensure_ascii=False)

    if normalized == "messages":
        try:
            messages = client.get_messages(timeout=min(30.0, effective_timeout))
        except Exception as exc:  # noqa: BLE001
            return tool_error(f"Could not read Pi session messages: {_bounded(exc, 1000)}")
        # Keep the tool result bounded while preserving the newest conversational state.
        safe = messages[-40:]
        encoded = json.dumps(safe, ensure_ascii=False, default=str)
        if len(encoded) > 40_000:
            encoded = encoded[-40_000:]
        return json.dumps({"success": True, "session_id": record["session_id"], "messages_json": encoded}, ensure_ascii=False)

    if normalized == "send":
        text = (message or goal or "").strip()
        if not text:
            return tool_error("action='send' requires message.")
        with _SESSION_LOCK:
            if record.get("status") == "running":
                return tool_error("Pi session is currently running. Use action='steer' to redirect it, or wait for idle.")
            if record.get("status") == "closed":
                return tool_error("Pi session is closed. Use action='resume' to reopen it.")
        _dispatch_turn(record, text, effective_timeout)
        return json.dumps({"success": True, "accepted": True, **_summary(record, include_result=False)}, ensure_ascii=False)

    if normalized == "steer":
        text = (message or "").strip()
        if not text:
            return tool_error("action='steer' requires message.")
        with _SESSION_LOCK:
            if record.get("status") != "running":
                return tool_error("Pi session is not running. Use action='send' for a new follow-up turn.")
        try:
            response = client.steer(text, timeout=min(30.0, effective_timeout))
        except Exception as exc:  # noqa: BLE001
            return tool_error(f"Could not steer Pi session: {_bounded(exc, 1000)}")
        return json.dumps({"success": True, "response": response, **_summary(record, include_result=False)}, ensure_ascii=False, default=str)

    if normalized == "stop":
        try:
            if record.get("status") == "running":
                client.abort(timeout=min(10.0, effective_timeout))
        except Exception:
            logger.debug("Pi delegate abort failed before close", exc_info=True)
        client.close()
        with _SESSION_LOCK:
            record["status"] = "closed"
            record["updated_at"] = time.time()
        return json.dumps({"success": True, "closed": True, **_summary(record)}, ensure_ascii=False)

    return tool_error("Unhandled delegate_session action.")


DELEGATE_SESSION_SCHEMA = {
    "name": "delegate_session",
    "description": (
        "Delegate coding work to Pi through a persistent native RPC session. "
        "Use this instead of delegate_task when the worker should be Pi. The Pi "
        "conversation survives across turns: start a session, send follow-ups, "
        "steer a running turn (including answering Pi questions), inspect status "
        "or messages, and stop/resume the same native Pi session. delegate_task "
        "remains the Hermes child-agent primitive."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "resume", "send", "steer", "status", "messages", "list", "stop"],
                "description": "Session lifecycle/control action. Omit for start.",
            },
            "session_id": {
                "type": "string",
                "description": "Persistent Pi delegation/session id returned by start. Required for all actions except start/list.",
            },
            "goal": {
                "type": "string",
                "description": "Initial coding objective for action='start'. The turn runs asynchronously in the persistent Pi session.",
            },
            "context": {
                "type": "string",
                "description": "Initial background/context passed with goal when starting the Pi session.",
            },
            "message": {
                "type": "string",
                "description": "Follow-up for send, or live course correction/question answer for steer.",
            },
            "timeout": {
                "type": "integer",
                "minimum": 10,
                "maximum": 3600,
                "description": "Maximum seconds allowed for each Pi turn (default 900).",
            },
        },
        "required": [],
    },
}


registry.register(
    name="delegate_session",
    toolset="delegation",
    schema=DELEGATE_SESSION_SCHEMA,
    handler=lambda args, **kw: delegate_session(
        action=args.get("action") or "start",
        session_id=args.get("session_id"),
        goal=args.get("goal"),
        context=args.get("context"),
        message=args.get("message"),
        timeout=args.get("timeout"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_session_requirements,
    emoji="🔁",
)
