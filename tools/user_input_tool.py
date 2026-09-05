"""Hermes-native non-blocking structured user-input tools.

``clarify`` remains the synchronous, in-turn prompt. These tools are for work
that should continue while the user decides: create a durable request, emit a
request event through the active surface, and return a handle immediately.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error

MAX_QUESTIONS = 5
MAX_OPTIONS = 4
MAX_TEXT_CHARS = 2000
MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
DEFAULT_TIMEOUT_SECONDS = 3600.0

# Test-only injection point. Production callers must pass the owning agent's DB
# through the inline executor/dispatcher; this remains None in normal runtime.
_shared_session_db = None


def _db_or_shared(session_db):
    return session_db if session_db is not None else _shared_session_db


def _text(value: Any, *, field: str, max_chars: int = MAX_TEXT_CHARS, required: bool = True) -> str:
    if not isinstance(value, str):
        if required:
            raise ValueError(f"{field} must be a string")
        return ""
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field} must not be blank")
    if len(value) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return value


def normalize_questions(questions: Any) -> List[Dict[str, Any]]:
    """Validate and normalize the public question payload."""
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"questions must contain 1-{MAX_QUESTIONS} entries")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"questions cannot contain more than {MAX_QUESTIONS} entries")

    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, raw in enumerate(questions):
        if not isinstance(raw, dict):
            raise ValueError(f"questions[{index}] must be an object")
        question_id = _text(raw.get("id"), field=f"questions[{index}].id", max_chars=128)
        if question_id in seen_ids:
            raise ValueError(f"question id {question_id!r} is duplicated")
        seen_ids.add(question_id)
        text = _text(raw.get("text"), field=f"questions[{index}].text")
        raw_options = raw.get("options", [])
        if raw_options is None:
            raw_options = []
        if not isinstance(raw_options, list):
            raise ValueError(f"questions[{index}].options must be an array")
        if len(raw_options) > MAX_OPTIONS:
            raise ValueError(f"questions[{index}].options cannot contain more than {MAX_OPTIONS} entries")
        options = []
        for option_index, option in enumerate(raw_options):
            option_text = _text(
                option,
                field=f"questions[{index}].options[{option_index}]",
                max_chars=MAX_TEXT_CHARS,
            )
            if option_text in options:
                raise ValueError(f"questions[{index}].options contains a duplicate")
            options.append(option_text)
        allow_free_text = raw.get("allow_free_text", False)
        if not isinstance(allow_free_text, bool):
            raise ValueError(f"questions[{index}].allow_free_text must be a boolean")
        default = raw.get("default", "")
        if default is None:
            default = ""
        if not isinstance(default, (str, int, float, bool, list, dict)):
            raise ValueError(f"questions[{index}].default must be JSON-compatible")
        if options and not allow_free_text and default not in options:
            raise ValueError(f"questions[{index}].default must match one of its options")
        normalized.append({
            "id": question_id,
            "text": text,
            "options": options,
            "allow_free_text": allow_free_text,
            "default": default,
        })
    return normalized


def _timeout_seconds(timeout_s: Any) -> float:
    if timeout_s is None:
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout_s, bool):
        raise ValueError("timeout_s must be a number")
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_s must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError("timeout_s must be finite")
    # <= 0 is the documented no-expiry sentinel, useful for a request that is
    # owned by an operator and explicitly cancelled later.
    if timeout <= 0:
        return 0.0
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_s cannot exceed {MAX_TIMEOUT_SECONDS} seconds")
    return timeout


def _public_request(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": record["request_id"],
        "session_id": record["session_id"],
        "turn_id": record.get("turn_id", ""),
        "questions": record["questions"],
        "context": record.get("context", ""),
        "status": record.get("status", "pending"),
        "answer": record.get("answer"),
        "expires_at": record.get("expires_at", 0),
    }


def _notify(callback: Optional[Callable], payload: Dict[str, Any]) -> None:
    if not callable(callback):
        return
    try:
        callback(payload)
    except Exception:
        # The request is already durable. A broken renderer must not turn a
        # successful model tool call into a failed turn.
        import logging
        logging.getLogger(__name__).debug("user-input request callback failed", exc_info=True)


def request_user_input(
    questions: Any,
    context: str = "",
    timeout_s: Any = DEFAULT_TIMEOUT_SECONDS,
    *,
    session_id: str = "",
    turn_id: str = "",
    session_db=None,
    session_key: str = "",
    source: str = "",
    user_id: str = "",
    chat_id: str = "",
    thread_id: str = "",
    request_id: str = "",
    callback: Optional[Callable] = None,
    event_callback: Optional[Callable] = None,
    now: Optional[float] = None,
) -> str:
    """Create a durable structured request and return without waiting.

    The keyword-only fields after ``timeout_s`` are runtime context supplied by
    Hermes's dispatcher. They are intentionally absent from the public schema.
    """
    try:
        normalized = normalize_questions(questions)
        session_id = _text(session_id, field="session_id", max_chars=256)
        context = _text(context, field="context", max_chars=4096, required=False)
        timeout = _timeout_seconds(timeout_s)
        db = _db_or_shared(session_db)
        if db is None:
            return tool_error("request_user_input requires an active Hermes session")
        request_id = _text(request_id, field="request_id", max_chars=256, required=False) or uuid.uuid4().hex
        now_value = float(now) if now is not None else __import__("time").time()
        expires_at = 0.0 if timeout <= 0 else now_value + timeout
        record = db.create_pending_user_input(
            request_id=request_id,
            session_id=session_id,
            questions=normalized,
            context=context,
            expires_at=expires_at,
            turn_id=turn_id,
            session_key=session_key,
            source=source,
            user_id=user_id,
            chat_id=chat_id,
            thread_id=thread_id,
            now=now_value,
        )
        public = _public_request(record)
        public["hint"] = (
            "The request is pending. Continue your work; Hermes will deliver the answer "
            "through this same turn when the user responds."
        )
        _notify(callback or event_callback, public)
        return json.dumps(public, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


def check_user_input(
    request_id: str,
    *,
    session_id: str = "",
    session_db=None,
    now: Optional[float] = None,
) -> str:
    """Read the durable state of one request, scoped to its owning session."""
    db = _db_or_shared(session_db)
    if db is None:
        return tool_error("check_user_input requires an active Hermes session")
    try:
        request_id = _text(request_id, field="request_id", max_chars=256)
        session_id = _text(session_id, field="session_id", max_chars=256)
        record = db.get_pending_user_input(request_id, session_id=session_id, now=now)
        if record is None:
            return json.dumps({"status": "not_found", "request_id": request_id}, ensure_ascii=False)
        return json.dumps(_public_request(record), ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


def list_pending_user_inputs(session_id: str, *, session_db=None, now: Optional[float] = None) -> list:
    """Return durable pending records for one session (used by reconnecting UIs)."""
    db = _db_or_shared(session_db)
    if db is None:
        return []
    return [_public_request(item) for item in db.list_pending_user_inputs(session_id, now=now)]


def answer_user_input(
    request_id: str,
    answer: Dict[str, Any],
    *,
    session_id: str,
    session_db=None,
    turn_id: Optional[str] = None,
    agent=None,
) -> Dict[str, Any]:
    """Answer one request and deliver it through the live agent when present."""
    db = _db_or_shared(session_db)
    if db is None:
        return {"status": "not_found", "accepted": False}
    result = db.answer_pending_user_input(
        request_id, answer, session_id=session_id, turn_id=turn_id
    )
    if result.get("accepted") and agent is not None:
        result["delivery"] = deliver_answer_to_agent(agent, result.get("turn_id", turn_id or ""), answer)
    return result


def deliver_answer_to_agent(agent, turn_id: str, answer: Dict[str, Any]) -> str:
    """Use the existing role-safe live-turn boundary; never append transcript rows."""
    if not agent or not turn_id or getattr(agent, "_current_turn_id", "") != turn_id:
        return "deferred"
    in_flight = getattr(agent, "_inflight_turn_id", None)
    if hasattr(agent, "_inflight_turn_id") and in_flight != turn_id:
        return "deferred"
    if getattr(agent, "_interrupt_requested", False):
        return "deferred"
    text = json.dumps(answer, ensure_ascii=False)
    model_active = getattr(agent, "_model_request_active", None)
    if model_active is not None and model_active.is_set():
        redirect = getattr(agent, "redirect", None)
        if callable(redirect) and redirect(text):
            return "redirected"
    if getattr(agent, "_executing_tools", False):
        steer = getattr(agent, "steer", None)
        if callable(steer) and steer(text):
            return "steered"
    # Between API calls the normal steer queue is still the safest hand-off. The
    # conversation loop will consume it at its existing tool-result boundary.
    steer = getattr(agent, "steer", None)
    if callable(steer) and steer(text):
        return "queued"
    return "deferred"


def check_user_input_requirements() -> bool:
    return True


REQUEST_USER_INPUT_SCHEMA = {
    "name": "request_user_input",
    "description": (
        "Create a durable, non-blocking Hermes user-input request. Use this when the agent can "
        "continue working while the user decides. The tool returns a request_id immediately; "
        "the answer is delivered through the same active Hermes turn. Pass 1-5 independent "
        "questions in one call. Each question requires a stable id and text, may include up to "
        "4 options, allow_free_text, and a default. Do not use this for secrets or dangerous "
        "command confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array", "minItems": 1, "maxItems": MAX_QUESTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "options": {"type": "array", "maxItems": MAX_OPTIONS, "items": {"type": "string"}},
                        "allow_free_text": {"type": "boolean"},
                        "default": {},
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            },
            "context": {"type": "string", "description": "Why the decision is needed (optional)."},
            "timeout_s": {
                "type": "number", "minimum": 0, "maximum": MAX_TIMEOUT_SECONDS,
                "description": "Seconds before expiry; 0 means no expiry (default: 3600).",
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}

CHECK_USER_INPUT_SCHEMA = {
    "name": "check_user_input",
    "description": (
        "Check a previously created Hermes user-input request by request_id. "
        "The result is scoped to the current session and includes pending, answered, expired, or not_found status."
    ),
    "parameters": {
        "type": "object",
        "properties": {"request_id": {"type": "string"}},
        "required": ["request_id"],
        "additionalProperties": False,
    },
}


registry.register(
    name="request_user_input",
    toolset="clarify",
    schema=REQUEST_USER_INPUT_SCHEMA,
    handler=lambda args, **kw: request_user_input(
        questions=args.get("questions"),
        context=args.get("context", ""),
        timeout_s=args.get("timeout_s", DEFAULT_TIMEOUT_SECONDS),
        session_id=kw.get("session_id", ""),
        turn_id=kw.get("turn_id", ""),
        session_db=kw.get("session_db"),
        session_key=kw.get("session_key", ""),
        source=kw.get("source", ""),
        user_id=kw.get("user_id", ""),
        chat_id=kw.get("chat_id", ""),
        thread_id=kw.get("thread_id", ""),
        callback=kw.get("callback") or kw.get("event_callback"),
    ),
    check_fn=check_user_input_requirements,
    emoji="📝",
)
registry.register(
    name="check_user_input",
    toolset="clarify",
    schema=CHECK_USER_INPUT_SCHEMA,
    handler=lambda args, **kw: check_user_input(
        args.get("request_id", ""),
        session_id=kw.get("session_id", ""),
        session_db=kw.get("session_db"),
    ),
    check_fn=check_user_input_requirements,
    emoji="🔎",
)


__all__ = [
    "MAX_QUESTIONS", "MAX_OPTIONS", "REQUEST_USER_INPUT_SCHEMA", "CHECK_USER_INPUT_SCHEMA",
    "normalize_questions", "request_user_input", "check_user_input", "list_pending_user_inputs",
    "answer_user_input", "deliver_answer_to_agent",
]
