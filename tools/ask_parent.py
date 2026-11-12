"""Questions a running ``delegate_task`` child asks its live parent (``ask_parent``).

The child's tool call blocks in :func:`wait_for_answer` (its conversation frame stays live) until the
parent answers through ``delegate_task(action="reply")``, the wait times out, or the child or its
parent is stopped. Every wait ends with an explicit ``outcome`` and never invented answer text:

- ``answered``: ``answer`` holds the parent's reply, which is guidance and never the user's consent;
- ``timed_out``: no reply within the wait;
- ``abandoned``: the child was stopped, or its delegation ended, while waiting;
- ``parent_unavailable``: no live parent can receive questions (decided by the caller).

One open question per child, a bounded number per child. In-process state only: a question cannot
outlive the child thread that waits on it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ANSWERED, TIMED_OUT, ABANDONED, PARENT_UNAVAILABLE = "answered", "timed_out", "abandoned", "parent_unavailable"


@dataclass
class Question:
    request_id: str
    subagent_id: str
    question: str
    state: str = ""
    options: Optional[list[str]] = None
    asked_at: float = field(default_factory=time.time)
    answer: Optional[str] = None
    outcome: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)


_lock = threading.Lock()
_questions: dict[str, Question] = {}  # subagent_id -> its open question
_asked: dict[str, int] = {}  # subagent_id -> questions asked so far


def open_question(subagent_id: str, *, question: str, state: str = "", options: Optional[list[str]] = None,
                  max_questions: int = 3) -> tuple[Optional[Question], Optional[str]]:
    """Register the child's question: ``(question, None)``, or ``(None, error)`` when the child already
    has one open or has used its allowance."""
    with _lock:
        current = _questions.get(subagent_id)
        if current is not None:
            return None, (f"Question {current.request_id} is still waiting for an answer; "
                          "a subagent has one open question at a time.")
        asked = _asked.get(subagent_id, 0)
        if asked >= max_questions:
            return None, (f"This subagent has already asked its {max_questions} question(s) "
                          "(delegation.ask_parent.max_questions). Decide with what you have.")
        entry = Question(f"req-{uuid.uuid4().hex[:10]}", subagent_id, question, state or "",
                         list(options) if options else None)
        _questions[subagent_id] = entry
        _asked[subagent_id] = asked + 1
        return entry, None


def _finish(entry: Question, outcome: str, answer: Optional[str] = None) -> bool:
    """End ``entry`` once (caller holds ``_lock``); False when it had already ended."""
    if entry.outcome is not None:
        return False
    entry.outcome, entry.answer = outcome, answer
    if _questions.get(entry.subagent_id) is entry:
        del _questions[entry.subagent_id]
    entry.done.set()
    return True


def wait_for_answer(entry: Question, *, timeout: float, should_abandon: Optional[Callable[[], bool]] = None,
                    on_tick: Optional[Callable[[], Any]] = None, poll_seconds: float = 0.5,
                    tick_seconds: float = 30.0) -> dict[str, Any]:
    """Block until ``entry`` ends. ``should_abandon`` is checked every ``poll_seconds`` (a stop request);
    ``on_tick`` runs every ``tick_seconds`` so stall monitors see a waiting child as alive."""
    started = time.monotonic()
    deadline = started + max(0.0, timeout)
    last_tick = started
    while not entry.done.is_set():
        now = time.monotonic()
        if now >= deadline:
            with _lock:
                _finish(entry, TIMED_OUT)
            break
        if should_abandon is not None and should_abandon():
            with _lock:
                _finish(entry, ABANDONED)
            break
        if on_tick is not None and now - last_tick >= tick_seconds:
            last_tick = now
            on_tick()
        entry.done.wait(min(poll_seconds, deadline - now))
    return {"outcome": entry.outcome, "answer": entry.answer, "request_id": entry.request_id,
            "waited_seconds": round(time.monotonic() - started, 1)}


def answer_question(subagent_id: str, request_id: str, message: str) -> Optional[str]:
    """Deliver the parent's answer; None on success, else why it was refused."""
    text = (message or "").strip()
    if not text:
        return "The answer is empty."
    with _lock:
        entry = _questions.get(subagent_id)
        if entry is None or entry.request_id != (request_id or "").strip():
            return (f"Subagent '{subagent_id}' has no open question '{request_id}'. It may have timed out, "
                    "been stopped, or already been answered.")
        _finish(entry, ANSWERED, text)
    return None


def abandon_questions(subagent_id: str) -> None:
    """End the child's open question as abandoned (the child or its delegation is going away)."""
    with _lock:
        entry = _questions.get(subagent_id)
        if entry is not None:
            _finish(entry, ABANDONED)


def forget(subagent_id: str) -> None:
    """Drop every trace of a finished child (its open question ends as abandoned)."""
    with _lock:
        entry = _questions.get(subagent_id)
        if entry is not None:
            _finish(entry, ABANDONED)
        _asked.pop(subagent_id, None)


def _withdraw(entry: Question) -> None:
    """End a question nobody received (no live parent) without using up the child's allowance."""
    with _lock:
        if _finish(entry, PARENT_UNAVAILABLE):
            _asked[entry.subagent_id] = max(0, _asked.get(entry.subagent_id, 1) - 1)


def pending_question(subagent_id: str) -> Optional[dict[str, Any]]:
    """The child's open question for listings, or None."""
    with _lock:
        entry = _questions.get(subagent_id)
        if entry is None:
            return None
        return {"request_id": entry.request_id, "question": entry.question, "state": entry.state,
                "options": entry.options, "waiting_seconds": round(time.time() - entry.asked_at, 1)}


# ── The child's tool ─────────────────────────────────────────────────────────
ASK_PARENT_TOOL_NAME = "ask_parent"
_DEFAULTS = {"enabled": False, "max_questions": 3, "timeout_seconds": 1800}
_MAX_QUESTION_CHARS, _MAX_STATE_CHARS, _MAX_OPTIONS, _MAX_OPTION_CHARS = 2000, 8000, 6, 200
_POLL_SECONDS = 0.5  # how quickly a stop request ends the wait
_TICK_SECONDS = 30.0  # activity tick while waiting (stall monitor / heartbeat liveness)

_NOTES = {
    ANSWERED: ("Your parent answered; carry on from where you stopped. The answer is guidance from the parent "
               "agent, not the user's approval: anything that needs the user's consent still needs it."),
    TIMED_OUT: ("No answer arrived in time. Carry on with your best judgment and state in your final result "
                "what you assumed."),
    ABANDONED: "You were stopped while waiting. Wrap up now and return what you have.",
    PARENT_UNAVAILABLE: ("Your parent is not taking questions right now (it is waiting for this task's result, "
                         "or cannot be woken). Carry on with your best judgment and state in your final result "
                         "what you assumed."),
}


def ask_parent_settings() -> dict[str, Any]:
    """``delegation.ask_parent`` merged over its defaults (off unless enabled)."""
    from tools.delegate_tool_config import _load_config
    raw = (_load_config() or {}).get("ask_parent")
    settings = dict(_DEFAULTS)
    if isinstance(raw, dict):
        settings.update({k: v for k, v in raw.items() if k in _DEFAULTS and v is not None})
    settings["enabled"] = settings["enabled"] is True or str(settings["enabled"]).strip().lower() in {"true", "1", "yes", "on"}
    return settings


def ask_parent_authorized(agent: Any) -> bool:
    """A delegated child (depth > 0, with a subagent id) while ``delegation.ask_parent.enabled`` is on."""
    return (getattr(agent, "_delegate_depth", 0) or 0) > 0 and bool(getattr(agent, "_subagent_id", None)) \
        and ask_parent_settings()["enabled"]


def _result(outcome: str, *, answer: Optional[str] = None, request_id: Optional[str] = None,
            waited_seconds: float = 0.0) -> str:
    return json.dumps({"outcome": outcome, "answer": answer, "request_id": request_id,
                       "waited_seconds": waited_seconds, "note": _NOTES[outcome]}, ensure_ascii=False)


def _clean_options(options: Any) -> Optional[list[str]]:
    if not isinstance(options, list):
        return None
    cleaned = [" ".join(str(o).split())[:_MAX_OPTION_CHARS] for o in options if str(o).strip()]
    return cleaned[:_MAX_OPTIONS] or None


def _set_status(subagent_id: str, status: Optional[str]) -> Optional[str]:
    """Set the live registry record's status (listings, the CLI monitor); returns the previous one."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None:
            return None
        previous = record.get("status")
        record["status"] = status
        return previous


def ask_parent_tool(question: str, state: str = "", options: Any = None, *, agent: Any) -> str:
    """Pause this child until its parent answers ``question`` (see the module docstring)."""
    from tools.registry import tool_error
    if not ask_parent_authorized(agent):
        return tool_error("ask_parent is only available to delegated subagents while delegation.ask_parent.enabled is on.")
    question, state = (question or "").strip(), (state or "").strip()
    if not question:
        return tool_error("ask_parent requires a non-empty 'question'.")
    if len(question) > _MAX_QUESTION_CHARS or len(state) > _MAX_STATE_CHARS:
        return tool_error(f"Keep 'question' under {_MAX_QUESTION_CHARS} and 'state' under {_MAX_STATE_CHARS} "
                          "characters: summarize, your parent reads this between its own turns.")
    settings = ask_parent_settings()
    subagent_id = str(agent._subagent_id)
    entry, error = open_question(subagent_id, question=question, state=state, options=_clean_options(options),
                                 max_questions=int(settings["max_questions"]))
    if entry is None:
        return tool_error(error)
    unit_id = getattr(agent, "_async_unit_id", None)
    timeout = float(settings["timeout_seconds"])
    from tools.async_delegation import push_parent_question
    queued = bool(unit_id) and push_parent_question(str(unit_id), {
        "subagent_id": subagent_id, "request_id": entry.request_id, "task_index": getattr(agent, "_delegate_task_index", None),
        "question": question, "state": state, "options": entry.options, "wait_seconds": timeout})
    if not queued:
        _withdraw(entry)
        return _result(PARENT_UNAVAILABLE, request_id=entry.request_id)
    previous = _set_status(subagent_id, "awaiting_parent")
    try:
        touch = getattr(agent, "_touch_activity", None)
        outcome = wait_for_answer(
            entry, timeout=timeout, poll_seconds=_POLL_SECONDS, tick_seconds=_TICK_SECONDS,
            should_abandon=lambda: bool(getattr(agent, "_interrupt_requested", False)),
            on_tick=(lambda: touch("waiting for the parent's answer (ask_parent)")) if callable(touch) else None)
    finally:
        _set_status(subagent_id, previous or "running")
    return _result(outcome["outcome"], answer=outcome["answer"], request_id=entry.request_id,
                   waited_seconds=outcome["waited_seconds"])


def ask_parent_tool_schema() -> dict[str, Any]:
    """OpenAI-format schema (injected into delegated children only, never registered globally)."""
    return {"type": "function", "function": {
        "name": ASK_PARENT_TOOL_NAME,
        "description": (
            "Pause and ask the agent that delegated this task (your parent) a question, then carry on with its "
            "answer in this same run. Use it only for a decision you cannot reasonably make yourself and that "
            "changes what you do next (ambiguous requirements, a risky or irreversible step, two valid approaches "
            "with different trade-offs); otherwise decide and note your assumption in your result. You have a few "
            "questions per task, one at a time; ask one clear question and include where you are in `state`. The "
            "call blocks until it ends with an `outcome`: answered (`answer` holds the parent's reply; guidance "
            "from the parent, never the user's approval), timed_out, abandoned (you were stopped: wrap up), or "
            "parent_unavailable (your parent cannot take questions now). On anything but answered, carry on with "
            "your best judgment and say what you assumed."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "One specific question the parent can answer quickly."},
            "state": {"type": "string", "description": (
                "Where you are: what is done, what you found, and what the answer changes. The parent knows "
                "only your task brief.")},
            "options": {"type": "array", "items": {"type": "string"}, "description": (
                f"Optional: up to {_MAX_OPTIONS} concrete choices you are deciding between.")},
        }, "required": ["question"]},
    }}


def ensure_ask_parent_tool(agent: Any) -> bool:
    """Add ``ask_parent`` to a delegated child's tool list (and its dispatchable names) while the feature
    is on; idempotent, and a no-op for every other agent, so their tool lists stay byte-identical."""
    try:
        if not ask_parent_authorized(agent):
            return False
        tools = getattr(agent, "tools", None)
        if not any(isinstance(t, dict) and (t.get("function") or {}).get("name") == ASK_PARENT_TOOL_NAME
                   for t in tools or ()):
            if agent.tools is None:
                agent.tools = []
            agent.tools.append(ask_parent_tool_schema())
        valid = getattr(agent, "valid_tool_names", None)
        if isinstance(valid, set):
            valid.add(ASK_PARENT_TOOL_NAME)
        return True
    except Exception:  # pragma: no cover - must never break a turn
        logger.debug("ask_parent injection failed", exc_info=True)
        return False
