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

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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


def pending_question(subagent_id: str) -> Optional[dict[str, Any]]:
    """The child's open question for listings, or None."""
    with _lock:
        entry = _questions.get(subagent_id)
        if entry is None:
            return None
        return {"request_id": entry.request_id, "question": entry.question, "state": entry.state,
                "options": entry.options, "waiting_seconds": round(time.time() - entry.asked_at, 1)}
