"""The terminal result of a batch clarify, shared by every registry that holds one.

A batch (``clarify`` server request with question ids) is answered per question with ``clarify.lock``
and ends one of four ways: the last lock, a closing response frame, a timeout, or an interrupt/cancel.
Main's TUI sidecar (``tui_gateway/server_requests.py``) and the session owner
(``gateway/session_server_requests.py``) both settle batches; each used to compute ``{answers, outcome}``
itself and they drifted. Both call :func:`settle_batch`, so the model is told the same thing on every
surface.
"""
from __future__ import annotations

OUTCOMES = ("submitted", "timed_out", "cancelled")


def settle_batch(locked: dict, frame_result=None, *, ended: str | None = None) -> dict:
    """``{answers, outcome}`` for a batch whose locks so far are *locked*.

    - ``ended`` (``"timed_out"`` / ``"cancelled"``): the wait ended with no frame (timeout, interrupt,
      a ``cancel()``), or ``"submitted"`` for the last lock. The locked answers stand.
    - Otherwise *frame_result* is the closing frame's ``result``. A dict with an ``answers`` dict lays
      those answers over the locks (the frame wins per question) and keeps the frame's own outcome when it
      names a known one (main's classic CLI reports ``timed_out``), else ``submitted``. Anything else (Ink
      Esc / Desktop Skip send ``{}``; an error frame gives ``None``) is a cancel-all: locks kept, ``cancelled``.
    """
    if ended is not None:
        return {"answers": dict(locked), "outcome": ended if ended in OUTCOMES else "cancelled"}
    if isinstance(frame_result, dict) and isinstance(frame_result.get("answers"), dict):
        outcome = frame_result.get("outcome")
        return {**frame_result, "answers": {**locked, **frame_result["answers"]},
                "outcome": outcome if outcome in OUTCOMES else "submitted"}
    return {"answers": dict(locked), "outcome": "cancelled"}
