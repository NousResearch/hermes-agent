"""End-to-end: the TUI gateway delivers a leftover /steer as the next turn.

A /steer that arrives during a text-only final answer is broken out of the
stream (see agent/chat_completion_helpers.py) and handed back by the turn
finalizer in result["pending_steer"] (see agent/turn_finalizer.py). The
gateway path re-fires it as the next user turn (gateway/run.py). The TUI
consumes the SAME run_conversation() result, so it must re-fire it too — the
shared stream break would otherwise truncate the answer AND drop the steer on
the TUI, since the finalizer already drained the slot.

Drives ``_run_prompt_submit`` end to end via ``handle_request`` with the
``_ImmediateThread`` pattern so the leftover-steer re-fire (which calls
``_run_prompt_submit`` again) runs inline within a single request.
"""

import threading
import types

import pytest

from tui_gateway import server


def _tui_session(agent, session_key="sid-key", **extra):
    return {
        "agent": agent,
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "pending_title": None,
        **extra,
    }


class _ImmediateThread:
    """Run the turn worker synchronously so the whole re-fire chain unwinds
    inside the handle_request() call."""

    def __init__(self, target=None, daemon=None, **kw):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


def _patch_common(monkeypatch):
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *a, **kw: None
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)


def test_leftover_steer_delivered_as_next_turn(monkeypatch):
    """First turn comes back with pending_steer → a second turn fires
    automatically carrying the steer text."""
    prompts = []

    class _Agent:
        session_id = "sid-key"
        _cached_system_prompt = ""

        def run_conversation(self, prompt, **kw):
            prompts.append(prompt)
            result = {
                "final_response": "answer",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "answer"},
                ],
            }
            # Only the first (text-only-answer) turn leaves a leftover steer.
            if len(prompts) == 1:
                result["pending_steer"] = "check the logs instead"
            return result

    session = _tui_session(_Agent())
    _patch_common(monkeypatch)
    server._sessions["sid"] = session
    try:
        server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "write the summary"},
        })

        assert prompts == ["write the summary", "check the logs instead"], (
            "the leftover steer must be delivered as the immediate next turn, "
            "not dropped"
        )
        # The chain settles idle (no third turn, session released).
        assert session["running"] is False
    finally:
        server._sessions.pop("sid", None)


def test_queued_user_prompt_outranks_leftover_steer(monkeypatch):
    """A real user message that queued mid-turn wins over the leftover steer
    (mirrors gateway/run.py, which only delivers pending_steer when nothing
    else is pending). The steer is not re-fired."""
    prompts = []

    class _Agent:
        session_id = "sid-key"
        _cached_system_prompt = ""

        def __init__(self):
            self._queued = False

        def run_conversation(self, prompt, **kw):
            prompts.append(prompt)
            result = {
                "final_response": "answer",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "answer"},
                ],
            }
            if len(prompts) == 1:
                # A user prompt lands mid-turn and is queued (as _handle_busy_submit
                # would), AND the turn also comes back with a leftover steer.
                server._enqueue_prompt(_session_ref["s"], "user typed this", None)
                result["pending_steer"] = "check the logs instead"
            return result

    _session_ref = {}
    session = _tui_session(_Agent())
    _session_ref["s"] = session
    _patch_common(monkeypatch)
    server._sessions["sid"] = session
    try:
        server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "write the summary"},
        })

        assert prompts == ["write the summary", "user typed this"], (
            "the queued user prompt must win; the leftover steer is dropped "
            "(never re-fired) when a real message is waiting"
        )
        assert "check the logs instead" not in prompts
        assert session.get("queued_prompt") is None
        assert session["running"] is False
    finally:
        server._sessions.pop("sid", None)
