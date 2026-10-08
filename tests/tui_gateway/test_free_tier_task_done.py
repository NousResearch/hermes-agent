"""A finished task on the free tier: a turn that completed on its own with a reply arms the sign-in
offer (``hermes_cli.free_tier_offer``); any user intervention while it ran disqualifies it."""

from __future__ import annotations

import threading
import types

import pytest

import tui_gateway.server as server
from hermes_cli import anon_auth, free_tier_offer
from tui_gateway.free_tier_task_done import note_task_done

CLEAN = {"completed": True, "final_response": "Here is your plan.", "user_intervened": False}


@pytest.fixture(autouse=True)
def free_tier(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(anon_auth, "has_guest", lambda: True)
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: True)
    monkeypatch.setattr(anon_auth, "is_anonymous_agent", lambda agent: getattr(agent, "anonymous", True))
    monkeypatch.setattr(free_tier_offer, "_clock", lambda: 5_000.0)


def _session(**extra):
    return {"agent": types.SimpleNamespace(anonymous=True), "session_key": "k", "history": [],
            "history_lock": threading.Lock(), "running": True, "attached_images": [], **extra}


def _recorded() -> bool:
    return free_tier_offer.offer_due_in() is not None


def test_a_clean_task_arms_the_offer():
    session = _session()
    note_task_done(session, dict(CLEAN), session["agent"], None)
    assert free_tier_offer.offer_due_in() == free_tier_offer.OFFER_DELAY_S


@pytest.mark.parametrize("result,anonymous,display_kind,user_input", [
    ({**CLEAN, "completed": False}, True, None, False),
    ({**CLEAN, "final_response": "  "}, True, None, False),
    ({**CLEAN, "pending_steer": "and the weather"}, True, None, False),
    ({**CLEAN, "user_intervened": True}, True, None, False),
    (CLEAN, True, None, True),
    (CLEAN, False, None, False),
    (CLEAN, True, "auto_continue", False),
], ids=["not-completed", "empty-reply", "pending-steer", "steered-or-interrupted", "typed-while-busy",
        "own-provider", "crash-resume"])
def test_disqualified_turns_record_nothing(result, anonymous, display_kind, user_input):
    session = _session(**({"_turn_user_input": True} if user_input else {}))
    session["agent"].anonymous = anonymous
    note_task_done(session, dict(result), session["agent"], display_kind)
    assert not _recorded()


@pytest.mark.parametrize("method,params", [
    ("session.steer", {"text": "also check my calendar"}),
    ("session.redirect", {"text": "actually only Monday"}),
], ids=["steer", "redirect"])
def test_a_correction_rpc_during_the_task_disqualifies_it(method, params):
    """The session-level mark covers agents whose own flag never reaches this process (compute host)."""
    agent = types.SimpleNamespace(anonymous=True, steer=lambda text: True, redirect=lambda text: True,
                                  _supports_active_turn_redirect=True)
    server._sessions["sid"] = session = _session(agent=agent)
    try:
        resp = server.handle_request({"id": "1", "method": method, "params": {"session_id": "sid", **params}})
        assert "result" in resp, resp
        note_task_done(session, dict(CLEAN), agent, None)
    finally:
        server._sessions.pop("sid", None)
    assert not _recorded()


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, **kw):
        self._target = target

    def start(self):
        self._target()


@pytest.mark.parametrize("touch,recorded_turns,records_at_each_complete", [
    (None, [1], [1]), ("steer", [], [0]), ("typed", [2], [0, 1]),
], ids=["untouched", "steered-over-rpc", "typed-while-busy"])
def test_prompt_submit_records_only_untouched_turns(monkeypatch, touch, recorded_turns, records_at_each_complete):
    """Typing into a running turn disqualifies that turn; the queued prompt then runs as its own task.
    Each record is written before its turn's message.complete goes out."""
    runs: list[str] = []
    recorded: list[int] = []
    monkeypatch.setattr(free_tier_offer, "record_task_done", lambda: recorded.append(len(runs)))

    class _Agent:
        anonymous = True
        session_id = "k"
        _cached_system_prompt = ""

        def steer(self, text):
            return True

        def run_conversation(self, prompt, **kw):
            runs.append(prompt)
            if touch == "steer":
                server.handle_request({"id": "s", "method": "session.steer",
                                       "params": {"session_id": "sid", "text": "and my calendar"}})
            if touch == "typed" and len(runs) == 1:
                resp = server.handle_request({"id": "t", "method": "prompt.submit",
                                              "params": {"session_id": "sid", "text": "and my calendar"}})
                assert resp["result"]["status"] == "queued", resp
            return {"final_response": "Here is your plan.", "completed": True, "user_intervened": False,
                    "messages": [{"role": "assistant", "content": "Here is your plan."}]}

    from unittest.mock import MagicMock
    monkeypatch.setattr(server, "_get_db", lambda: MagicMock())
    recorded_at_complete: list[int] = []

    def _emit(event, *a, **kw):
        if event == "message.complete":
            recorded_at_complete.append(len(recorded))
    monkeypatch.setattr(server, "_emit", _emit)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **kw: None)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(agent=_Agent(), running=False, history_version=0, image_counter=0, cols=80,
                       slash_worker=None, show_reasoning=False, tool_progress_mode="all", pending_title=None)
    server._sessions["sid"] = session
    try:
        resp = server.handle_request({"id": "1", "method": "prompt.submit",
                                      "params": {"session_id": "sid", "text": "Plan my week"}})
        assert "result" in resp, resp
    finally:
        server._sessions.pop("sid", None)
    assert recorded == recorded_turns
    # The desktop re-reads the offer on message.complete, so the record lands before that event.
    assert recorded_at_complete == records_at_each_complete
    assert "_turn_user_input" not in session  # per turn: never leaks into the next one
