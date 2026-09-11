"""A relayed dm into a live Bot Chat keeps its sender all the way into ``run_conversation``.

The relay handler stamps the author as an in-process ``DeliveryAuthor``; ``prompt.submit`` accepts only
that object, so a dashboard client cannot claim a bot identity through the same RPC.
"""

from __future__ import annotations

import threading
import types

import pytest

from tools.bot_relay import DeliveryAuthor
from tui_gateway import server as srv

AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}
OTHER = {"id": "bot:writer", "name": "writer", "is_bot": True}


def _result(resp):
    return resp["result"] if "result" in resp else resp


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "gw-session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
        "transport": None, **extra,
    }


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    for name in ("_emit", "_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_tts_stream_begin", "_sync_session_key_after_compress"):
        monkeypatch.setattr(srv, name, lambda *a, **k: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda agent: {})


def test_live_relay_stamps_the_sender_as_a_delivery_author(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    submitted = []
    monkeypatch.setitem(srv._methods, "prompt.submit", lambda rid, p: submitted.append(p) or srv._ok(rid, {"status": "streaming"}))
    monkeypatch.setattr(srv, "_profile_home", lambda name: home / "profiles" / name)
    monkeypatch.setitem(srv._sessions, "live-ops",
                        {"profile_home": str(home / "profiles" / "ops"), "pending_title": "Bot Chat", "history": []})

    _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping", "from_profile": "coder", "from_handle": "coder"}))

    assert submitted == [{"session_id": "live-ops", "text": "ping", "queued": True, "_turn_author": DeliveryAuthor(AUTHOR)}]


def test_prompt_submit_refuses_a_client_supplied_author(monkeypatch):
    srv._sessions["sid"] = _session()
    try:
        resp = srv._methods["prompt.submit"]("r", {"session_id": "sid", "text": "hi", "_turn_author": dict(AUTHOR)})
    finally:
        srv._sessions.pop("sid", None)
    assert resp["error"]["code"] == 4124


def test_busy_relay_dms_queue_with_their_authors_and_drain_with_them(monkeypatch):
    dispatched = []
    monkeypatch.setattr(srv, "_run_prompt_submit", lambda rid, sid, _s, text, **kw: dispatched.append((text, kw)))
    monkeypatch.setattr(srv, "_interrupt_busy_session", lambda *a, **k: None)
    session = _session(agent=types.SimpleNamespace(interrupt=lambda: None), running=True)
    srv._sessions["sid"] = session
    try:
        for text, author in (("ping", AUTHOR), ("hello", OTHER), ("human note", None)):
            params = {"session_id": "sid", "text": text, "queued": True}
            if author:
                params["_turn_author"] = DeliveryAuthor(author)
            assert _result(srv._methods["prompt.submit"]("r", params)) == {"status": "queued"}
        # Authored envelopes never merge with each other or with the human's text.
        assert session["queued_prompt"]["turn_author"] == AUTHOR
        assert [e["text"] for e in session["queued_prompts"]] == ["hello", "human note"]
        assert session["queued_prompts"][0]["turn_author"] == OTHER
        assert "turn_author" not in session["queued_prompts"][1]
        for _ in range(3):
            session["running"] = False
            assert srv._drain_queued_prompt("d", "sid", session) is True
    finally:
        srv._sessions.pop("sid", None)
    assert [(t, kw.get("turn_author")) for t, kw in dispatched] == [("ping", AUTHOR), ("hello", OTHER), ("human note", None)]


def test_turn_runner_passes_the_author_only_when_set_and_only_to_an_agent_that_declares_it(turn_env, monkeypatch):
    from tui_gateway import wisdom_mediation

    seen = []
    activity = []
    monkeypatch.setattr(wisdom_mediation, "note_activity", lambda session, **kwargs: activity.append(session))

    def accepting(user_message, *, turn_author="not passed", **kwargs):
        seen.append(turn_author)
        return {"final_response": "ok"}

    def legacy(user_message, conversation_history=None, stream_callback=None, persist_user_message=None, task_id=None):
        seen.append("legacy called")
        return {"final_response": "ok"}

    for fn, author in ((accepting, AUTHOR), (accepting, None), (legacy, AUTHOR)):
        agent = types.SimpleNamespace(session_id="a", run_conversation=fn, clear_interrupt=lambda: None)
        session = _session(agent=agent, running=True)
        session["_wisdom_activity_tracking"] = True
        srv._run_prompt_submit("rid", "ui-sid", session, "ping", turn_author=author)
        assert session["_wisdom_user_activity"] > 0
        assert activity[-1] is session

    assert seen == [AUTHOR, "not passed", "legacy called"]
    assert len(activity) == 3


def test_a_human_prompt_after_a_relayed_dm_runs_without_an_author(turn_env, monkeypatch):
    """The author rides on the queued entry and the run call, never on the session, so the human prompt that
    follows a drained relayed dm reaches ``run_conversation`` unattributed."""
    seen = []

    def run_conversation(user_message, *, turn_author="not passed", **kwargs):
        seen.append((user_message, turn_author))
        return {"final_response": "ok"}

    monkeypatch.setattr(srv, "_interrupt_busy_session", lambda *a, **k: None)
    agent = types.SimpleNamespace(session_id="a", run_conversation=run_conversation, clear_interrupt=lambda: None,
                                  interrupt=lambda: None)
    session = _session(agent=agent, running=True)
    srv._sessions["sid"] = session
    try:
        params = {"session_id": "sid", "text": "ping", "queued": True, "_turn_author": DeliveryAuthor(AUTHOR)}
        assert _result(srv._methods["prompt.submit"]("r", params)) == {"status": "queued"}
        session["running"] = False
        assert srv._drain_queued_prompt("d", "sid", session) is True
        session["running"] = True
        srv._run_prompt_submit("r2", "sid", session, "human note")
    finally:
        srv._sessions.pop("sid", None)

    assert seen == [("ping", AUTHOR), ("human note", "not passed")]
    assert "turn_author" not in session and not session.get("queued_prompt")
