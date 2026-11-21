"""A retried ``prompt.submit`` runs ONE turn: the client's ``submission_id`` is its identity.

A client that loses the reply to a submit (app relaunch, mobile network drop, reconnect mid-turn)
resends the same words. Nothing on this surface told the retry apart from a second intent, so the
session ran a second turn — a second reply and a second bill for one keystroke. The canonical
session runtime already keys its admission on exactly this id; this suite pins the same guarantee
on ``/api/ws``, plus the boundary the id must NOT cross (a different session, a different id).
"""

from __future__ import annotations

import threading

import pytest

from tui_gateway import prompt_submission_id, server


class _InlineThread:
    """Run the turn synchronously so the test observes the turn it caused."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _CountingAgent:
    """Counts the turns actually run, the one number a de-duplicated submit may not increment."""

    def __init__(self):
        self.session_id = "agent-sid"
        self.turns: list[str] = []
        self.interim_assistant_callback = None

    def run_conversation(self, *args, **kwargs):
        self.turns.append(kwargs.get("user_message") or (args[0] if args else ""))
        return {"final_response": "ok"}

    def clear_interrupt(self):
        pass


def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_update_session_cwd", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    prompt_submission_id._answers.clear()


def _new_session(monkeypatch):
    resp = server.handle_request(
        {"jsonrpc": "2.0", "id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    sid = resp["result"]["session_id"]
    session = server._sessions[sid]
    agent = _CountingAgent()
    if session.get("agent_ready") is not None:
        session["agent_ready"].set()  # the deferred build is stubbed out: no build will ever set it
    session["agent"] = agent
    return sid, agent


def _submit(sid, text, submission_id=None):
    params = {"session_id": sid, "text": text}
    if submission_id is not None:
        params["submission_id"] = submission_id
    return server.handle_request(
        {"jsonrpc": "2.0", "id": "r", "method": "prompt.submit", "params": params})


def test_resend_with_the_same_submission_id_answers_the_first_turn(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    sid, agent = _new_session(monkeypatch)
    try:
        first = _submit(sid, "summarise the thread", "client-1")
        assert "result" in first, first
        assert first["result"]["status"] == "streaming"
        assert "duplicate" not in first["result"] and agent.turns == ["summarise the thread"]

        retry = _submit(sid, "summarise the thread", "client-1")
        assert "result" in retry, retry
        assert retry["result"]["duplicate"] is True
        # The original acceptance is answered verbatim: same status, same row id when one was proven.
        assert retry["result"]["status"] == first["result"]["status"]
        assert retry["result"].get("user_row_id") == first["result"].get("user_row_id")
        assert agent.turns == ["summarise the thread"], "a retried submit must not run a second turn"
    finally:
        server._sessions.pop(sid, None)


def test_a_new_submission_id_is_a_new_turn(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    sid, agent = _new_session(monkeypatch)
    try:
        _submit(sid, "first", "client-1")
        second = _submit(sid, "second", "client-2")
        assert "result" in second, second
        assert "duplicate" not in second["result"]
        assert agent.turns == ["first", "second"]
    finally:
        server._sessions.pop(sid, None)


def test_the_same_submission_id_in_another_session_is_its_own_turn(monkeypatch, tmp_path):
    """The key is (session, id): a client that mints ids per conversation keeps working."""
    _env(monkeypatch, tmp_path)
    sid_a, agent_a = _new_session(monkeypatch)
    sid_b, agent_b = _new_session(monkeypatch)
    try:
        _submit(sid_a, "hello", "shared-1")
        other = _submit(sid_b, "hello", "shared-1")
        assert "result" in other, other
        assert "duplicate" not in other["result"]
        assert agent_a.turns == ["hello"] and agent_b.turns == ["hello"]
    finally:
        server._sessions.pop(sid_a, None)
        server._sessions.pop(sid_b, None)


def test_a_submit_without_a_submission_id_keeps_todays_behaviour(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    sid, agent = _new_session(monkeypatch)
    try:
        for _ in range(2):
            reply = _submit(sid, "repeat me")
            assert "result" in reply, reply
            assert "duplicate" not in reply["result"]
        assert agent.turns == ["repeat me", "repeat me"]
    finally:
        server._sessions.pop(sid, None)


def test_a_non_string_submission_id_is_refused_before_anything_runs(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    sid, agent = _new_session(monkeypatch)
    try:
        for bad in (123, "", "   ", {"id": "x"}):
            reply = _submit(sid, "text", bad)
            assert reply["error"]["code"] == 4000, (bad, reply)
            assert "submission_id" in reply["error"]["message"]
        assert agent.turns == []
    finally:
        server._sessions.pop(sid, None)


def test_a_submit_refused_at_the_door_is_not_remembered(monkeypatch, tmp_path):
    """Nothing was admitted, so the id stays free for the corrected send."""
    _env(monkeypatch, tmp_path)
    refused = _submit("no-such-session", "hello", "client-1")
    assert "error" in refused, refused
    assert prompt_submission_id.replay("no-such-session", "client-1") is None


@pytest.mark.parametrize("age, expected", [(0.0, True), (299.0, True), (301.0, False)])
def test_the_retry_window_expires(monkeypatch, age, expected):
    monkeypatch.setattr(prompt_submission_id.time, "monotonic", lambda: 1000.0)
    prompt_submission_id._answers.clear()
    prompt_submission_id.remember("sid", "client-1", {"status": "streaming", "user_row_id": 7})
    monkeypatch.setattr(prompt_submission_id.time, "monotonic", lambda: 1000.0 + age)
    replayed = prompt_submission_id.replay("sid", "client-1")
    assert (replayed is not None) is expected
    if expected:
        assert replayed == {"status": "streaming", "user_row_id": 7, "duplicate": True}


def test_the_registry_keeps_only_accepted_turns_and_stays_bounded(monkeypatch):
    prompt_submission_id._answers.clear()
    # The typed-stop-phrase reply admits nothing: it must never answer a later real submit.
    prompt_submission_id.remember("sid", "stop-1", {"voice_stopped": True})
    assert prompt_submission_id.replay("sid", "stop-1") is None
    for index in range(prompt_submission_id._MAX_ENTRIES + 10):
        prompt_submission_id.remember("sid", f"client-{index}", {"status": "streaming"})
    assert len(prompt_submission_id._answers) == prompt_submission_id._MAX_ENTRIES
    assert prompt_submission_id.replay("sid", "client-0") is None  # evicted oldest-first
    assert prompt_submission_id.replay("sid", f"client-{prompt_submission_id._MAX_ENTRIES + 9}") is not None
