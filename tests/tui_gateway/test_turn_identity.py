"""Client-turn identity: ``prompt.submit.client_turn_id`` and ``turn.status``.

The contract is the mobile/desktop client's: an opaque client-minted id makes a submit retry-safe
(one run, one ack, one server turn id) and ``turn.status`` reports what the server knows. These
tests drive the REAL handler, the REAL turn thread and the REAL registry — a first submit, a
replay, a different-text reuse, the legacy ack, invalid ids, a finished failure, the unknown echo,
and the registry's bounds. Nothing here reads source text.
"""

from __future__ import annotations

import contextlib
import threading
import time
import types

import pytest

from tui_gateway import server, turn_identity

SID = "turn-idem-sid"
KEY = "turn-idem-key"


class _InlineThread:
    """Run every turn synchronously so the test observes the turn's final state, not a race."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra) -> dict:
    return {
        "agent": agent, "session_key": KEY, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0,
        "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all", **extra}


def _agent(turns_run: list, result: dict | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        session_id=KEY, clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: turns_run.append(a) or (
            result if result is not None else {"final_response": "done"}))


def _ready() -> threading.Event:
    ready = threading.Event()
    ready.set()
    return ready


@pytest.fixture()
def registry(monkeypatch):
    """A fresh registry per test; the process-wide one must never leak between cases."""
    fresh = turn_identity.TurnRegistry()
    monkeypatch.setattr(turn_identity, "REGISTRY", fresh)
    return fresh


@pytest.fixture()
def sessions(monkeypatch, tmp_path):
    """The live-session table plus the environment-heavy side paths of the turn pipeline.

    Storage is neutralized OUT, not faked for the identity logic under test: this suite is about
    the turn's identity and outcome, and a durable transcript needs its own fixtures. Everything
    else — admission, the turn thread, ``run_body`` — is the production code path.
    """
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_db", lambda session: contextlib.nullcontext(None))
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: True)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
    monkeypatch.setattr(server, "_persist_submit_user_row", lambda session, text, display_kind: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    live: list[dict] = []
    yield live
    for session in live:
        server._release_active_session_slot(session)
    server._sessions.pop(SID, None)


def _install(sessions, agent=None, **extra) -> dict:
    session = _session(agent=agent, agent_ready=_ready(), **extra)
    sessions.append(session)
    server._sessions[SID] = session
    return session


def _submit(**params) -> dict:
    return server.handle_request(
        {"id": "r", "method": "prompt.submit", "params": {"session_id": SID, **params}})


def _status(client_turn_id: str, session_id: str = SID) -> dict:
    return server.handle_request(
        {"id": "s", "method": "turn.status",
         "params": {"session_id": session_id, "client_turn_id": client_turn_id}})


def test_first_submit_acknowledges_a_new_turn(registry, sessions):
    turns: list = []
    _install(sessions, _agent(turns))

    ack = _submit(text="hello", client_turn_id="ct-1")["result"]

    assert ack["status"] == "streaming"
    assert ack["accepted"] is True
    assert ack["client_turn_id"] == "ct-1"
    assert isinstance(ack["server_turn_id"], str) and ack["server_turn_id"]
    assert ack["state"] == "accepted"
    assert ack["duplicate"] is False
    assert ack["replayed"] is False
    assert ack["idempotent"] is True
    assert "outcome" not in ack

    assert len(turns) == 1

    report = _status("ct-1")["result"]
    assert report == {"known": True, "client_turn_id": "ct-1",
                      "server_turn_id": ack["server_turn_id"], "state": "terminal"}


def test_replay_answers_the_stored_ack_without_running_again(registry, sessions):
    turns: list = []
    _install(sessions, _agent(turns))
    first = _submit(text="hello", client_turn_id="ct-replay")["result"]

    second = _submit(text="hello", client_turn_id="ct-replay")["result"]

    assert len(turns) == 1, "a replay must not start a second agent turn"
    assert second["status"] == "replayed"
    assert second["accepted"] is True
    assert second["client_turn_id"] == "ct-replay"
    assert second["server_turn_id"] == first["server_turn_id"]
    assert second["duplicate"] is True
    assert second["replayed"] is True
    assert second["idempotent"] is True
    # The replay reports what actually happened, not what was asked.
    assert second["state"] == "terminal"
    assert second["outcome"] == "succeeded"


def test_same_id_with_different_text_is_a_new_submission(registry, sessions):
    turns: list = []
    _install(sessions, _agent(turns))
    first = _submit(text="hello", client_turn_id="ct-reuse")["result"]

    second = _submit(text="something else", client_turn_id="ct-reuse")["result"]

    assert len(turns) == 2, "different words under the same id are not a retry"
    assert second["replayed"] is False
    assert second["duplicate"] is False
    assert second["state"] == "accepted"
    assert second["server_turn_id"] != first["server_turn_id"]


def test_a_retry_while_the_prompt_waits_in_the_queue_replays(registry, sessions):
    """The retry-after-a-drop case that matters most: the first submit is queued behind a live turn."""
    turns: list = []
    session = _install(sessions, _agent(turns), running=True)
    session["inflight_turn"] = {"user": "the live turn", "assistant": "partial"}

    first = _submit(text="hello", client_turn_id="ct-queued", queued=True)["result"]

    assert first["status"] == "queued"
    assert first["state"] == "accepted"
    assert session["queued_prompt"]["client_turn_id"] == "ct-queued"

    again = _submit(text="hello", client_turn_id="ct-queued", queued=True)["result"]

    assert again["status"] == "replayed"
    assert again["replayed"] is True and again["duplicate"] is True
    assert again["server_turn_id"] == first["server_turn_id"]
    assert again["state"] == "accepted", "nothing has run yet: the message is still queued"
    assert session.get("queued_prompts") is None, "a replay must not enqueue a second copy"
    assert turns == []


def test_a_redirected_submit_ends_with_the_turn_it_joined(registry, sessions):
    """A steer/redirect has no turn of its own: its id ends when the live turn does."""
    turns: list = []
    agent = _agent(turns)
    agent._supports_active_turn_redirect = True
    agent.redirect = lambda text: True
    session = _install(sessions, agent, running=True)
    session["inflight_turn"] = {"user": "the live turn", "assistant": "partial"}

    ack = _submit(text="actually, the other one", client_turn_id="ct-ride")["result"]

    assert ack["status"] == "redirected"
    assert ack["state"] == "running", ack
    assert session["_active_client_turns"][0].client_turn_id == "ct-ride"

    # The turn loop's finally, with the outcome of the turn the correction joined.
    server._finish_client_turn(session, "cancelled")

    report = _status("ct-ride")["result"]
    assert report["state"] == "cancelled"
    assert report["server_turn_id"] == ack["server_turn_id"]


def test_absent_client_turn_id_keeps_the_legacy_ack(registry, sessions):
    turns: list = []
    session = _install(sessions, _agent(turns))

    result = _submit(text="hello")["result"]

    assert result == {"status": "streaming"}, "the pre-idempotency ack is byte-identical"
    assert len(turns) == 1
    assert len(registry) == 0


@pytest.mark.parametrize("bad", ["x" * 181, 123, "", ["ct"]])
def test_invalid_client_turn_id_is_invalid_params(registry, sessions, bad):
    turns: list = []
    _install(sessions, _agent(turns))

    response = _submit(text="hello", client_turn_id=bad)

    assert response["error"]["code"] == 4001, response
    assert "client_turn_id" in response["error"]["message"]
    assert turns == []


def test_a_failed_turn_replays_as_terminal_failed(registry, sessions):
    turns: list = []
    _install(sessions, _agent(turns, {"final_response": "", "error": "provider exploded",
                                      "failed": True}))
    first = _submit(text="hello", client_turn_id="ct-fail")["result"]

    replay = _submit(text="hello", client_turn_id="ct-fail")["result"]

    assert len(turns) == 1
    assert replay["server_turn_id"] == first["server_turn_id"]
    assert replay["state"] == "terminal"
    assert replay["outcome"] == "failed"
    # ``turn.status`` reports the failure in its own vocabulary, not as a bare terminal.
    assert _status("ct-fail")["result"]["state"] == "failed"


def test_turn_status_echoes_an_unknown_id(registry, sessions):
    _install(sessions)

    report = _status("never-seen")["result"]

    assert report == {"known": False, "client_turn_id": "never-seen"}
    # A runtime session this process no longer holds is "known: false", not an error.
    gone = _status("never-seen", session_id="reaped-sid")["result"]
    assert gone == {"known": False, "client_turn_id": "never-seen"}


def test_turn_status_requires_a_valid_id(registry, sessions):
    _install(sessions)

    assert _status("")["error"]["code"] == 4001
    assert _status("y" * 181)["error"]["code"] == 4001


def test_registry_caps_entries_and_drops_the_oldest():
    bounded = turn_identity.TurnRegistry(max_entries=2)
    for index in range(3):
        bounded.begin(profile="p", session_id="s", client_turn_id=f"ct-{index}", text="same")
    assert len(bounded) == 2
    assert bounded.lookup(profile="p", session_id="s", client_turn_id="ct-0") is None
    assert bounded.lookup(profile="p", session_id="s", client_turn_id="ct-2") is not None


def test_registry_expires_a_turn_after_its_ttl():
    expiring = turn_identity.TurnRegistry(ttl_seconds=0.02)
    expiring.begin(profile="p", session_id="s", client_turn_id="ct-old", text="same")
    time.sleep(0.05)
    expiring.begin(profile="p", session_id="s", client_turn_id="ct-new", text="same")
    assert expiring.lookup(profile="p", session_id="s", client_turn_id="ct-old") is None
    assert len(expiring) == 1


def test_a_replaced_turn_cannot_be_finished_by_its_old_thread():
    """Same id + different text replaces the record: the older run's ending must not rewrite it."""
    bounded = turn_identity.TurnRegistry()
    old, _ = bounded.begin(profile="p", session_id="s", client_turn_id="ct", text="first words")
    new, replayed = bounded.begin(profile="p", session_id="s", client_turn_id="ct", text="other words")
    assert replayed is False and new is not old

    assert turn_identity.TurnRegistry.finish(bounded, old, turn_identity.SUCCEEDED) is False
    stored = bounded.lookup(profile="p", session_id="s", client_turn_id="ct")
    assert stored is new and stored.state == turn_identity.ACCEPTED
