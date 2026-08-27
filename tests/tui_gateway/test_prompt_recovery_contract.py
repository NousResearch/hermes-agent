"""Hostile contracts for deferred prompts and failed-turn recovery."""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, **_extra):
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


class _BuildThread:
    def is_alive(self):
        return True


class _ReadyAfterTwoSlices:
    def __init__(self):
        self.calls = 0
        self.ready = False

    def wait(self, timeout=None):
        self.calls += 1
        self.ready = self.calls >= 2
        return self.ready

    def is_set(self):
        return self.ready


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "prompt-recovery-session",
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
        "inflight_turn": None,
        **extra,
    }


def test_patient_first_prompt_wait_outlives_old_30_second_cliff(monkeypatch):
    ready = _ReadyAfterTwoSlices()
    session = _session(
        agent_ready=ready,
        running=True,
        agent_error=None,
        _agent_build_thread=_BuildThread(),
    )
    events = []
    times = iter((0.0, 31.0))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, payload)))

    assert server._wait_agent_for_prompt(session, "rid", "sid") is None
    assert [event for event, _ in events] == [
        "notification.show",
        "notification.clear",
    ]
    assert events[0][1]["key"] == server._AGENT_BUILD_SLOW_NOTICE_KEY


def test_patient_first_prompt_wait_honors_cancel(monkeypatch):
    class _NeverReady:
        def wait(self, timeout=None):
            session["_turn_cancel_requested"] = True
            return False

        def is_set(self):
            return False

    session = _session(
        agent_ready=_NeverReady(),
        running=True,
        agent_error=None,
        _agent_build_thread=_BuildThread(),
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    assert server._wait_agent_for_prompt(session, "rid", "sid") is None


@pytest.fixture()
def emits(monkeypatch):
    captured = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda agent, session=None: {
            "model": getattr(agent, "model", "fake-model"),
            "provider": getattr(agent, "provider", "fake-provider"),
        },
    )
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)
    yield
    goals._DB_CACHE.clear()


def _events(captured, name):
    return [payload for event, _sid, payload in captured if event == name]


def test_returned_provider_error_is_terminal_and_replayable(emits, turn_env):
    agent = types.SimpleNamespace(
        session_id="prompt-recovery-session",
        model="fake-model",
        provider="fake-provider",
        run_conversation=lambda *args, **kwargs: {
            "final_response": "",
            "error": "provider 402: billing wall",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "error"
    assert completes[0]["error"] == "provider 402: billing wall"
    snapshot = server._inflight_snapshot(session)
    assert snapshot["user"] == "do the thing"
    assert snapshot["error"] == "provider 402: billing wall"
    assert snapshot["recoverable"] is True
    assert session["running"] is False


def test_exception_restores_agent_transcript_and_retains_partial(emits, turn_env):
    def _boom(message, stream_callback=None, **kwargs):
        if stream_callback is not None:
            stream_callback("half an answer")
        agent._session_messages = [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "half an answer"},
        ]
        raise RuntimeError("connection reset mid-stream")

    agent = types.SimpleNamespace(
        session_id="prompt-recovery-session",
        model="fake-model",
        provider="fake-provider",
        run_conversation=_boom,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "do the thing")

    server._run_prompt_submit("rid", "sid", session, "do the thing")

    completes = _events(emits, "message.complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "error"
    assert completes[0]["partial"] is True
    assert completes[0]["text"] == "half an answer"
    assert session["history"] == agent._session_messages
    assert session["history_version"] == 1
    assert server._inflight_snapshot(session)["error"] == "connection reset mid-stream"


def test_next_turn_replaces_retained_failure(emits, turn_env):
    seen = []

    def _run_ok(message, **kwargs):
        seen.append(server._inflight_snapshot(session)["user"])
        return {"final_response": "fresh answer"}

    agent = types.SimpleNamespace(
        session_id="prompt-recovery-session",
        model="fake-model",
        provider="fake-provider",
        run_conversation=_run_ok,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._start_inflight_turn(session, "old prompt")
    with session["history_lock"]:
        server._fail_inflight_turn(session, "previous failure")

    server._run_prompt_submit("rid", "sid", session, "new prompt")

    assert seen == ["new prompt"]
    assert server._inflight_snapshot(session) is None
    assert _events(emits, "message.complete")[-1]["status"] == "complete"


def test_live_resume_payload_exposes_retained_failure(monkeypatch):
    session = _session(running=False)
    server._start_inflight_turn(session, "lost terminal frame")
    with session["history_lock"]:
        server._fail_inflight_turn(session, "provider unavailable")
    monkeypatch.setattr(server, "_get_db", lambda: None)

    payload = server._live_session_payload("sid", session)

    assert payload["running"] is False
    assert payload["inflight"]["status"] == "error"
    assert payload["inflight"]["error"] == "provider unavailable"


def test_twenty_sequential_prompts_reach_provider_and_persist_in_one_session(
    monkeypatch,
    tmp_path,
    emits,
    turn_env,
):
    """The second-prompt canary is a 20-turn owner/persistence trace."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / ".hermes" / "state.db")

    class _ProviderBackedAgent:
        model = "deterministic-canary"
        provider = "local-fake-provider"
        session_id = "twenty-turn-session"
        context_compressor = None

        def __init__(self):
            self.calls = []

        def clear_interrupt(self):
            return None

        def run_conversation(
            self,
            prompt,
            *,
            conversation_history=None,
            stream_callback=None,
            **_kwargs,
        ):
            # This is the deterministic provider invocation boundary. It has no
            # retries: one gateway turn must produce exactly one provider call.
            self.calls.append(prompt)
            reply = f"ack-{len(self.calls):02d}"
            if stream_callback is not None:
                stream_callback(reply)
            messages = [
                *(conversation_history or []),
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]
            db.replace_messages(self.session_id, messages)
            return {"final_response": reply, "messages": messages}

    agent = _ProviderBackedAgent()
    session = _session(agent=agent, session_key=agent.session_id)
    server._sessions["twenty-turn-sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        for turn in range(1, 21):
            response = server.handle_request(
                {
                    "id": f"turn-{turn}",
                    "method": "prompt.submit",
                    "params": {
                        "session_id": "twenty-turn-sid",
                        "text": f"prompt-{turn:02d}",
                    },
                }
            )
            assert response["result"]["status"] == "streaming"
            assert session["running"] is False

        persisted = db.get_messages_as_conversation(agent.session_id)
    finally:
        server._sessions.pop("twenty-turn-sid", None)
        db.close()

    assert agent.calls == [f"prompt-{turn:02d}" for turn in range(1, 21)]
    assert len(session["history"]) == 40
    assert len(persisted) == 40
    assert persisted[0]["content"] == "prompt-01"
    assert persisted[1]["content"] == "ack-01"
    assert persisted[-2]["content"] == "prompt-20"
    assert persisted[-1]["content"] == "ack-20"
    completes = _events(emits, "message.complete")
    assert len(completes) == 20
    assert [payload["text"] for payload in completes] == [
        f"ack-{turn:02d}" for turn in range(1, 21)
    ]
