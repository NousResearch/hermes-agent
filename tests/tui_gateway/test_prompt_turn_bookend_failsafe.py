"""Tests for fail-safe turn finalization and bookend preservation (#131740).

When an agent turn finishes, the session MUST be released (session["running"] = False),
the "tui turn finished" bookend logged, and the turn crash marker retired, even if
auxiliary post-turn operations (memory trim, audio/TTS, model restore, scopes) fail.
"""

from __future__ import annotations

import logging
import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

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


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _agent_returning(result):
    return types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: result,
        clear_interrupt=lambda: None,
        _execution_thread_id=None,
        _mute_notification_reply=False,
    )


def test_session_released_and_marker_retired_when_housekeeping_fails(turn_env, caplog, monkeypatch):
    """If post-turn housekeeping (e.g. trim_memory) fails, settlement must still complete:
    session released, bookend logged, and crash marker retired (#131740)."""
    import hermes_cli.mem_trim as mem_trim

    retired_markers = []
    original_retire = server._retire_turn_marker

    def spy_retire(session, key):
        retired_markers.append(key)
        original_retire(session, key)

    monkeypatch.setattr(server, "_retire_turn_marker", spy_retire)
    monkeypatch.setattr(mem_trim, "trim_memory", lambda **k: (_ for _ in ()).throw(RuntimeError("trim crash")))

    agent = _agent_returning({"final_response": "done", "completed": True, "messages": []})
    session = _session(agent=agent)

    with caplog.at_level(logging.INFO):
        server._run_prompt_submit("rid", "ui-sid", session, "hello")

    # Session must be released so next prompt is not blocked by "session busy"
    assert session["running"] is False, "session was left in running state after turn"

    # Bookend must be logged
    bookends = [r for r in caplog.records if "tui turn finished" in r.getMessage()]
    assert len(bookends) == 1, f"expected exactly 1 bookend, got {len(bookends)}"

    # Crash marker must be retired
    assert len(retired_markers) >= 1, f"expected crash marker to be retired, got {retired_markers}"


def test_session_released_and_marker_retired_when_pre_settle_teardown_fails(turn_env, caplog, monkeypatch):
    """If pre-settle teardown encounters an unhandled exception, session['running'] must still be False,
    the 'tui turn finished' bookend must still be logged, and the crash marker retired (#131740)."""
    retired_markers = []
    original_retire = server._retire_turn_marker

    def spy_retire(session, key):
        retired_markers.append(key)
        original_retire(session, key)

    monkeypatch.setattr(server, "_retire_turn_marker", spy_retire)

    agent = _agent_returning({"final_response": "done", "completed": True, "messages": []})
    session = _session(agent=agent)

    # Simulate failure in pre-settle teardown (e.g. scopes reset or snapshot cleanup raising)
    def broken_pre_settle(*args, **kwargs):
        raise RuntimeError("simulated pre-settle cleanup crash")

    monkeypatch.setattr(server, "_pre_settle_teardown", broken_pre_settle)

    with caplog.at_level(logging.INFO):
        server._run_prompt_submit("rid", "ui-sid", session, "hello")

    # Session must be released so next prompt is not blocked by "session busy"
    assert session["running"] is False, "session was left in running state after turn"

    # Bookend must be logged
    bookends = [r for r in caplog.records if "tui turn finished" in r.getMessage()]
    assert len(bookends) == 1, f"expected exactly 1 bookend, got {len(bookends)}"

    # Pre-settle failure must be logged
    errors = [r for r in caplog.records if "pre-settle teardown failed" in r.getMessage()]
    assert len(errors) == 1

    # Crash marker must be retired
    assert len(retired_markers) >= 1, f"expected crash marker to be retired, got {retired_markers}"


def test_settle_turn_resilient_to_inner_step_failure(turn_env, caplog, monkeypatch):
    """Even if an inner settlement step (e.g. emitting settled session info) fails,
    session release, bookend, and marker retirement must not be prevented (#131740)."""
    retired_markers = []
    original_retire = server._retire_turn_marker

    def spy_retire(session, key):
        retired_markers.append(key)
        original_retire(session, key)

    monkeypatch.setattr(server, "_retire_turn_marker", spy_retire)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("emit failure")))

    agent = _agent_returning({"final_response": "done", "completed": True, "messages": []})
    session = _session(agent=agent)

    with caplog.at_level(logging.INFO):
        server._run_prompt_submit("rid", "ui-sid", session, "hello")

    assert session["running"] is False
    bookends = [r for r in caplog.records if "tui turn finished" in r.getMessage()]
    assert len(bookends) == 1
    assert len(retired_markers) >= 1


def test_usage_ticker_join_bounded(turn_env, monkeypatch):
    """_invoke_agent must not join the usage ticker thread indefinitely (#131740)."""
    join_timeout_called = []

    class DummyUsageThread:
        def join(self, timeout=None):
            join_timeout_called.append(timeout)

    monkeypatch.setattr(server, "_start_usage_ticker", lambda sid, agent: (threading.Event(), DummyUsageThread()))

    agent = _agent_returning({"final_response": "done", "completed": True, "messages": []})
    session = _session(agent=agent)

    server._run_prompt_submit("rid", "ui-sid", session, "hello")

    assert len(join_timeout_called) == 1
    # Must have a non-None, bounded timeout (on origin/main, join() was called with timeout=None)
    assert join_timeout_called[0] is not None
    assert 0 < join_timeout_called[0] <= 5.0


def test_settlement_not_gated_by_blocked_trim_memory(monkeypatch, tmp_path, caplog):
    """While trim_memory is blocked in post-turn housekeeping, settlement must already be done (#131740)."""
    import hermes_cli.mem_trim as mem_trim

    entered = threading.Event()
    release = threading.Event()

    def blocking_trim(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=10.0), "timed out waiting to release blocked trim"
        return False

    monkeypatch.setattr(mem_trim, "trim_memory", blocking_trim)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})

    retired_markers = []
    original_retire = server._retire_turn_marker

    def spy_retire(session, key):
        retired_markers.append(key)
        original_retire(session, key)

    monkeypatch.setattr(server, "_retire_turn_marker", spy_retire)

    agent = _agent_returning({"final_response": "done", "completed": True, "messages": []})
    session = _session(agent=agent)

    # Clean sessions registry hygiene (#58576 quiescence isolation)
    server._sessions.clear()
    server._sessions["ui-sid-async"] = session

    try:
        # Note: turn_env is NOT used here so threading.Thread runs genuinely in the background
        with caplog.at_level(logging.INFO):
            started = server._run_prompt_submit("rid-async", "ui-sid-async", session, "hello async")
            assert started is True

            # Wait until trim_memory is entered
            assert entered.wait(timeout=5.0), "trim_memory was never entered"

            # While trim_memory is still blocked, settlement must already be complete!
            assert session["running"] is False, "session was still marked running while trim was blocked"
            bookends = [r for r in caplog.records if "tui turn finished" in r.getMessage()]
            assert len(bookends) == 1, f"expected 1 bookend while trim blocked, got {len(bookends)}"
            assert len(retired_markers) >= 1, f"expected marker retired while trim blocked, got {retired_markers}"

            # Unblock trim and clean up
            release.set()
            thread = session.get("_run_thread")
            if thread is not None:
                thread.join(timeout=5.0)
    finally:
        release.set()
        server._sessions.clear()


def test_post_turn_trim_skipped_if_subsequent_turn_started_on_same_session(turn_env, monkeypatch):
    """If a subsequent turn has already started on the same session (running=True), post-turn housekeeping
    must skip trim_memory so it does not stall the new in-flight turn (#58576/#131740)."""
    calls = []
    import hermes_cli.mem_trim as mem_trim
    monkeypatch.setattr(mem_trim, "trim_memory", lambda **k: calls.append(k))

    server._sessions.clear()
    session = _session()
    server._sessions["test-sid"] = session

    try:
        st = server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=True)
        st.settled = True

        # Case 1: Subsequent turn started on this session -> session["running"] is True
        session["running"] = True
        server._post_turn_housekeeping("test-sid", session, st)
        assert len(calls) == 0, "trim_memory should have been skipped when subsequent turn is running"

        # Case 2: Session is idle (no subsequent turn) -> session["running"] is False
        session["running"] = False
        server._post_turn_housekeeping("test-sid", session, st)
        assert len(calls) == 1, "trim_memory should have run when session is quiescent and idle"
    finally:
        server._sessions.clear()


def test_settle_turn_succeeds_even_if_agent_is_none(turn_env, caplog, monkeypatch):
    """_settle_turn must not fail if st.agent is None or raises when clearing interim callback (#131740)."""
    agent = _agent_returning({"final_response": "done", "completed": True, "messages": []})
    session = _session(agent=agent)

    # Force st.agent to None during settle to test resilience against None agent
    original_settle = server._settle_turn

    def settle_with_none_agent(sid, sess, st, monotonic):
        st.agent = None
        original_settle(sid, sess, st, monotonic)

    monkeypatch.setattr(server, "_settle_turn", settle_with_none_agent)

    with caplog.at_level(logging.INFO):
        server._run_prompt_submit("rid", "ui-sid", session, "hello")

    assert session["running"] is False
    bookends = [r for r in caplog.records if "tui turn finished" in r.getMessage()]
    assert len(bookends) == 1

