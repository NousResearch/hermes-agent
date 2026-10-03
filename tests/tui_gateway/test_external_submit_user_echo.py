"""Externally submitted prompts reach the session's other watchers (#55564).

Transport stealing itself was fixed by additive FanoutTransport attachment (#105571): an external
``prompt.submit`` no longer displaces the Desktop window's transport, and the turn's
``message.start``/``message.delta``/``message.complete`` stream reaches every peer. The gap this
suite pins is the missing user-input announcement: the submitted text itself reached only the
submitter, so a Desktop window watching the session rendered the assistant's answer over an
invisible prompt — indicator-only chrome, no user bubble.

The fix: ``prompt.submit`` echoes ``message.user_echo`` to every OTHER live peer (the submitter
already painted its own optimistic bubble), stamped into the replay ring so a disconnected window
that reconnects mid-turn seeds the same row. Prompt content rides the event (the client must
render it); submitter identity and credentials never do.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server

from tui_gateway.transport import FanoutTransport


class RecordingTransport:
    """In-process Transport: live-peer-shaped, records every frame."""

    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        self.closed = True


def _echo_frames(transport) -> list[dict]:
    return [f for f in transport.frames if f.get("params", {}).get("type") == "message.user_echo"]


def _session(transport) -> dict:
    return {
        "agent": None,
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": transport,
    }


def test_external_submit_echoes_to_other_peers_but_not_the_submitter(monkeypatch):
    """The echo reaches every other live peer; the submitter's own transport is skipped."""
    submitter, watcher = RecordingTransport(), RecordingTransport()
    session = _session(FanoutTransport(submitter, watcher))
    monkeypatch.setattr(server, "current_transport", lambda: submitter)

    server._echo_external_submit_to_session_peers(
        "ui-sid", session, "hello from the CLI", None, 42)

    assert len(_echo_frames(watcher)) == 1
    frame = _echo_frames(watcher)[0]
    assert frame["params"]["session_id"] == "ui-sid"
    assert frame["params"]["payload"]["text"] == "hello from the CLI"
    assert frame["params"]["payload"]["row_id"] == 42
    # The submitter already rendered its optimistic bubble — no double.
    assert _echo_frames(submitter) == []
    # Stamped for the replay ring: a reconnecting window replays the echo.
    assert isinstance(frame["params"].get("seq"), int)


def test_no_other_peer_no_echo(monkeypatch):
    """A single-client submit (the common case) emits nothing — no fan-out, no ring entry."""
    submitter = RecordingTransport()
    session = _session(submitter)
    monkeypatch.setattr(server, "current_transport", lambda: submitter)

    server._echo_external_submit_to_session_peers("ui-sid", session, "solo", None, None)

    assert _echo_frames(submitter) == []


def test_hidden_and_empty_submits_do_not_echo(monkeypatch):
    """Off-screen (``display_kind: hidden``) and empty sends never mint bubbles."""
    watcher = RecordingTransport()
    session = _session(FanoutTransport(RecordingTransport(), watcher))
    monkeypatch.setattr(server, "current_transport", lambda: None)

    server._echo_external_submit_to_session_peers("ui-sid", session, "scaffolding", "hidden", 7)
    server._echo_external_submit_to_session_peers("ui-sid", session, "   ", None, None)

    assert _echo_frames(watcher) == []


@pytest.fixture()
def submit_env(monkeypatch, tmp_path):
    """Neutralize prompt.submit's environment-heavy side paths, keep the echo real."""
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *a, **k: True)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    # The turn thread never runs here: its streaming is other suites' subject.
    monkeypatch.setattr(
        server.threading, "Thread",
        types.SimpleNamespace)


def test_prompt_submit_emits_the_echo_after_the_row_persists(submit_env, monkeypatch, tmp_path):
    """End to end: an external prompt.submit announces the user text to the watching peer."""
    sid = "ui-sid"
    submitter, watcher = RecordingTransport(), RecordingTransport()
    session = _session(FanoutTransport(submitter, watcher))
    session.update({
        "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
        "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
        "cwd": str(tmp_path), "profile_home": None, "agent": types.SimpleNamespace(),
    })
    server._sessions[sid] = session
    monkeypatch.setattr(server, "current_transport", lambda: submitter)
    # Persisted submit row: a plain dict with a durable row id.
    monkeypatch.setattr(
        server, "_persist_session_row_for_submit",
        lambda rid, s, text=None, display_kind=None: s.__setitem__("_submit_user_row", {"_row_id": 99}) or None)

    class _NoStart:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return None

        def is_alive(self):
            return False

    monkeypatch.setattr(server.threading, "Thread", _NoStart)

    response = server._methods["prompt.submit"](1, {"session_id": sid, "text": "hi from elsewhere"})

    assert "error" not in response, response
    echoes = _echo_frames(watcher)
    assert len(echoes) == 1
    assert echoes[0]["params"]["payload"]["text"] == "hi from elsewhere"
    assert echoes[0]["params"]["payload"]["row_id"] == 99
    assert _echo_frames(submitter) == []
    server._sessions.pop(sid, None)
