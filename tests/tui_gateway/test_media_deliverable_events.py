"""Serve-path media-deliverable events (D1) — real turn integration.

The desktop backend runs the agent in-process (``_run_prompt_submit`` →
``agent.run_conversation``). When the final reply carries explicit ``MEDIA:``
tags, the server emits a ``media.deliverable`` event frame through the
session's own transport — full payload {path, kind, mime, size, session_id,
origin} — riding the same event transport (seq + replay) as every other
session event. Uses temp HERMES_HOME (conftest sandbox) and a real media file.
"""

import threading
import types

import pytest

from tui_gateway import server


class _CollectingTransport:
    def __init__(self):
        self.frames = []
        self._lock = threading.Lock()

    def write(self, obj):
        with self._lock:
            self.frames.append(obj)
        return True

    def close(self):
        pass


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
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
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _media_file(tmp_path, name="hello.png"):
    f = tmp_path / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"z" * 64)
    return f


def _session(agent, sid="serve-media-sid", tmp_path=None):
    session = {
        "agent": agent,
        "session_key": sid,
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
    }
    if tmp_path is not None:
        session["cwd"] = str(tmp_path)
        session["profile_home"] = str(tmp_path)
    server._sessions[sid] = session
    return session


def _run_turn(sid, prompt):
    server._run_prompt_submit("rid", sid, server._sessions[sid], prompt)


def _media_events(captured):
    return [(sid, payload) for event, sid, payload in captured
            if event == "media.deliverable"]


def test_media_tag_in_reply_emits_deliverable_event(
    emits, turn_env, tmp_path
):
    f = _media_file(tmp_path)
    agent = types.SimpleNamespace(
        session_id="serve-media-sid",
        run_conversation=lambda *a, **k: {
            "final_response": f"done\nMEDIA:{f}",
            "messages": [],
        },
        clear_interrupt=lambda: None,
    )
    _session(agent, tmp_path=tmp_path)

    _run_turn("serve-media-sid", "make me a picture")

    events = _media_events(emits)
    assert len(events) == 1
    sid, payload = events[0]
    assert sid == "serve-media-sid"
    assert payload["path"] == str(f)
    assert payload["kind"] == "image"
    assert payload["mime"] == "image/png"
    assert payload["size"] == f.stat().st_size
    assert payload["session_id"] == "serve-media-sid"
    assert payload["origin"] == "serve"


def test_no_media_in_reply_emits_no_event(emits, turn_env, tmp_path):
    agent = types.SimpleNamespace(
        session_id="serve-media-sid",
        run_conversation=lambda *a, **k: {
            "final_response": "just text, no files",
            "messages": [],
        },
        clear_interrupt=lambda: None,
    )
    _session(agent, tmp_path=tmp_path)

    _run_turn("serve-media-sid", "hello")

    assert _media_events(emits) == []


def test_media_event_carries_session_id_payload_and_rides_replay(
    turn_env, tmp_path
):
    """The frame reaches the session's transport AND the replay ring, so a
    reconnecting client recovers it via session.events.since. Uses the real
    emit boundary (no _emit monkeypatch) to exercise write_json routing."""
    from tui_gateway.event_replay import events_since, reset_replay_state

    reset_replay_state()
    f = _media_file(tmp_path, "replay.png")
    agent = types.SimpleNamespace(
        session_id="serve-media-sid",
        run_conversation=lambda *a, **k: {
            "final_response": f"MEDIA:{f}",
            "messages": [],
        },
        clear_interrupt=lambda: None,
    )
    sid = "serve-media-sid"
    session = _session(agent, sid=sid, tmp_path=tmp_path)
    transport = _CollectingTransport()
    session["transport"] = transport

    _run_turn(sid, "make a picture")

    # Landed on the live transport, seq-stamped (other turn events share
    # the same per-session counter, so the exact value is turn-dependent).
    frames = [fr for fr in transport.frames
              if fr.get("method") == "event"
              and fr["params"].get("type") == "media.deliverable"]
    assert len(frames) == 1
    params = frames[0]["params"]
    assert params["session_id"] == sid
    assert isinstance(params["seq"], int) and params["seq"] >= 1
    assert params["payload"]["path"] == str(f)
    assert params["payload"]["origin"] == "serve"
    # And is replayable.
    replayed = [p for p in events_since(sid, 0)
                if p.get("type") == "media.deliverable"]
    assert [p["payload"]["path"] for p in replayed] == [str(f)]
    reset_replay_state()


def test_turn_level_auto_appended_tool_media_emits_on_serve_path(
    emits, turn_env, tmp_path
):
    """TTS/image_generate-style tool payloads (auto-append shape) surface on
    the serve path too: the final response the turn completes with carries
    the MEDIA tag, and the event mirrors exactly that file."""
    f = _media_file(tmp_path, "tts.wav")
    final_with_tag = f"speaking\nMEDIA:{f}"
    agent = types.SimpleNamespace(
        session_id="serve-media-sid",
        run_conversation=lambda *a, **k: {
            # Simulate what the auto-append machinery produces upstream:
            # tool tags merged into the final response.
            "final_response": final_with_tag,
            "messages": [],
        },
        clear_interrupt=lambda: None,
    )
    _session(agent, tmp_path=tmp_path)

    _run_turn("serve-media-sid", "say something")

    events = _media_events(emits)
    assert len(events) == 1
    assert events[0][1]["path"] == str(f)
    assert events[0][1]["mime"] == "audio/wav"
