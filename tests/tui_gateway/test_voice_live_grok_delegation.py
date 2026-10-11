"""Grok-Live delegation (renderer-submits design, revised 2026-09-15 — kanban t_07a77402):

- The RENDERER is the single ``prompt.submit`` caller (gpt-live parity): the backend NEVER
  submits on a delegation — the old ``_grok_delegation_sink`` double-submitted every
  delegation on an existing chat and no-oped on a fresh draft's synthetic id.
- ``voice.grok.*`` events reach the owning app even when the bridge id is NOT a registered
  Hermes session (the fresh-draft case): they route to the caller transport captured at
  ``voice.grok.start`` instead of falling to stdio.
- The spoken reply is pulled by the renderer via ``voice.grok.speak`` when its turn settles.

These tests exercise the real handlers through ``server``'s rebound globals (the dispatch
path a client's RPC actually uses), mock the ws bridge only, and assert the invariants above.
"""

import threading
import types

import pytest

import tools.voice_live_grok as grok_config
import tools.voice_live_grok_bridge as bridge_module
from tui_gateway import server

# The handler modules are split (facade + siblings, see AGENTS.md) and rebound onto
# server's globals at import time (method_ctx.bind_module) — the real dispatch path a
# client's RPC actually uses is server's rebound copy, not the bare module-level function.


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(valid_tool_names=set()),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


class FakeTransport:
    """Records written frames; stands in for the websocket the app speaks on."""

    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        return None


@pytest.fixture
def registered_session():
    session = _session()
    server._sessions["sid"] = session
    yield session
    server._sessions.pop("sid", None)


@pytest.fixture
def grok_rpc(monkeypatch):
    """Fake bridge class + available status; clean per-test bridge/transport registries."""
    class FakeBridge:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.spoken = []
            self.mic = []
            FakeBridge.instances.append(self)

        def start(self):
            self.started = True

        def stop(self, timeout=10.0):
            self.stopped = True
            return True

        @property
        def alive(self):
            return self.started and not self.stopped

        def send_mic(self, pcm):
            self.mic.append(pcm)
            return {"accepted": True}

        def set_muted(self, muted):
            return bool(muted)

        def speak_reply(self, text):
            self.spoken.append(text)
            return True

    FakeBridge.instances = []
    monkeypatch.setattr(bridge_module, "GrokLiveBridge", FakeBridge)
    monkeypatch.setattr(grok_config, "resolve_grok_live_status", lambda: {
        "mode": "grok-live", "available": True, "reason": None, "model": "m", "voice": "v"})
    import sys
    fake_wake = types.ModuleType("tools.wake_word")
    fake_wake.pause_listening = lambda owner=None: False  # noqa: F841 — attribute on a stub module
    monkeypatch.setitem(sys.modules, "tools.wake_word", fake_wake)
    with server._grok_bridges_lock:
        server._grok_bridges.clear()
        server._grok_wake_owners.clear()
        server._grok_client_transports.clear()
    with server._grok_alias_lock:
        server._grok_alias.clear()
    yield FakeBridge
    with server._grok_bridges_lock:
        server._grok_bridges.clear()
        server._grok_wake_owners.clear()
        server._grok_client_transports.clear()
    with server._grok_alias_lock:
        server._grok_alias.clear()


def _start(bridge_cls, sid, transport):
    """voice.grok.start with a pinned caller transport (the reconnecting-app shape)."""
    import tui_gateway.transport as transport_mod
    token = transport_mod.bind_transport(transport)
    try:
        return server._methods["voice.grok.start"](1, {"session_id": sid})
    finally:
        transport_mod.reset_transport(token)


class TestBackendNeverSubmits:
    """Single-submitter invariant: a settled delegation emits the event (prompt + context)
    and NOTHING calls prompt.submit on the backend — exactly one submitter, the renderer."""

    def test_no_delegation_sink_exists_on_the_server(self):
        assert not hasattr(server, "_grok_delegation_sink")

    def test_a_delegation_flush_never_calls_prompt_submit(self, grok_rpc):
        submitted = []
        original = server._methods["prompt.submit"]
        server._methods["prompt.submit"] = lambda rid, params: submitted.append(params) or {
            "jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}
        try:
            transport = FakeTransport()
            _start(grok_rpc, "draft-1", transport)
            (bridge,) = grok_rpc.instances
            # Drive the bridge's delegation flush the way the live bridge does on settle.
            bridge.kwargs["emit"]("voice.grok.delegation", {
                "session_id": "draft-1", "delegation_id": "grok-abc",
                "prompt": "what's the weather", "context": "User: what's the weather"})
        finally:
            server._methods["prompt.submit"] = original

        assert submitted == []  # the backend is not a submitter, period
        events = [f for f in transport.frames
                  if f.get("params", {}).get("type") == "voice.grok.delegation"]
        assert events, "the delegation event must reach the app"
        payload = events[0]["params"]["payload"]
        assert payload["prompt"] == "what's the weather"
        assert payload["context"] == "User: what's the weather"

    def test_no_voice_reply_sink_is_registered_on_the_session(self, grok_rpc, registered_session):
        """prompt_turn.py's reply-sink seam is gone: a delegated turn leaves no server-set
        sink on the session dict (the renderer pulls the reply via voice.grok.speak)."""
        _start(grok_rpc, "sid", FakeTransport())
        (bridge,) = grok_rpc.instances
        bridge.kwargs["emit"]("voice.grok.delegation", {
            "session_id": "sid", "delegation_id": "grok-1", "prompt": "hi", "context": "User: hi"})
        assert "_voice_reply_sink" not in registered_session


class TestFreshDraftRouting:
    """Invariant (i): a fresh-draft voice start reaches the app with NO pre-existing Hermes
    session — events route to the caller transport captured at start, never stdio."""

    def test_events_reach_the_app_without_a_registered_session(self, grok_rpc):
        transport = FakeTransport()
        answer = _start(grok_rpc, "grok-live-1789482317652-2qi1hr", transport)
        assert answer["result"]["started"] is True
        assert "grok-live-1789482317652-2qi1hr" not in server._sessions  # draft: no session

        (bridge,) = grok_rpc.instances
        emit = bridge.kwargs["emit"]
        emit("voice.grok.state", {"session_id": "grok-live-1789482317652-2qi1hr", "state": "listening"})
        emit("voice.grok.transcript", {"session_id": "grok-live-1789482317652-2qi1hr",
                                       "speaker": "user", "text": "hello"})

        assert len(transport.frames) == 2  # heard by the app, not dropped to stdio

    def test_existing_chat_events_prefer_the_session_transport(self, grok_rpc, registered_session):
        session_transport = FakeTransport()
        registered_session["transport"] = session_transport
        caller_transport = FakeTransport()
        _start(grok_rpc, "sid", caller_transport)

        (bridge,) = grok_rpc.instances
        bridge.kwargs["emit"]("voice.grok.state", {"session_id": "sid", "state": "listening"})

        assert len(session_transport.frames) == 1  # byte-identical existing-chat routing
        assert caller_transport.frames == []

    def test_rekey_moves_the_bridge_onto_the_real_session_id(self, grok_rpc):
        transport = FakeTransport()
        draft_id = "grok-live-1"
        _start(grok_rpc, draft_id, transport)
        (bridge,) = grok_rpc.instances

        answer = server._methods["voice.grok.rekey"](
            1, {"from_session_id": draft_id, "to_session_id": "1d96782d"})
        assert answer["result"]["rekeyed"] is True
        assert server._grok_bridges["1d96782d"] is bridge
        assert draft_id not in server._grok_bridges

        # The renderer keeps sending by its synthetic id; the alias resolves to the bridge.
        pcm = __import__("base64").b64encode(b"\x00" * 100).decode()
        assert server._methods["voice.grok.audio"](
            1, {"session_id": draft_id, "pcm_b64": pcm})["result"]["accepted"] is True
        assert bridge.mic

        # Stopping by EITHER id tears the bridge down.
        assert server._methods["voice.grok.stop"](1, {"session_id": draft_id})["result"]["stopped"] is True
        assert bridge.stopped


class TestSpeakIsPulled:
    """Invariant: the finished reply is spoken via voice.grok.speak (bridge.speak_reply)."""

    def test_speak_relays_the_reply_to_the_bridge(self, grok_rpc):
        _start(grok_rpc, "s1", FakeTransport())
        (bridge,) = grok_rpc.instances
        answer = server._methods["voice.grok.speak"](
            1, {"session_id": "s1", "text": "It's sunny and 22 degrees."})
        assert answer["result"]["spoken"] is True
        assert bridge.spoken == ["It's sunny and 22 degrees."]

    def test_speak_without_a_live_bridge_is_a_clean_not_running(self, grok_rpc):
        answer = server._methods["voice.grok.speak"](1, {"session_id": "ghost", "text": "hi"})
        assert answer["result"] == {"spoken": False, "reason": "not_running"}

    def test_speak_ignores_empty_text(self, grok_rpc):
        _start(grok_rpc, "s1", FakeTransport())
        (bridge,) = grok_rpc.instances
        answer = server._methods["voice.grok.speak"](1, {"session_id": "s1", "text": "  "})
        assert answer["result"]["spoken"] is False
        assert bridge.spoken == []
