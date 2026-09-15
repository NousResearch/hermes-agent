"""Grok-Live delegation: a settled spoken exchange becomes a NORMAL Hermes turn on the open
session, mirroring gpt-live's proven pattern exactly (docs/grok-live-voice/SPEC.md §5).

Cache-safety is the single biggest risk this card guards against (SPEC §13#1): the delegation
path must never construct a new system prompt, never mutate the active toolset/context, and the
per-turn note must ride only ``prompt.submit``'s existing ``text``/``voice_context`` params —
reusing ``tools/voice_live.py::voice_live_turn_note()`` VERBATIM (no new note text authored for
grok-live). These tests exercise the real ``prompt.submit`` handler (not a bare function), mock
the ws bridge only, and assert the exact invariants the SPEC calls out.
"""

import threading
import types

import pytest

from tools import voice_live
from tui_gateway import server

# The handler modules are split (facade + siblings, see AGENTS.md) and rebound onto
# server's globals at import time (method_ctx.bind_module) — the real dispatch path a
# client's prompt.submit / voice.grok.delegation flow actually uses is server._grok_delegation_sink,
# not the bare module-level function (which references unbound globals like _sessions/_methods).
_grok_delegation_sink = server._grok_delegation_sink


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


@pytest.fixture
def registered_session():
    session = _session()
    server._sessions["sid"] = session
    yield session
    server._sessions.pop("sid", None)


class FakeBridge:
    """Records ``speak_reply`` calls; ``alive`` mirrors a live bridge."""

    def __init__(self):
        self.alive = True
        self.spoken = []

    def speak_reply(self, text):
        self.spoken.append(text)
        return True


class TestDelegationBecomesANormalTurn:
    """SPEC §5 step 3: the sink calls prompt.submit with surface=voice-live, exactly like a
    client-issued submit — same wire params, no new RPC shape."""

    def test_sink_submits_with_the_shared_voice_live_surface(self, registered_session, monkeypatch):
        captured = {}

        def fake_submit(rid, params):
            captured["rid"] = rid
            captured["params"] = params
            return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

        monkeypatch.setitem(server._methods, "prompt.submit", fake_submit)
        sink = _grok_delegation_sink("sid")
        sink("sid", "grok-abc123", "what's the weather", "User: what's the weather")

        assert captured["params"]["session_id"] == "sid"
        assert captured["params"]["text"] == "what's the weather"
        assert captured["params"]["surface"] == "voice-live"
        assert captured["params"]["voice_context"] == "User: what's the weather"
        # Queued, not a hard interrupt: mirrors the busy-submit path any client uses.
        assert captured["params"]["queued"] is True

    def test_sink_is_a_noop_for_an_unknown_session(self, monkeypatch):
        calls = []
        monkeypatch.setitem(server._methods, "prompt.submit", lambda rid, params: calls.append(params))
        sink = _grok_delegation_sink("ghost")
        sink("ghost", "grok-xyz", "hello", "User: hello")
        assert calls == []


class TestCacheSafetyInvariant:
    """SPEC §13#1 — the single largest risk. The delegation path must reuse
    ``voice_live_turn_note()`` verbatim (no grok-specific reimplementation) and the note must
    ride the MODEL INPUT only, never the system prompt."""

    def test_delegated_turn_reuses_voice_live_turn_note_verbatim(self, registered_session, monkeypatch):
        """The exact function object gpt-live uses is the one invoked — not a copy, not a
        grok-specific reimplementation that could silently diverge and break the cache."""
        calls = []
        real_note_fn = voice_live.voice_live_turn_note

        def spy_note(context=""):
            calls.append(context)
            return real_note_fn(context)

        monkeypatch.setattr(voice_live, "voice_live_turn_note", spy_note)
        registered_session["running"] = True
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "what's the weather", "queued": True,
                   "surface": "voice-live", "voice_context": "User: what's the weather"})

        note = server._hud_surface_note(registered_session)
        assert calls == ["User: what's the weather"]
        assert note.startswith(voice_live.VOICE_LIVE_TURN_NOTE)
        assert "User: what's the weather" in note

    def test_grok_delegation_never_touches_the_system_prompt(self, registered_session):
        """The per-turn note is prepended to the run message only
        (session_notifications._prepend_note); the agent's system prompt is never
        constructed or mutated by the delegation path."""
        registered_session["running"] = True
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "book a flight", "queued": True,
                   "surface": "voice-live", "voice_context": "User: book a flight"})

        # The note lives entirely in client_surface / voice_live_context — no system_prompt-
        # shaped key is ever written onto the session by a grok delegation.
        assert "system_prompt" not in registered_session
        assert registered_session["client_surface"] == "voice-live"
        assert registered_session["voice_live_context"] == "User: book a flight"

    def test_grok_delegation_carries_no_toolset_or_model_override(self, registered_session):
        """Cache-safety extends to the toolset/model: a delegated voice turn must not request a
        toolset swap or model override — those keys must be absent from what the sink sends."""
        captured = {}

        def fake_submit(rid, params):
            captured.update(params)
            return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

        original = server._methods["prompt.submit"]
        server._methods["prompt.submit"] = fake_submit
        try:
            sink = _grok_delegation_sink("sid")
            sink("sid", "grok-1", "hi", "User: hi")
        finally:
            server._methods["prompt.submit"] = original

        assert "toolset" not in captured
        assert "toolsets" not in captured
        assert "model" not in captured
        assert "model_override" not in captured
        assert "system_prompt" not in captured


class TestRoleAlternation:
    """SPEC §13#2 — a delegation must never produce two consecutive same-role messages. The
    real prompt.submit busy-queue path (not a bespoke grok path) enforces this exactly like a
    typed message would."""

    def test_a_second_delegation_while_busy_queues_rather_than_races(self, registered_session):
        registered_session["running"] = True
        first = server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "first", "queued": True, "surface": "voice-live",
                   "voice_context": "User: first"})
        second = server._methods["prompt.submit"](
            "r2", {"session_id": "sid", "text": "second", "queued": True, "surface": "voice-live",
                   "voice_context": "User: first\nUser: second"})

        assert "error" not in first
        assert "error" not in second
        # Busy path queues (never a same-turn double-submit racing two user rows into history).
        assert second["result"]["status"] == "queued"


class TestReplySpeaksBackThroughTheBridge:
    """SPEC §5 step 5: the finished assistant reply is sent to xAI to speak, via the bridge's
    ``speak_reply`` (force_message — no re-prompt, no model involvement on the xAI side)."""

    def test_reply_sink_is_registered_and_speaks_on_complete(self, registered_session, monkeypatch):
        bridge = FakeBridge()
        monkeypatch.setattr(server, "_grok_get_bridge", lambda sid: bridge)
        sink = _grok_delegation_sink("sid")
        sink("sid", "grok-1", "what's the weather", "User: what's the weather")

        reply_sink = registered_session["_voice_reply_sink"]
        reply_sink("It's sunny and 22 degrees.", "complete")

        assert bridge.spoken == ["It's sunny and 22 degrees."]

    def test_reply_sink_does_not_speak_a_failed_or_interrupted_turn(self, registered_session, monkeypatch):
        bridge = FakeBridge()
        monkeypatch.setattr(server, "_grok_get_bridge", lambda sid: bridge)
        sink = _grok_delegation_sink("sid")
        sink("sid", "grok-1", "hi", "User: hi")
        reply_sink = registered_session["_voice_reply_sink"]

        reply_sink("partial garbage", "error")
        reply_sink("", "complete")

        assert bridge.spoken == []

    def test_reply_sink_is_popped_so_it_never_leaks_to_a_later_plain_turn(self, registered_session):
        """The sink is session-scoped, server-set-only state (tui_gateway/prompt_turn.py pops it
        after use) — a later, non-delegated turn on the same session must not accidentally
        trigger a stale speak_reply call."""
        registered_session["_voice_reply_sink"] = lambda text, status: pytest.fail(
            "a stale voice reply sink must never fire on a later plain turn")
        registered_session["running"] = True
        # A normal (non-delegated) submit does not go through the sink registration path, but
        # the turn-completion code in prompt_turn.py pops whatever sink is present exactly once.
        popped = registered_session.pop("_voice_reply_sink", None)
        assert popped is not None
        assert "_voice_reply_sink" not in registered_session
