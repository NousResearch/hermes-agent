"""GPT-Live voice chat mode: the full-duplex voice frontend that delegates to Hermes.

The live voice model owns the microphone and speaker and has no tools; every real
request is delegated to Hermes as a normal turn on the open session. Two contracts
matter and are pinned here:

* the gateway never hands the OpenAI key to the renderer — ``POST /v1/live/sessions``
  is performed server-side from the renderer's SDP offer, with the session pinned to
  client delegation so Hermes (any model) is the backend;
* a turn submitted from the live voice surface carries the spoken-delegation note on
  the MODEL INPUT only (the byte-stable system prompt is untouched), exactly like the
  HUD note it sits beside.
"""

import json
import threading
import types

import pytest

from tools import voice_live
from tui_gateway import server


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(valid_tool_names=set()),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "transport": None,
        "attached_images": [],
        **extra,
    }


class TestSessionCreation:
    def test_client_delegation_and_key_stay_server_side(self, monkeypatch):
        """Whatever the renderer sends, the vendor request pins ``delegation.type == client``
        (Hermes is the backend) and authenticates with the resolved key; the client only ever
        sees the vendor answer."""
        captured = {}

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["body"] = json.loads(req.data)
            return _Resp(json.dumps({"session": {"id": "live_x"}, "transport": {"type": "webrtc", "sdp": "answer"}}).encode())

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {"voice": "willow", "instructions": "Speak Spanish."})
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://api.example/v1"))

        result = voice_live.create_webrtc_session("v=0 offer", history=[{"type": "message", "role": "user", "content": []}])

        assert result["transport"]["sdp"] == "answer"
        assert captured["url"] == "https://api.example/v1/live/sessions"
        assert captured["auth"] == "Bearer sk-test"
        session = captured["body"]["session"]
        assert session["delegation"] == {"type": "client"}
        assert session["audio"]["output"]["voice"] == "willow"
        assert session["instructions"].endswith("Speak Spanish.")
        assert session["input"][0]["role"] == "user"
        assert captured["body"]["transport"] == {"type": "webrtc", "sdp": "v=0 offer"}
        assert "sk-test" not in json.dumps(result)

    def test_missing_key_refuses_before_any_network(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("", voice_live.DEFAULT_LIVE_BASE_URL))
        monkeypatch.setattr(voice_live.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the vendor"))

        with pytest.raises(ValueError):
            voice_live.create_webrtc_session("v=0 offer")
        status = voice_live.resolve_gpt_live_status()
        assert status["available"] is False
        assert status["busy_delegation_mode"] == "interrupt"

    def test_status_exposes_validated_profile_busy_policy(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_voice_section", lambda: {
            "voice_chat_mode": "gpt-live",
            "gpt_live": {"busy_delegation_mode": "queue"},
        })
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("sk-test", voice_live.DEFAULT_LIVE_BASE_URL))

        assert voice_live.resolve_gpt_live_status()["busy_delegation_mode"] == "queue"

        monkeypatch.setattr(voice_live, "_voice_section", lambda: {
            "gpt_live": {"busy_delegation_mode": "steer"},
        })
        assert voice_live.resolve_gpt_live_status()["busy_delegation_mode"] == "interrupt"


class TestVoiceLiveTurnNote:
    @pytest.fixture
    def busy_session(self):
        session = _session()
        server._sessions["sid"] = session
        yield session
        server._sessions.pop("sid", None)

    def test_busy_live_surface_queues_when_profile_opts_in_even_if_client_omits_queued(self, busy_session, monkeypatch):
        class _RedirectingAgent:
            valid_tool_names = set()
            _supports_active_turn_redirect = True

            def __init__(self):
                self.redirected = []

            def redirect(self, text):
                self.redirected.append(text)
                return True

        agent = _RedirectingAgent()
        busy_session["agent"] = agent
        monkeypatch.setattr(server, "_voice_live_busy_delegation_mode", lambda session: "queue")

        response = server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "queue me", "surface": "voice-live"})

        assert response["result"]["status"] == "queued"
        assert agent.redirected == []
        assert busy_session["queued_prompt"]["text"] == "queue me"

    def test_busy_live_surface_reads_the_session_profiles_real_queue_policy(self, busy_session, tmp_path):
        class _RedirectingAgent:
            valid_tool_names = set()
            _supports_active_turn_redirect = True

            def redirect(self, text):
                pytest.fail(f"queue policy must not redirect: {text}")

        profile_home = tmp_path / "queue-profile"
        profile_home.mkdir()
        (profile_home / "config.yaml").write_text(
            "voice:\n  gpt_live:\n    busy_delegation_mode: queue\n",
            encoding="utf-8",
        )
        busy_session["agent"] = _RedirectingAgent()
        busy_session["profile_home"] = str(profile_home)

        response = server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "profile queue", "surface": "voice-live"})

        assert response["result"]["status"] == "queued"
        assert busy_session["queued_prompt"]["text"] == "profile queue"

    def test_busy_live_surface_preserves_canonical_interrupt_default(self, busy_session, monkeypatch):
        class _RedirectingAgent:
            valid_tool_names = set()
            _supports_active_turn_redirect = True

            def __init__(self):
                self.redirected = []

            def redirect(self, text):
                self.redirected.append(text)
                return True

        agent = _RedirectingAgent()
        busy_session["agent"] = agent
        monkeypatch.setattr(server, "_voice_live_busy_delegation_mode", lambda session: "interrupt")

        response = server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "replace this", "surface": "voice-live"})

        assert response["result"]["status"] == "redirected"
        assert agent.redirected == ["replace this"]
        assert busy_session.get("queued_prompt") is None

    def test_repeated_busy_live_request_is_not_silently_deduplicated(self, busy_session, monkeypatch):
        monkeypatch.setattr(server, "_voice_live_busy_delegation_mode", lambda session: "queue")
        busy_session["inflight_turn"] = {"user": "repeat this"}

        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "repeat this", "surface": "voice-live"})
        server._methods["prompt.submit"](
            "r2", {"session_id": "sid", "text": "repeat this", "surface": "voice-live"})

        assert busy_session["queued_prompt"]["text"] == "repeat this"
        assert busy_session["queued_prompts"][0]["text"] == "repeat this"

    def test_each_busy_live_delegation_keeps_its_own_surface_context_envelope(self, busy_session, monkeypatch):
        monkeypatch.setattr(server, "_voice_live_busy_delegation_mode", lambda session: "queue")
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "first", "surface": "voice-live",
                   "voice_context": "FIRST-CONTEXT"})
        server._methods["prompt.submit"](
            "r2", {"session_id": "sid", "text": "second", "surface": "voice-live",
                   "voice_context": "SECOND-CONTEXT"})

        first = busy_session["queued_prompt"]
        second = busy_session["queued_prompts"][0]
        assert (first["text"], first["client_surface"], first["voice_live_context"]) == (
            "first", "voice-live", "FIRST-CONTEXT")
        assert (second["text"], second["client_surface"], second["voice_live_context"]) == (
            "second", "voice-live", "SECOND-CONTEXT")

    def test_live_surface_recorded_and_noted_with_spoken_context(self, busy_session):
        """The persisted row is the user's words; the transcript window reaches the model only."""
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "what's the weather", "queued": True, "surface": "voice-live",
                   "voice_context": "Voice assistant: Hi\nUser: what's the weather"})

        assert busy_session["client_surface"] == "voice-live"
        note = server._hud_surface_note(busy_session)
        assert note.startswith(voice_live.VOICE_LIVE_TURN_NOTE)
        assert "spoken" in note and "no markdown" in note
        assert "User: what's the weather" in note

    def test_long_prompt_is_untouched_while_context_cap_keeps_the_newest_context(self, busy_session):
        beginning = "BEGIN-LONG-REQUEST"
        middle = "MIDDLE-LONG-REQUEST"
        end = "END-LONG-REQUEST"
        prompt = f"{beginning} {'alpha ' * 1200}{middle} {'omega ' * 1200}{end}"
        newest_context = "NEWEST-CONTEXT-MARKER"
        context = f"OLDEST-CONTEXT-MARKER {'old ' * 2000}\nVoice assistant: {'new ' * 1000}{newest_context}"

        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": prompt, "queued": True, "surface": "voice-live",
                   "voice_context": context})

        assert beginning in busy_session["queued_prompt"]["text"]
        assert middle in busy_session["queued_prompt"]["text"]
        assert end in busy_session["queued_prompt"]["text"]
        assert len(busy_session["voice_live_context"]) <= 6000
        assert newest_context in busy_session["voice_live_context"]
        assert "OLDEST-CONTEXT-MARKER" not in busy_session["voice_live_context"]

    def test_voice_context_ignored_off_the_live_surface(self, busy_session):
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "x", "queued": True, "voice_context": "User: smuggled"})

        assert busy_session["voice_live_context"] == ""
        assert server._hud_surface_note(busy_session) == ""

    def test_plain_window_submit_clears_the_live_surface(self, busy_session):
        server._methods["prompt.submit"]("r1", {"session_id": "sid", "text": "x", "queued": True, "surface": "voice-live"})
        server._methods["prompt.submit"]("r2", {"session_id": "sid", "text": "y", "queued": True})

        assert busy_session["client_surface"] == ""
        assert server._hud_surface_note(busy_session) == ""
