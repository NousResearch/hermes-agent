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

import base64
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

    def test_codex_model_uses_subscription_oauth_call(self, monkeypatch):
        """The Codex-only model uses the ChatGPT subscription endpoint and account
        header while preserving the renderer's existing normalized answer shape."""
        captured = {}

        class _Headers(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        class _Resp:
            headers = _Headers({"Location": "/backend-api/codex/realtime/calls/rtc_live_123"})

            def read(self):
                return b"v=0 answer\r\n"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = {key.lower(): value for key, value in req.header_items()}
            captured["body"] = json.loads(req.data)
            return _Resp()

        claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        token = f"header.{payload}.signature"
        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            voice_live, "_live_section",
            lambda voice=None: {"model": "gpt-live-1-codex", "voice": "cove"})
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_codex_runtime_credentials",
            lambda: {"api_key": token})

        result = voice_live.create_webrtc_session("v=0 offer")

        assert captured["url"] == voice_live.CODEX_LIVE_CALL_URL
        assert captured["headers"]["authorization"] == f"Bearer {token}"
        assert captured["headers"]["chatgpt-account-id"] == "acct-test"
        assert captured["headers"]["originator"] == "hermes-agent"
        assert captured["headers"]["openai-alpha"] == "quicksilver=v2"
        assert captured["headers"]["session-id"]
        assert captured["headers"]["thread-id"]
        assert captured["headers"]["x-session-id"]
        assert captured["body"]["sdp"] == "v=0 offer"
        assert captured["body"]["session"]["model"] == "gpt-live-1-codex"
        assert captured["body"]["session"]["delegation"] == {"type": "client"}
        assert result == {
            "session": {"id": "rtc_live_123"},
            "transport": {"type": "webrtc", "sdp": "v=0 answer\r\n"},
        }
        assert token not in json.dumps(result)

    def test_codex_model_requires_subscription_login_before_network(self, monkeypatch):
        monkeypatch.setattr(
            voice_live, "_live_section", lambda voice=None: {"model": "gpt-live-1-codex"})
        monkeypatch.setattr(
            voice_live, "_resolve_codex_subscription_credentials", lambda: ("", ""))
        monkeypatch.setattr(
            voice_live.urllib.request, "urlopen",
            lambda *a, **k: pytest.fail("must not call the vendor"))

        with pytest.raises(ValueError, match="hermes auth add openai-codex"):
            voice_live.create_webrtc_session("v=0 offer")
        status = voice_live.resolve_gpt_live_status()
        assert status["available"] is False
        assert "openai-codex" in status["reason"]

    def test_missing_key_refuses_before_any_network(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("", voice_live.DEFAULT_LIVE_BASE_URL))
        monkeypatch.setattr(voice_live.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the vendor"))

        with pytest.raises(ValueError):
            voice_live.create_webrtc_session("v=0 offer")
        assert voice_live.resolve_gpt_live_status()["available"] is False


class TestVoiceLiveTurnNote:
    @pytest.fixture
    def busy_session(self):
        session = _session()
        server._sessions["sid"] = session
        yield session
        server._sessions.pop("sid", None)

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
