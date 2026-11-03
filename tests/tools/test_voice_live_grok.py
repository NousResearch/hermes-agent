"""Unit tests for the Grok Live voice engine's config + auth path (no network, no real creds)."""

from __future__ import annotations

from tools import xai_http
import tools.voice_live_grok as vg


def _patch_credentials(monkeypatch, provider="xai-oauth", api_key="token"):
    calls = []

    def fake_resolve(prefer_api_key=False):
        calls.append(prefer_api_key)
        return {"provider": provider, "api_key": api_key}

    monkeypatch.setattr(xai_http, "resolve_xai_http_credentials", fake_resolve)
    return calls


def test_defaults():
    section = vg._grok_live_section({})
    assert section == {}
    assert vg.DEFAULT_GROK_LIVE_MODEL == "grok-voice-latest"
    assert vg.DEFAULT_GROK_LIVE_VOICE == "eve"


def test_voice_chat_mode_resolution():
    from tools.voice_live import voice_chat_mode

    assert voice_chat_mode({"voice_chat_mode": "grok-live"}) == "grok-live"
    assert voice_chat_mode({"voice_chat_mode": "gpt-live"}) == "gpt-live"
    assert voice_chat_mode({"voice_chat_mode": "chained"}) == "chained"
    assert voice_chat_mode({}) == "chained"
    assert voice_chat_mode({"voice_chat_mode": "bogus"}) == "chained"


def test_auth_auto_uses_shared_oauth_resolution(monkeypatch):
    calls = _patch_credentials(monkeypatch, provider="xai-oauth", api_key="oauth-token")
    assert vg._resolve_grok_credentials({}) == "oauth-token"
    assert calls == [False]


def test_auth_auto_accepts_shared_api_key_fallback(monkeypatch):
    _patch_credentials(monkeypatch, provider="xai", api_key="xai-test-key")
    assert vg._resolve_grok_credentials({}) == "xai-test-key"


def test_auth_oauth_rejects_api_key_fallback(monkeypatch):
    _patch_credentials(monkeypatch, provider="xai", api_key="xai-test-key")
    assert vg._resolve_grok_credentials({"auth": "oauth"}) == ""


def test_auth_apikey_restricts_and_prefers_api_key(monkeypatch):
    calls = _patch_credentials(monkeypatch, provider="xai", api_key="xai-test-key")
    assert vg._resolve_grok_credentials({"auth": "apikey"}) == "xai-test-key"
    assert calls == [True]
    _patch_credentials(monkeypatch, provider="xai-oauth", api_key="oauth-token")
    assert vg._resolve_grok_credentials({"auth": "apikey"}) == ""


def test_status_no_credential(monkeypatch):
    _patch_credentials(monkeypatch, api_key="")
    status = vg.resolve_grok_live_status()
    assert status["available"] is False
    assert "no xAI credential" in status["reason"]
    assert "mode" in status
    assert status["mode"] in {"gpt-live", "grok-live", "chained"}


def test_status_never_contains_the_key(monkeypatch):
    secret = "xai-secret-token-for-test"
    _patch_credentials(monkeypatch, provider="xai", api_key=secret)
    status = vg.resolve_grok_live_status()
    assert status["available"] is True
    assert secret not in str(status)
    assert "credential" not in status and "api_key" not in status


def test_live_instructions_includes_persona_and_backend_policy():
    text = vg.live_instructions({})
    assert "Hermes" in text
    assert "acknowledge it briefly" in text
    assert "Backend tools" in text
    assert "Delegate to the backend when" in text
    assert "Do not delegate to the backend when" in text


def test_live_instructions_appends_extras():
    text = vg.live_instructions({"instructions": "Be concise."})
    assert "Be concise." in text


def test_bridge_module_imports_and_payload_shape():
    from tools.voice_live_grok_bridge import GrokLiveBridge, session_update_payload

    assert GrokLiveBridge is not None
    payload = session_update_payload({"model": "grok-voice-latest", "voice": "eve"})
    session = payload["session"]
    assert session["voice"] == "eve"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert "tools" not in session
