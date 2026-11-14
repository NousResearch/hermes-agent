"""Grok Voice minting: OAuth bearer first, API key only as fallback, secret never echoes the bearer.

No network. The poster stands in for ``POST /realtime/client_secrets``.
"""

import json

import pytest

from tools import grok_voice
from tools import voice_live


OAUTH_BEARER = "oauth-bearer-not-the-secret"
API_KEY = "xai-api-key-not-the-secret"
MINTED = "ephemeral-client-secret"


def _ok_poster(captured):
    def poster(url, headers, payload, timeout):
        captured.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return 200, {"value": MINTED, "expires_at": 1_700_000_000}

    return poster


def _install_oauth(monkeypatch, token=OAUTH_BEARER, base="https://api.x.ai/v1"):
    def resolve(**_kwargs):
        return {"api_key": token, "base_url": base}

    monkeypatch.setattr(
        "hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials", resolve,
    )


def _install_api_key(monkeypatch, key=API_KEY):
    def get_secret(name, default=None):
        if name == "XAI_API_KEY":
            return key
        return default

    monkeypatch.setattr("agent.secret_scope.get_secret", get_secret)


def test_oauth_is_used_even_when_an_api_key_is_set(monkeypatch):
    captured = []
    _install_oauth(monkeypatch)

    def get_secret(name, default=None):
        raise AssertionError(f"profile env {name} must not be read when OAuth resolves")

    monkeypatch.setattr("agent.secret_scope.get_secret", get_secret)
    monkeypatch.setattr(grok_voice, "_grok_section", dict)

    result = grok_voice.mint_grok_voice_session(_ok_poster(captured))

    assert captured[0]["headers"]["Authorization"] == f"Bearer {OAUTH_BEARER}"
    assert result["client_secret"]["value"] == MINTED
    assert grok_voice.resolve_grok_voice_status()["auth"] == "oauth"


def test_api_key_is_used_only_when_oauth_is_absent(monkeypatch):
    captured = []
    _install_oauth(monkeypatch, token="")
    _install_api_key(monkeypatch)
    monkeypatch.setattr(grok_voice, "_grok_section", dict)

    result = grok_voice.mint_grok_voice_session(_ok_poster(captured))

    assert captured[0]["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert result["client_secret"]["value"] == MINTED
    assert grok_voice.resolve_grok_voice_status()["auth"] == "api_key"


def test_api_key_is_used_when_oauth_raises(monkeypatch):
    captured = []

    def explode(**_kwargs):
        raise RuntimeError(f"refresh failed for {OAUTH_BEARER}")

    monkeypatch.setattr("hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials", explode)
    _install_api_key(monkeypatch)
    monkeypatch.setattr(grok_voice, "_grok_section", dict)

    result = grok_voice.mint_grok_voice_session(_ok_poster(captured))

    assert captured[0]["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert OAUTH_BEARER not in json.dumps(result)


def test_secret_equal_to_the_bearer_is_rejected(monkeypatch):
    _install_oauth(monkeypatch)
    monkeypatch.setattr(grok_voice, "_grok_section", dict)

    def poster(url, headers, payload, timeout):
        return 200, {"client_secret": {"value": OAUTH_BEARER, "expires_at": 5}}

    with pytest.raises(grok_voice.GrokVoiceError) as raised:
        grok_voice.mint_grok_voice_session(poster)

    assert raised.value.status == 502
    assert OAUTH_BEARER not in str(raised.value)
    assert OAUTH_BEARER not in raised.value.message


def test_bearer_never_appears_in_errors_or_the_session(monkeypatch):
    _install_oauth(monkeypatch)
    monkeypatch.setattr(grok_voice, "_grok_section", lambda: {"instructions": "be brief"})

    def poster(url, headers, payload, timeout):
        return 401, {"error": f"unauthorized {OAUTH_BEARER}"}

    with pytest.raises(grok_voice.GrokVoiceError) as raised:
        grok_voice.mint_grok_voice_session(poster)

    assert raised.value.status == 502
    assert OAUTH_BEARER not in raised.value.message
    assert OAUTH_BEARER not in str(raised.value)

    def ok(url, headers, payload, timeout):
        return 200, {"value": MINTED, "expires_at": 9, "echo": OAUTH_BEARER}

    result = grok_voice.mint_grok_voice_session(ok)
    blob = json.dumps(result)
    assert OAUTH_BEARER not in blob
    assert result["client_secret"]["value"] == MINTED
    assert result["client_secret"]["value"] != OAUTH_BEARER
    assert result["session"]["voice"] == "eve"
    assert "be brief" in result["session"]["instructions"]


def test_invalid_model_and_voice_fall_back_and_never_hit_the_wire(monkeypatch):
    captured = []
    _install_oauth(monkeypatch)
    monkeypatch.setattr(
        grok_voice, "_grok_section",
        lambda: {"model": "not a model", "voice": "bad voice!", "reasoning_effort": "low"},
    )

    grok_voice.mint_grok_voice_session(_ok_poster(captured))

    request = json.dumps(captured)
    assert "not a model" not in request
    assert "bad voice!" not in request
    assert captured[0]["payload"] == {"expires_after": {"seconds": 300}}
    assert captured[0]["url"].endswith("/realtime/client_secrets")


def test_invalid_model_and_voice_are_replaced_in_the_session(monkeypatch):
    _install_oauth(monkeypatch)
    monkeypatch.setattr(
        grok_voice, "_grok_section",
        lambda: {"model": "not a model", "voice": "bad voice!", "reasoning_effort": "low"},
    )

    def poster(url, headers, payload, timeout):
        return 200, {"value": MINTED, "expires_at": 3}

    result = grok_voice.mint_grok_voice_session(poster)

    assert result["model"] == grok_voice.DEFAULT_MODEL
    assert result["url"].endswith("model=grok-voice-latest")
    assert result["session"]["voice"] == "eve"
    assert result["session"]["reasoning"] == {"effort": "none"}
    assert "not a model" not in result["url"]


def test_nested_client_secret_value_is_accepted(monkeypatch):
    _install_oauth(monkeypatch)
    monkeypatch.setattr(grok_voice, "_grok_section", dict)

    def poster(url, headers, payload, timeout):
        return 200, {"client_secret": {"value": MINTED, "expires_at": 4}}

    result = grok_voice.mint_grok_voice_session(poster)

    assert result["client_secret"] == {"value": MINTED, "expires_at": 4}
    assert result["protocol_prefix"] == "xai-client-secret."


def test_missing_credential_is_503_and_names_no_secret(monkeypatch):
    _install_oauth(monkeypatch, token="")
    _install_api_key(monkeypatch, key="")

    with pytest.raises(grok_voice.GrokVoiceError) as raised:
        grok_voice.mint_grok_voice_session(lambda *a, **k: pytest.fail("must not mint"))

    assert raised.value.status == 503
    assert raised.value.reason == "no_credential"
    status = grok_voice.resolve_grok_voice_status()
    assert status["available"] is False
    assert status["auth"] == "none"
    assert OAUTH_BEARER not in json.dumps(status)
    assert API_KEY not in json.dumps(status)


def test_voice_chat_mode_accepts_grok_live():
    assert voice_live.voice_chat_mode({"voice_chat_mode": "grok-live"}) == "grok-live"
    assert voice_live.voice_chat_mode({"voice_chat_mode": "grok_live"}) == "grok-live"
    assert voice_live.voice_chat_mode({"voice_chat_mode": "groklive"}) == "grok-live"
    assert voice_live.voice_chat_mode({"voice_chat_mode": "live"}) == "gpt-live"
    assert voice_live.voice_chat_mode({"voice_chat_mode": "gpt-live"}) == "gpt-live"
    assert voice_live.voice_chat_mode({"voice_chat_mode": "chained"}) == "chained"
    assert voice_live.voice_chat_mode({}) == "chained"


def test_status_uses_grok_availability_only_when_that_mode_is_saved(monkeypatch):
    monkeypatch.setattr(voice_live, "resolve_gpt_live_status", lambda: {
        "mode": "grok-live",
        "available": True,
        "reason": None,
        "model": "gpt-live-1",
        "voice": "marin",
    })
    monkeypatch.setattr(grok_voice, "resolve_grok_voice_status", lambda: {
        "mode": "grok-live",
        "available": False,
        "reason": "no xAI credential (hermes auth add xai-oauth or XAI_API_KEY)",
        "model": "grok-voice-latest",
        "voice": "eve",
        "auth": "none",
    })

    status = voice_live.resolve_voice_chat_status()

    assert status["mode"] == "grok-live"
    assert status["available"] is False
    assert status["model"] == "grok-voice-latest"
    assert status["voice"] == "eve"
    assert status["auth"] == "none"
    assert status["gpt_live_available"] is True
    assert OAUTH_BEARER not in json.dumps(status)

    monkeypatch.setattr(voice_live, "resolve_gpt_live_status", lambda: {
        "mode": "gpt-live",
        "available": True,
        "reason": None,
        "model": "gpt-live-1",
        "voice": "marin",
    })
    kept = voice_live.resolve_voice_chat_status()
    assert kept["available"] is True
    assert kept["model"] == "gpt-live-1"
    assert kept["grok_available"] is False
