"""Tests for GPT-Live credential resolution — the ``openai-codex`` pool fallback.

Codex OAuth access tokens are accepted by ``POST /v1/live/sessions`` (verified
against the live vendor endpoint), so a Codex subscriber can run GPT-Live voice
sessions without a separate platform API key. ``_resolve_credentials`` falls back
to the ``openai-codex`` credential pool after the standard audio-key chain
(config → ``VOICE_TOOLS_OPENAI_KEY`` → ``OPENAI_API_KEY`` → ``openai-api`` pool);
the fallback is Live-only and must not leak into ``resolve_openai_audio_api_key``,
whose STT/TTS endpoints have not been verified to accept Codex tokens.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import voice_live


def _fake_pool(key: str = "", *, has: bool = True):
    entry = SimpleNamespace(runtime_api_key=key, access_token=key) if key else None
    return SimpleNamespace(
        has_credentials=lambda: has and bool(key),
        peek=lambda: entry,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep the developer's real ~/.hermes/.env out of these tests."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_env", lambda: {})
    yield


class TestCodexPoolFallback:
    """``_resolve_credentials`` resolves a key from the ``openai-codex`` pool."""

    def test_codex_pool_used_when_no_other_key(self):
        with patch("agent.credential_pool.load_pool",
                   side_effect=lambda pid: _fake_pool("codex-token") if pid == "openai-codex" else None):
            key, base = voice_live._resolve_credentials({})
        assert key == "codex-token"
        assert base == voice_live.DEFAULT_LIVE_BASE_URL

    def test_audio_key_wins_over_codex_pool(self):
        with patch("tools.tool_backend_helpers.resolve_openai_audio_api_key",
                   return_value="platform-key"), \
             patch("agent.credential_pool.load_pool",
                   side_effect=lambda pid: _fake_pool("codex-token") if pid == "openai-codex" else None):
            key, _ = voice_live._resolve_credentials({})
        assert key == "platform-key"

    def test_explicit_config_key_wins_over_everything(self):
        with patch("tools.tool_backend_helpers.resolve_openai_audio_api_key",
                   return_value="platform-key"), \
             patch("agent.credential_pool.load_pool") as load_pool:
            key, _ = voice_live._resolve_credentials({"api_key": "explicit-key"})
        assert key == "explicit-key"
        load_pool.assert_not_called()

    def test_pool_error_is_swallowed(self):
        with patch("agent.credential_pool.load_pool", side_effect=Exception("disk error")):
            key, _ = voice_live._resolve_credentials({})
        assert key == ""

    def test_pool_without_credentials_yields_empty(self):
        with patch("agent.credential_pool.load_pool",
                   side_effect=lambda pid: _fake_pool("", has=False)):
            key, _ = voice_live._resolve_credentials({})
        assert key == ""


class TestStatusReflectsCodexPool:
    """``resolve_gpt_live_status`` reports ``available`` when only the Codex pool has a token."""

    def test_available_true_with_only_codex_pool(self):
        with patch("tools.voice_live._voice_section", return_value={}), \
             patch("tools.tool_backend_helpers.resolve_openai_audio_api_key", return_value=""), \
             patch("agent.credential_pool.load_pool",
                   side_effect=lambda pid: _fake_pool("codex-token") if pid == "openai-codex" else None):
            status = voice_live.resolve_gpt_live_status()
        assert status["available"] is True
        assert status["reason"] is None

    def test_unavailable_when_no_key_anywhere(self):
        with patch("tools.voice_live._voice_section", return_value={}), \
             patch("tools.tool_backend_helpers.resolve_openai_audio_api_key", return_value=""), \
             patch("agent.credential_pool.load_pool", side_effect=lambda pid: None):
            status = voice_live.resolve_gpt_live_status()
        assert status["available"] is False
        assert "no OpenAI API key" in (status["reason"] or "")
