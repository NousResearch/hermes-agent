"""Tests for self-hosted / OpenAI-compatible TTS via ``tts.openai`` config.

Covers the config-supplied ``api_key`` (previously never read) and the
keyless-self-hosted contract: a configured ``tts.openai.base_url`` is
authoritative and may run without any key (a placeholder Bearer is sent),
mirroring the STT side.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


SELF = "http://192.168.1.50:8001/v1"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _fake_client_capturing(captured):
    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            speech = MagicMock()
            speech.create = MagicMock(
                return_value=MagicMock(stream_to_file=lambda p: None)
            )
            self.audio = MagicMock(speech=speech)

        def close(self):
            pass

    return _FakeClient


def test_config_base_url_honored_keyless(monkeypatch, tmp_path):
    """No key anywhere + config base_url → placeholder Bearer, self-hosted base."""
    captured: dict = {}
    from tools import tts_tool
    from tools.tts_tool import _generate_openai_tts, _PLACEHOLDER_OPENAI_KEY

    with patch.object(tts_tool, "_import_openai_client",
                      return_value=_fake_client_capturing(captured)):
        _generate_openai_tts(
            "hallo", str(tmp_path / "out.mp3"),
            {"openai": {"base_url": SELF, "model": "qwen3-tts", "voice": "Serena"}},
        )

    assert captured["base_url"] == SELF
    assert captured["api_key"] == _PLACEHOLDER_OPENAI_KEY


def test_config_api_key_is_read(monkeypatch, tmp_path):
    """``tts.openai.api_key`` (previously ignored) is used as the Bearer."""
    captured: dict = {}
    from tools import tts_tool
    from tools.tts_tool import _generate_openai_tts

    with patch.object(tts_tool, "_import_openai_client",
                      return_value=_fake_client_capturing(captured)):
        _generate_openai_tts(
            "hi", str(tmp_path / "out.mp3"),
            {"openai": {"base_url": SELF, "api_key": "sk-local", "model": "m"}},
        )

    assert captured["api_key"] == "sk-local"
    assert captured["base_url"] == SELF


def test_env_key_not_sent_to_private_base_url(monkeypatch, tmp_path):
    """An env OpenAI key (set for chat) must not travel to a private/LAN
    base_url — the placeholder is sent instead."""
    captured: dict = {}
    from tools import tts_tool
    from tools.tts_tool import _generate_openai_tts, _PLACEHOLDER_OPENAI_KEY

    with patch.object(tts_tool, "_import_openai_client",
                      return_value=_fake_client_capturing(captured)), \
         patch.object(tts_tool, "resolve_openai_audio_api_key",
                      return_value="sk-real-openai-key"):
        _generate_openai_tts(
            "hallo", str(tmp_path / "out.mp3"),
            {"openai": {"base_url": SELF, "model": "m", "voice": "v"}},
        )

    assert captured["base_url"] == SELF
    assert captured["api_key"] == _PLACEHOLDER_OPENAI_KEY


def test_env_key_still_used_for_public_https_base_url(monkeypatch, tmp_path):
    """A public https OpenAI-compatible proxy keeps the env key."""
    captured: dict = {}
    public = "https://tts-proxy.example.com/v1"
    from tools import tts_tool
    from tools.tts_tool import _generate_openai_tts

    with patch.object(tts_tool, "_import_openai_client",
                      return_value=_fake_client_capturing(captured)), \
         patch.object(tts_tool, "resolve_openai_audio_api_key",
                      return_value="sk-proxy-key"):
        _generate_openai_tts(
            "hallo", str(tmp_path / "out.mp3"),
            {"openai": {"base_url": public, "model": "m", "voice": "v"}},
        )

    assert captured["base_url"] == public
    assert captured["api_key"] == "sk-proxy-key"


def test_requirements_true_for_self_hosted_without_env_key(monkeypatch):
    """check_tts_requirements() is True for a config base_url even with no env key."""
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool, "_load_tts_config",
        lambda: {"provider": "openai", "openai": {"base_url": SELF, "model": "m"}},
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)

    assert tts_tool.check_tts_requirements() is True


def test_requirements_true_for_config_api_key_only(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool, "_load_tts_config",
        lambda: {"provider": "openai", "openai": {"api_key": "sk-local", "model": "m"}},
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)

    assert tts_tool.check_tts_requirements() is True
