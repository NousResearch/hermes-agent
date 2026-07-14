"""Tests for the DeepInfra STT provider.

``_transcribe_deepinfra`` is a thin shim that resolves credentials/model
then delegates to ``_transcribe_openai``. These two tests pin the
STT-specific gating (so an unset DEEPINFRA_API_KEY refuses dispatch) and
the delegation happy path; shared catalog/tag-filter behavior is covered
in ``tests/hermes_cli/test_api_key_providers.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    import hermes_cli.models as _models_mod
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_cache", {})
    yield


def test_get_provider_gating_keys_on_deepinfra_api_key(monkeypatch):
    """Explicit-provider gate: DEEPINFRA_API_KEY presence flips ``deepinfra`` on/off."""
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    from tools.transcription_tools import _get_provider
    assert _get_provider({"provider": "deepinfra"}) == "none"
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    assert _get_provider({"provider": "deepinfra"}) == "deepinfra"


def test_delegates_to_openai_handler_with_deepinfra_creds(monkeypatch, tmp_path):
    """Happy path: pinned model → openai SDK invoked with DeepInfra base_url + key,
    and the response carries ``provider="deepinfra"`` (not openai)."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            transcriptions = MagicMock()
            transcriptions.create = MagicMock(return_value=MagicMock(text="ok"))
            self.audio = MagicMock(transcriptions=transcriptions)
        def close(self):
            pass

    fake_openai = MagicMock()
    fake_openai.OpenAI = _FakeClient
    fake_openai.APIError = Exception
    fake_openai.APIConnectionError = ConnectionError
    fake_openai.APITimeoutError = TimeoutError

    with patch.dict("sys.modules", {"openai": fake_openai}), \
         patch("tools.transcription_tools._load_stt_config", return_value={}):
        from tools.transcription_tools import _transcribe_deepinfra
        result = _transcribe_deepinfra(str(audio), "vendor/test-stt")

    assert result["success"] is True
    assert result["provider"] == "deepinfra"
    assert "deepinfra" in captured["base_url"]
    assert captured["api_key"] == "test-key"
