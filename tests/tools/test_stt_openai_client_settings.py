"""The OpenAI SDK transcription client honors stt.<provider>.timeout / max_retries (#112939).

The constructor used to hardcode timeout=30 and max_retries=0, and the SDK has no env layer
for these values (openai 2.x reads no OPENAI_TIMEOUT / OPENAI_MAX_RETRIES), so a self-hosted
OpenAI-compatible endpoint whose model cold start exceeds 30s (measured 35.3s on
parakeet-mlx) lost the voice message on the first attempt with no retry and no user-side
fix. These pin the configurable behavior: defaults, per-provider override over the openai
fallback, quoted-numeric coercion like every other stt numeric key, boolean rejection,
httpx zero-means-no-timeout, and negative handling.
"""
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _sample_wav(tmp_path):
    """Minimal valid WAV (1s of silence at 16kHz), same shape as the sibling suite's fixture."""
    import wave
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    with wave.open(str(wav_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return str(wav_path)


def _client_kwargs(monkeypatch, tmp_path, stt_config, config_section=None):
    """Drive _with_openai_client through the Groq rider; return the OpenAI constructor kwargs."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = "hi"
    openai_cls = MagicMock(return_value=mock_client)
    with patch("tools.transcription_tools._HAS_OPENAI", True), \
         patch("openai.OpenAI", openai_cls), \
         patch("tools.transcription_tools._load_stt_config", return_value=stt_config):
        from tools.transcription_cloud import _with_openai_client
        result = _with_openai_client(
            "gsk-test", "https://api.groq.com/openai/v1", _sample_wav(tmp_path), "Groq",
            lambda client: "hi", config_section=config_section or "groq")
    assert result == "hi"
    return openai_cls.call_args.kwargs


@pytest.mark.parametrize("stt_config, want_timeout, want_retries", [
    ({"groq": None}, 60, 1),                                        # unset: defaults
    ({"openai": {"timeout": 120, "max_retries": 2}}, 120, 2),      # openai fallback applies to riders
    ({"groq": {"timeout": 45}, "openai": {"timeout": 120}}, 45, 1),  # provider section beats the fallback
    ({"openai": {"timeout": "120"}}, 120.0, 1),                     # quoted numerics coerce (stt idiom)
    ({"openai": {"max_retries": 2.0}}, 60, 2),                      # integral float retries coerce
    ({"openai": {"timeout": 0}}, None, 1),                          # httpx semantics: 0 = no timeout
    ({"openai": {"timeout": -5}}, 60, 1),                           # negative timeout falls back
    ({"openai": {"max_retries": -3}}, 60, 0),                       # negative retries = no retries
    ({"openai": {"timeout": 30, "max_retries": 0}}, 30, 0),         # the old shape round-trips
])
def test_transport_settings_matrix(monkeypatch, tmp_path, stt_config, want_timeout, want_retries):
    kwargs = _client_kwargs(monkeypatch, tmp_path, stt_config)
    assert kwargs["timeout"] == want_timeout
    assert kwargs["max_retries"] == want_retries


def test_booleans_are_rejected_with_the_previous_value_kept(monkeypatch, tmp_path, caplog):
    kwargs = _client_kwargs(
        monkeypatch, tmp_path, {"openai": {"timeout": True, "max_retries": False}})
    assert kwargs["timeout"] == 60 and kwargs["max_retries"] == 1  # True is not a timeout


def test_discarded_values_are_logged(monkeypatch, tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="tools.transcription_tools"):
        _client_kwargs(monkeypatch, tmp_path, {"openai": {"timeout": -5}})
    assert any("negative" in r.message for r in caplog.records)
