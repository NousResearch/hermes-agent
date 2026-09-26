"""Invariants for sizing Gemini TTS requests: response budget and composed-prompt headroom.

Gemini's ``:generateContent`` TTS endpoint is not streaming — the whole clip comes back in one
base64 PCM body, read through a bounded reader. A 4,510-character reply measured an 18.5 MB body.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_BASE_URL", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_gemini_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": "audio/L16;codec=pcm;rate=24000",
            "data": base64.b64encode(b"\x00" * 4800).decode()}}]}}]
    }
    return resp


def test_one_gemini_request_fits_the_bounded_response_reader():
    """The Gemini chunk cap and the response-body limit must stay consistent with each other.

    Gemini answers with base64 24kHz/16-bit/mono PCM: 48,000 B/s of audio, ~64,000 B/s encoded.
    A cap that can produce more audio than ``TTS_RESPONSE_BODY_LIMIT_BYTES`` holds turns a
    finished synthesis into "response exceeds N bytes" after the caller already waited for it —
    which a context-window-derived cap of 32000 allowed.
    """
    from tools.tts_tool_delivery import PROVIDER_MAX_TEXT_LENGTH
    from tools.tts_tool_providers import TTS_RESPONSE_BODY_LIMIT_BYTES

    # A deliberately slow delivery: 10 transcript characters per second of audio is what an
    # audio tag like "[very slow]" can produce.
    worst_case_audio_seconds = PROVIDER_MAX_TEXT_LENGTH["gemini"] / 10
    assert worst_case_audio_seconds * 64000 < TTS_RESPONSE_BODY_LIMIT_BYTES


def test_a_chunk_the_splitter_produced_survives_persona_and_audio_tag_expansion(
    tmp_path, monkeypatch, mock_gemini_response
):
    """Chunking must not hand the provider text that its own prompt guard then rejects.

    ``max_text_length`` caps the transcript the splitter emits, but the request also carries the
    persona direction, the preamble and whatever the audio-tag rewrite adds. Comparing the composed
    prompt against the *chunk* cap made every cap small enough to split usefully raise instead.
    """
    from tools.tts_tool import _generate_gemini_tts
    from tools.tts_tool_delivery import _split_text_for_tts

    persona = tmp_path / "voice-persona.md"
    persona.write_text("AUDIO PROFILE\n" + ("director note " * 40) + "\n{{transcript}}\n")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "tools.tts_tool_providers._rewrite_gemini_tts_audio_tags",
        lambda text, persona_prompt="": "[warmly] " + text + " [sighs]")

    cap = 1500
    config = {"gemini": {
        "model": "gemini-3.1-flash-tts-preview", "audio_tags": True,
        "max_text_length": cap, "persona_prompt_file": str(persona)}}
    chunk = _split_text_for_tts("word " * 4000, cap)[0]

    with patch("requests.post", return_value=mock_gemini_response) as mock_post:
        _generate_gemini_tts(chunk, str(tmp_path / "out.wav"), config)

    sent = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
    assert chunk in sent
    assert len(sent) > cap, "the test is only meaningful if the prompt outgrows the chunk cap"


def test_a_prompt_over_the_provider_request_ceiling_still_raises(tmp_path, monkeypatch):
    """The guard still protects Gemini's real per-request limit, before any network call."""
    from tools.tts_tool import _generate_gemini_tts
    from tools.tts_tool_providers import GEMINI_MAX_REQUEST_CHARS

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with patch("requests.post") as mock_post:
        with pytest.raises(ValueError, match="request limit"):
            _generate_gemini_tts("a" * (GEMINI_MAX_REQUEST_CHARS + 1),
                                 str(tmp_path / "out.wav"), {"gemini": {}})
    mock_post.assert_not_called()
