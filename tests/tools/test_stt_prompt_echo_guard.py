"""A hosted whisper-family transcript that copies its own ``stt.prompt`` is retried once without it.

The real ``transcribe_audio`` runs; only the provider handler is replaced.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools import transcription_tools

VOCAB = "Hermes, Teknium, Nous Research, kanban, Ollama"  # the configuration docs example
LONG_VOCAB = ", ".join([VOCAB] * 6)  # over 200 characters once normalized
REAL_SPEECH = "Could you summarise the telemetry section before the call"


@pytest.mark.parametrize(("provider", "prompt", "transcripts", "retried"), [
    pytest.param("groq", VOCAB, ("Q" + VOCAB, REAL_SPEECH), True, id="echo"),
    pytest.param("groq", LONG_VOCAB, ("Q" + LONG_VOCAB, REAL_SPEECH), True, id="long-echo"),
    pytest.param("groq", VOCAB, ("Q" + VOCAB, "Q" + VOCAB), True, id="unbiased-retry-repeats-it"),
    pytest.param("groq", VOCAB, ("Please use Nous Research Hermes and Ollama", REAL_SPEECH), False,
                 id="speech-reordering-the-vocabulary"),
    pytest.param("groq", "PostgreSQL, OpenTelemetry", ("Use PostgreSQL and OpenTelemetry", REAL_SPEECH),
                 False, id="speech-adding-words-to-the-vocabulary"),
    pytest.param("xai", VOCAB, (VOCAB, REAL_SPEECH), False, id="backend-ignores-the-prompt"),
])
def test_only_a_copy_of_the_prompt_is_retried_without_it(tmp_path, provider, prompt, transcripts, retried):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fake audio data")
    prompts = []

    def handler(file_path, model_name, *, language=None, prompt=None):
        prompts.append(prompt)
        return {"success": True, "transcript": transcripts[len(prompts) - 1], "provider": provider}

    config = {"provider": provider, "prompt": prompt, "cloud_trim_silence": False}
    with patch.object(transcription_tools, "_load_stt_config", return_value=config), \
         patch.object(transcription_tools, "_get_provider", return_value=provider), \
         patch.object(transcription_tools, f"_transcribe_{provider}", side_effect=handler):
        result = transcription_tools.transcribe_audio(str(audio))

    assert prompts == ([prompt, None] if retried else [prompt])
    assert result["transcript"] == transcripts[1 if retried else 0]
