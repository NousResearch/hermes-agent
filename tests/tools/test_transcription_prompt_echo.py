"""A transcript that only echoes stt.prompt is decoded again without the prompt (#155798)."""
from unittest.mock import patch

import pytest

PROMPT = "Hermes, Teknium, Nous Research, kanban, Ollama"


def _transcribe(tmp_path, answers, prompt=PROMPT):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    calls = []

    def fake_groq(file_path, model_name, *, language=None, prompt=None):
        calls.append(prompt)
        return answers[len(calls) - 1]

    config = {"provider": "groq", "enabled": True}
    if prompt is not None:
        config["prompt"] = prompt
    with patch("tools.transcription_tools._load_stt_config", return_value=config), \
         patch("tools.transcription_tools._get_provider", return_value="groq"), \
         patch("tools.transcription_tools._validate_audio_file", return_value=None), \
         patch("tools.transcription_tools._read_block_error", return_value=None), \
         patch("tools.transcription_tools._trim_silence_for_cloud_stt", return_value=None), \
         patch("tools.transcription_tools.exceeds_upload_limit", return_value=False), \
         patch("tools.transcription_tools._transcribe_groq", side_effect=fake_groq):
        from tools.transcription_tools import _transcribe_prepared_audio
        return _transcribe_prepared_audio(str(wav)), calls


def _ok(text):
    return {"success": True, "transcript": text, "provider": "groq"}


@pytest.mark.parametrize("echo", [PROMPT, "hermes teknium nous research kanban ollama.", " Hermes, Teknium, Nous Research, kanban, Ollama "])
def test_an_echoed_prompt_is_retried_without_the_prompt(tmp_path, echo):
    result, calls = _transcribe(tmp_path, [_ok(echo), _ok("")])
    assert calls == [PROMPT, None]
    assert result["transcript"] == ""


def test_a_real_transcript_is_kept_and_not_retried(tmp_path):
    result, calls = _transcribe(tmp_path, [_ok("Ask Hermes to move the kanban card to done")])
    assert calls == [PROMPT]
    assert result["transcript"] == "Ask Hermes to move the kanban card to done"


def test_no_prompt_means_no_echo_check(tmp_path):
    result, calls = _transcribe(tmp_path, [_ok(PROMPT)], prompt=None)
    assert calls == [None]
    assert result["transcript"] == PROMPT


def test_a_failed_decode_is_not_retried(tmp_path):
    failure = {"success": False, "transcript": "", "error": "boom"}
    result, calls = _transcribe(tmp_path, [failure])
    assert calls == [PROMPT]
    assert result is failure
