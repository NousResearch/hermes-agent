"""Tests for OpenAI-compatible STT prompt-file and hotword support."""

from unittest.mock import MagicMock, patch


def test_openai_transcription_forwards_prompt_file_and_hotwords(monkeypatch, tmp_path):
    from tools import transcription_tools as tt

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Prompt depuis fichier.\n", encoding="utf-8")

    monkeypatch.setattr(
        tt,
        "_load_stt_config",
        lambda: {
            "openai": {
                "prompt_file": str(prompt_file),
                "hotwords": "Hermes, Hephaistos, Speaches, high",
            }
        },
    )

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = {"text": "ok"}

    with (
        patch.object(tt, "_HAS_OPENAI", True),
        patch("openai.OpenAI", return_value=mock_client),
    ):
        result = tt._transcribe_openai(
            str(audio),
            "Systran/faster-whisper-large-v3",
            api_key="test-key",
            base_url="http://speaches.test/v1",
        )

    assert result["success"] is True
    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["prompt"] == "Prompt depuis fichier."
    assert kwargs["extra_body"] == {
        "hotwords": "Hermes, Hephaistos, Speaches, high",
    }


def test_threaded_prompt_takes_precedence_over_prompt_file(monkeypatch, tmp_path):
    from tools import transcription_tools as tt

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")
    openai_cfg = {"prompt_file": str(prompt_file)}

    assert tt._get_openai_stt_prompt(openai_cfg, "hook prompt") == "hook prompt"


def test_missing_prompt_file_fails_open(monkeypatch, tmp_path, caplog):
    from tools import transcription_tools as tt

    missing = tmp_path / "missing.txt"

    assert tt._get_openai_stt_prompt({"prompt_file": str(missing)}) is None
    assert "could not be read" in caplog.text


def test_shared_openai_helper_does_not_leak_provider_specific_hints(
    monkeypatch, tmp_path
):
    from tools import transcription_tools as tt

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Speaches-only prompt", encoding="utf-8")
    monkeypatch.setattr(
        tt,
        "_load_stt_config",
        lambda: {
            "openai": {
                "prompt_file": str(prompt_file),
                "hotwords": "Speaches-only hotwords",
            }
        },
    )

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = {"text": "ok"}

    with (
        patch.object(tt, "_HAS_OPENAI", True),
        patch("openai.OpenAI", return_value=mock_client),
    ):
        result = tt._transcribe_openai(
            str(audio),
            "whisper-large-v3",
            api_key="test-key",
            base_url="http://deepinfra.test/v1",
            provider_label="deepinfra",
        )

    assert result["success"] is True
    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert "prompt" not in kwargs
    assert "extra_body" not in kwargs
