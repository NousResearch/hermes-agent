import sys
import types
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from tools import transcription_tools


def _write_dummy_audio(path: Path) -> None:
    path.write_bytes(b"OggSdummy")


def test_transcribe_audio_uses_openai_api_key_fallback_when_voice_tools_key_missing(monkeypatch, tmp_path):
    audio_file = tmp_path / "sample.ogg"
    _write_dummy_audio(audio_file)

    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)

    class FakeAPIError(Exception):
        pass

    class FakeAPIConnectionError(Exception):
        pass

    class FakeAPITimeoutError(Exception):
        pass

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            assert api_key == "openai-key"
            assert base_url == "https://api.openai.com/v1"
            self.audio = types.SimpleNamespace(
                transcriptions=types.SimpleNamespace(
                    create=lambda **kwargs: "hello from openai"
                )
            )

    fake_openai_module = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=FakeAPIError,
        APIConnectionError=FakeAPIConnectionError,
        APITimeoutError=FakeAPITimeoutError,
    )

    with patch.dict(sys.modules, {"openai": fake_openai_module}):
        result = transcription_tools.transcribe_audio(str(audio_file))

    assert result["success"] is True
    assert result["transcript"] == "hello from openai"


def test_transcribe_audio_falls_back_to_local_command_when_openai_keys_missing(monkeypatch, tmp_path):
    audio_file = tmp_path / "voice.ogg"
    _write_dummy_audio(audio_file)

    out_dir = tmp_path / "local-out"
    out_dir.mkdir()

    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "HERMES_LOCAL_STT_COMMAND",
        "mlx-whisper-local --input {input_path} --output-dir {output_dir} --language {language}",
    )
    monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "en")
    monkeypatch.setattr(transcription_tools, "_find_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

    @contextmanager
    def fake_tempdir(prefix=None):
        yield str(out_dir)

    monkeypatch.setattr(transcription_tools.tempfile, "TemporaryDirectory", fake_tempdir)

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            # ffmpeg conversion step -> create the converted wav
            Path(cmd[-1]).write_bytes(b"RIFF....WAVEfmt ")
            return CompletedProcess(cmd, 0, stdout="", stderr="")

        (out_dir / "voice.txt").write_text("hello from local whisper\n")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(transcription_tools.subprocess, "run", fake_run)

    result = transcription_tools.transcribe_audio(str(audio_file))

    assert result["success"] is True
    assert result["transcript"] == "hello from local whisper"


def test_transcribe_audio_reports_clear_error_when_no_backends_are_configured(monkeypatch, tmp_path):
    audio_file = tmp_path / "sample.ogg"
    _write_dummy_audio(audio_file)

    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)

    result = transcription_tools.transcribe_audio(str(audio_file))

    assert result["success"] is False
    assert "HERMES_LOCAL_STT_COMMAND" in result["error"]
    assert "OPENAI_API_KEY" in result["error"]
