"""Tests for local speech-to-text transcription helpers."""

import importlib.util
import subprocess
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "transcription_tools.py"
SPEC = importlib.util.spec_from_file_location("test_transcription_tools_module", MODULE_PATH)
tt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tt)


def _write_audio_file(tmp_path: Path, suffix: str = ".ogg") -> Path:
    audio_path = tmp_path / f"sample{suffix}"
    audio_path.write_bytes(b"not-real-audio")
    return audio_path


def _stt_config(model_path: Path) -> dict:
    return {
        "enabled": True,
        "provider": "whispercpp",
        "model": "whisper-1",
        "whispercpp": {
            "binary_path": "/usr/bin/whisper-cli",
            "model_path": str(model_path),
            "language": "auto",
            "ffmpeg_path": "/usr/bin/ffmpeg",
        },
    }


class TestWhisperCppTranscription:
    def test_transcribe_audio_success(self, tmp_path, monkeypatch):
        audio_path = _write_audio_file(tmp_path)
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"model")

        monkeypatch.setattr(tt, "resolve_stt_config", lambda: _stt_config(model_path))
        monkeypatch.setattr(tt, "resolve_whispercpp_binary", lambda config=None: "/usr/bin/whisper-cli")
        monkeypatch.setattr(tt, "resolve_ffmpeg_binary", lambda config=None: "/usr/bin/ffmpeg")

        def fake_run(command, capture_output=True, text=True):
            if command[0] == "/usr/bin/ffmpeg":
                return subprocess.CompletedProcess(command, 0, "", "")
            output_base = Path(command[command.index("-of") + 1])
            output_base.with_suffix(".txt").write_text("hello from local whisper\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(tt.subprocess, "run", fake_run)

        result = tt.transcribe_audio(str(audio_path))

        assert result["success"] is True
        assert result["provider"] == "whispercpp"
        assert result["transcript"] == "hello from local whisper"

    def test_transcribe_audio_missing_model(self, tmp_path, monkeypatch):
        audio_path = _write_audio_file(tmp_path)
        model_path = tmp_path / "missing-model.bin"

        monkeypatch.setattr(tt, "resolve_stt_config", lambda: _stt_config(model_path))
        monkeypatch.setattr(tt, "resolve_whispercpp_binary", lambda config=None: "/usr/bin/whisper-cli")
        monkeypatch.setattr(tt, "resolve_ffmpeg_binary", lambda config=None: "/usr/bin/ffmpeg")

        result = tt.transcribe_audio(str(audio_path))

        assert result["success"] is False
        assert "model not found" in result["error"]

    def test_transcribe_audio_conversion_failure(self, tmp_path, monkeypatch):
        audio_path = _write_audio_file(tmp_path)
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"model")

        monkeypatch.setattr(tt, "resolve_stt_config", lambda: _stt_config(model_path))
        monkeypatch.setattr(tt, "resolve_whispercpp_binary", lambda config=None: "/usr/bin/whisper-cli")
        monkeypatch.setattr(tt, "resolve_ffmpeg_binary", lambda config=None: "/usr/bin/ffmpeg")

        def fake_run(command, capture_output=True, text=True):
            if command[0] == "/usr/bin/ffmpeg":
                return subprocess.CompletedProcess(command, 1, "", "bad input")
            raise AssertionError("whisper.cpp should not run after ffmpeg conversion failure")

        monkeypatch.setattr(tt.subprocess, "run", fake_run)

        result = tt.transcribe_audio(str(audio_path))

        assert result["success"] is False
        assert "ffmpeg conversion failed" in result["error"]

    def test_transcribe_audio_missing_transcript_output(self, tmp_path, monkeypatch):
        audio_path = _write_audio_file(tmp_path)
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"model")

        monkeypatch.setattr(tt, "resolve_stt_config", lambda: _stt_config(model_path))
        monkeypatch.setattr(tt, "resolve_whispercpp_binary", lambda config=None: "/usr/bin/whisper-cli")
        monkeypatch.setattr(tt, "resolve_ffmpeg_binary", lambda config=None: "/usr/bin/ffmpeg")

        def fake_run(command, capture_output=True, text=True):
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(tt.subprocess, "run", fake_run)

        result = tt.transcribe_audio(str(audio_path))

        assert result["success"] is False
        assert "did not create transcript" in result["error"]
