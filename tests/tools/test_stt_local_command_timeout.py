"""stt.local.command_timeout bounds the local STT command run (default 300 s)."""

import struct
import subprocess
import wave

import pytest


@pytest.fixture
def sample_wav(tmp_path):
    wav_path = tmp_path / "voice.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<1600h", *([0] * 1600)))
    return str(wav_path)


def _run(monkeypatch, tmp_path, sample_wav, stt_config):
    from tools.transcription_tools import LOCAL_STT_COMMAND_ENV, _transcribe_local_command

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setenv(LOCAL_STT_COMMAND_ENV, "whisper {input_path} --output_dir {output_dir}")
    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: stt_config)

    class _TempDir:
        def __init__(self, prefix=None):
            pass

        def __enter__(self):
            return str(output_dir)

        def __exit__(self, exc_type, exc, tb):
            return False

    seen = {}

    def fake_run(command, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        (output_dir / "voice.txt").write_text("bonjour", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.transcription_local.tempfile.TemporaryDirectory", _TempDir)
    monkeypatch.setattr("tools.transcription_audio.subprocess.run", fake_run)
    result = _transcribe_local_command(sample_wav, "base")
    assert result["transcript"] == "bonjour"
    return seen["timeout"]


@pytest.mark.parametrize(("stt_config", "expected"), [
    ({}, 300),
    ({"local": {"command_timeout": 900}}, 900),
    ({"local": {"command_timeout": "600"}}, 600),
    ({"local": {"command_timeout": 0}}, 300),
    ({"local": {"command_timeout": "soon"}}, 300),
])
def test_local_command_timeout(monkeypatch, tmp_path, sample_wav, stt_config, expected):
    assert _run(monkeypatch, tmp_path, sample_wav, stt_config) == expected
