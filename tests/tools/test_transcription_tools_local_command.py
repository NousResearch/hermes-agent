import subprocess

import pytest

from tools import transcription_tools as tt


@pytest.fixture(autouse=True)
def _clear_local_command_health_cache():
    tt._local_command_health_cache.clear()
    yield
    tt._local_command_health_cache.clear()


def test_has_local_command_rejects_broken_auto_whisper(monkeypatch):
    monkeypatch.delenv(tt.LOCAL_STT_COMMAND_ENV, raising=False)
    monkeypatch.setattr(tt, "_find_whisper_binary", lambda: "/tmp/fake-whisper")

    def fake_run(cmd, **kwargs):
        assert cmd == ["/tmp/fake-whisper", "--help"]
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="ImportError: Library not loaded: libprotobuf.34.1.0.dylib\n",
        )

    monkeypatch.setattr(tt.subprocess, "run", fake_run)

    assert tt._has_local_command() is False


def test_has_local_command_allows_user_configured_command_without_probe(monkeypatch):
    monkeypatch.setenv(tt.LOCAL_STT_COMMAND_ENV, "custom-stt {input_path}")
    monkeypatch.setattr(
        tt,
        "_find_whisper_binary",
        lambda: pytest.fail("custom local STT command should not probe whisper"),
    )

    assert tt._has_local_command() is True


def test_transcribe_local_command_surfaces_startup_check_failure(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake wav")
    monkeypatch.delenv(tt.LOCAL_STT_COMMAND_ENV, raising=False)
    monkeypatch.setattr(tt, "_find_whisper_binary", lambda: "/tmp/fake-whisper")
    monkeypatch.setattr(
        tt,
        "_get_local_command_template",
        lambda: "/tmp/fake-whisper {input_path} --output_dir {output_dir}",
    )
    monkeypatch.setattr(
        tt,
        "_check_auto_whisper_command",
        lambda _binary: (
            "whisper --help exited with code 1: stderr: ImportError: "
            "Library not loaded: libprotobuf.34.1.0.dylib"
        ),
    )

    result = tt._transcribe_local_command(str(audio), "base")

    assert result["success"] is False
    assert result["provider"] == "local_command"
    assert "failed its startup check" in result["error"]
    assert "libprotobuf" in result["error"]


def test_transcribe_local_command_compacts_subprocess_stderr(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake wav")
    monkeypatch.setenv(
        tt.LOCAL_STT_COMMAND_ENV,
        "fake-stt {input_path} --output_dir {output_dir}",
    )
    monkeypatch.setattr(tt, "_load_stt_config", lambda: {})

    long_traceback = "\n".join(
        ["Traceback (most recent call last):"]
        + [f"frame {idx}" for idx in range(30)]
        + ["ImportError: Library not loaded: libprotobuf.34.1.0.dylib"]
    )

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            output="",
            stderr=long_traceback,
        )

    monkeypatch.setattr(tt.subprocess, "run", fake_run)

    result = tt._transcribe_local_command(str(audio), "base")

    assert result["success"] is False
    assert result["provider"] == "local_command"
    assert "Local STT failed" in result["error"]
    assert "libprotobuf" in result["error"]
    assert len(result["error"]) < 1000
    assert "frame 0" not in result["error"]


def test_transcribe_local_command_handles_timeout(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake wav")
    monkeypatch.setenv(
        tt.LOCAL_STT_COMMAND_ENV,
        "fake-stt {input_path} --output_dir {output_dir}",
    )
    monkeypatch.setattr(tt, "_load_stt_config", lambda: {})

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 300, output="", stderr="")

    monkeypatch.setattr(tt.subprocess, "run", fake_run)

    result = tt._transcribe_local_command(str(audio), "base")

    assert result["success"] is False
    assert result["provider"] == "local_command"
    assert "timed out after 300s" in result["error"]
