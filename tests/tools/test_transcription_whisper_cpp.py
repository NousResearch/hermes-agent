"""The managed command remains passive until transcription and caches verified models."""

import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_models_download_once_per_profile_and_reject_bad_bytes(tmp_path, monkeypatch):
    from pm.downloader import HashError
    from pm import downloader
    from tools import transcription_whisper_cpp as cpp

    served = tmp_path / "served"
    served.mkdir()
    data = b"model fixture"
    (served / "model.bin").write_bytes(data)
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            requests.append(args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(served)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/model.bin"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(cpp, "_MODEL_FILES", (("model.bin", url, digest),))
    hashes = []
    original_hash = downloader._sha256_file
    def hash_file(path):
        hashes.append(path)
        return original_hash(path)
    monkeypatch.setattr(downloader, "_sha256_file", hash_file)
    try:
        for profile in ("A", "B", "A"):
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / profile))
            before = len(requests)
            hashes_before = len(hashes)
            cached = cpp._model_dir() / "model.bin"
            was_cached = cached.exists()
            cpp.ensure_whisper_cpp_models("base")
            assert cached.read_bytes() == data
            assert (len(requests) == before) == was_cached
            assert (len(hashes) == hashes_before) == was_cached
        cached.write_bytes(b"corrupt")
        monkeypatch.setattr(cpp, "_MODEL_FILES", (("model.bin", url, "0" * 64),))
        with pytest.raises(HashError):
            cpp.ensure_whisper_cpp_models("base")
        assert cached.read_bytes() == b"corrupt"  # failed transfer was never published
        with pytest.raises(ValueError, match="currently supports"):
            cpp.ensure_whisper_cpp_models("small")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_local_resolution_is_passive_and_command_override_wins(tmp_path, monkeypatch):
    from tools import transcription_local as local
    from tools import transcription_tools as stt
    from tools import transcription_whisper_cpp as cpp
    import pm
    from pm.packages import WhisperCppCpu
    from pm.store import ALL_TARGETS

    package = WhisperCppCpu()
    assert [t for t in ALL_TARGETS if package.missing_reason(t) is None] == ["win32-arm64"]
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile with spaces"))
    binary = tmp_path / "program files" / "whisper-cli.exe"
    monkeypatch.setattr(pm, "installed_package",
                        lambda name: SimpleNamespace(binary=binary) if name == "whispercpp-cpu" else None)
    monkeypatch.setattr(stt, "_HAS_FASTER_WHISPER", False)
    calls = []
    monkeypatch.setattr(cpp, "ensure_whisper_cpp_models", lambda model: calls.append(model))
    assert stt._detect_local_backend() == "local_command"
    assert calls == []
    assert not cpp._model_dir().exists()
    audio = tmp_path / "voice note.wav"
    audio.write_bytes(b"RIFF")

    def run(command, **kwargs):
        assert command[command.index("-f") + 1] == str(audio)
        assert command[command.index("-l") + 1] == "en"
        assert "-ng" in command and "--vad" in command
        Path(command[command.index("-of") + 1] + ".txt").write_text("hello")

    monkeypatch.setattr(local, "_run_quiet", run)
    result = local._transcribe_local_command(str(audio), "base", language="en")
    assert result["success"] and result["transcript"] == "hello"
    assert calls == ["base"]
    monkeypatch.setenv("HERMES_LOCAL_STT_COMMAND", "custom-command {input_path}")
    assert local._get_local_command_template() == "custom-command {input_path}"
