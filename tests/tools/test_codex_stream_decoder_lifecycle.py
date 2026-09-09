"""The Codex live decoder must release its real child and audio file on interruption."""

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import tts_streaming, tts_tool_codex


@pytest.mark.parametrize("outcome", ["cancel", "stderr"])
@pytest.mark.live_system_guard_bypass  # Real cancellation signals only target children spawned here.
def test_codex_decoder_cannot_stall_on_cancellation_or_diagnostics(monkeypatch, outcome):
    monkeypatch.setattr(tts_tool_codex, "_codex_tts_credentials", lambda: (None, {}))
    monkeypatch.setattr(
        tts_tool_codex, "synthesize_codex_speech_with_credentials",
        lambda *args, **kwargs: SimpleNamespace(audio=b"ID3-decoder-input"),
    )
    monkeypatch.setattr(tts_streaming.shutil, "which", lambda name: sys.executable)
    spawned = threading.Event()
    finished = threading.Event()
    processes = []
    inputs = []
    errors = []
    chunks = []
    popen = subprocess.Popen

    def decoder(args, **kwargs):
        inputs.append(Path(args[args.index("-i") + 1]))
        code = (
            "import sys,time; time.sleep(30)"
            if outcome == "cancel" else
            "import sys; sys.stderr.buffer.write(b'x' * 1048576); "
            "sys.stderr.buffer.flush(); sys.stdout.buffer.write(b'\\x01\\x00')"
        )
        process = popen([sys.executable, "-c", code], **kwargs)
        processes.append(process)
        spawned.set()
        return process

    monkeypatch.setattr(tts_streaming.subprocess, "Popen", decoder)
    streamer = tts_streaming.OpenAICodexStreamer({}, {})

    def consume():
        try:
            chunks.extend(streamer.stream("The decoder must release this sentence."))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert spawned.wait(2), "decoder never started"
        if outcome == "cancel":
            streamer.cancel()
        assert finished.wait(2), "decoder remained blocked after cancellation or stderr output"
        assert not errors
        assert b"".join(chunks) == (b"" if outcome == "cancel" else b"\x01\x00")
        assert processes[0].poll() is not None
        assert processes[0].stdout.closed
        assert not inputs[0].exists()
    finally:
        streamer.cancel()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        worker.join(timeout=5)
