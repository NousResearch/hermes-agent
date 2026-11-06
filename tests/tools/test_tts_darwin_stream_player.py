"""Tests for the Darwin stdin-player path in ``tools/tts_tool_speaker.py``.

On macOS ``_StreamerPlayback._use_device()`` is False (PortAudio init raises a TCC prompt),
so a chunked provider used to be drained into a temp WAV and only then played — which wastes
the whole point of a streaming provider. ``_play_via_streaming_player`` pipes the PCM into a
SoX ``play`` process instead, so sound starts on the first chunk.

These tests pin the pieces that are easy to break silently: the command shape (raw PCM, the
configured rate, a gain), the opt-out switch, the fallback when no player exists, back-pressure
ordering of writes, and that a dead player never strands the prefetch thread.
"""

from __future__ import annotations

import queue
import threading

import pytest

import tools.tts_tool_speaker as sp


class _FakeStdin:
    def __init__(self, written: list[bytes], fail_after: int | None = None):
        self.written = written
        self.closed = False
        self.fail_after = fail_after
        self.calls = 0

    def write(self, data: bytes) -> int:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise BrokenPipeError("player exited")
        self.calls += 1
        self.written.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(self, fail_after: int | None = None):
        self.written: list[bytes] = []
        self.stdin = _FakeStdin(self.written, fail_after)
        self.terminated = False
        self.killed = False
        self.waited = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _queue(*chunks: bytes) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    for chunk in chunks:
        q.put(chunk)
    q.put(None)
    return q


@pytest.fixture
def play_available(monkeypatch):
    monkeypatch.setattr(sp.shutil, "which", lambda name: "/opt/homebrew/bin/play")
    monkeypatch.delenv("HERMES_TTS_STREAM_PLAYER", raising=False)
    monkeypatch.delenv("HERMES_TTS_GAIN", raising=False)
    return monkeypatch


def _install_fake_popen(monkeypatch, fail_after: int | None = None):
    proc = _FakeProc(fail_after)
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):
        calls.append(list(command))
        return proc

    monkeypatch.setattr(sp.subprocess, "Popen", fake_popen)
    return proc, calls


def test_command_streams_raw_pcm_with_a_gain(play_available):
    command = sp._streaming_player_command(44100, 1)
    assert command is not None
    assert command[0] == "/opt/homebrew/bin/play"
    assert command[1:3] == ["-q", "-t"]
    assert "raw" in command and "s16le" not in command  # SoX spells it -t raw -e signed
    assert "-r" in command and command[command.index("-r") + 1] == "44100"
    assert command[command.index("-e") + 1] == "signed"
    assert command[command.index("-b") + 1] == "16"
    assert command[command.index("-c") + 1] == "1"
    assert command[-2:] == ["vol", "0.1"]  # default gain: quiet, never a surprise blast
    assert command[command.index("-") ] == "-"  # stdin


def test_tempo_is_opt_in(play_available):
    command = sp._streaming_player_command(44100)
    assert command is not None and "tempo" not in command  # default: play as generated
    play_available.setenv("HERMES_TTS_TEMPO", "1.1")
    command = sp._streaming_player_command(44100)
    assert command is not None and command[-2:] == ["tempo", "1.1"]


def test_gain_is_configurable(play_available):
    play_available.setenv("HERMES_TTS_GAIN", "0.25")
    command = sp._streaming_player_command(44100)
    assert command is not None and command[-2:] == ["vol", "0.25"]


def test_opt_out_and_missing_player_return_none(play_available):
    play_available.setenv("HERMES_TTS_STREAM_PLAYER", "0")
    assert sp._streaming_player_command(44100) is None
    play_available.delenv("HERMES_TTS_STREAM_PLAYER")
    play_available.setattr(sp.shutil, "which", lambda name: None)
    assert sp._streaming_player_command(44100) is None


def test_player_receives_every_chunk_in_order(play_available):
    proc, calls = _install_fake_popen(play_available)
    stop = threading.Event()
    assert sp._play_via_streaming_player(_queue(b"\x01\x00", b"\x02\x00"), stop, 44100) is True
    assert proc.written == [b"\x01\x00", b"\x02\x00"]
    assert proc.stdin.closed and proc.waited
    assert calls and calls[0][-2:] == ["vol", "0.1"]


def test_missing_player_leaves_the_queue_untouched(play_available):
    play_available.setattr(sp.shutil, "which", lambda name: None)
    q = _queue(b"\x01\x00")
    assert sp._play_via_streaming_player(q, threading.Event(), 44100) is False
    # Fallback must still see the chunks: nothing was consumed.
    assert q.get_nowait() == b"\x01\x00"


def test_dead_player_is_drained_so_prefetch_cannot_block(play_available):
    proc, _ = _install_fake_popen(play_available, fail_after=1)
    chunks = [b"\x01\x00"] * 5
    stop = threading.Event()
    assert sp._play_via_streaming_player(_queue(*chunks), stop, 44100) is True
    assert len(proc.written) == 1  # the pipe died after the first write
    assert proc.stdin.closed and proc.waited


def test_stop_event_terminates_the_player(play_available):
    proc, _ = _install_fake_popen(play_available)
    stop = threading.Event()
    stop.set()
    assert sp._play_via_streaming_player(_queue(b"\x01\x00"), stop, 44100) is True
    assert proc.terminated


def test_sentence_play_falls_back_to_tempfile_without_a_player(play_available):
    play_available.setattr(sp.shutil, "which", lambda name: None)

    class _Streamer:
        sample_rate = 44100
        channels = 1

    playback = sp._StreamerPlayback.__new__(sp._StreamerPlayback)  # no threads
    playback.streamer = _Streamer()
    playback.stop_event = threading.Event()
    drained: list[queue.Queue] = []
    playback._play_sentence_via_tempfile = drained.append  # type: ignore[method-assign]

    q: queue.Queue = queue.Queue()
    playback._play_sentence_via_player(q)
    assert drained == [q]


def test_worker_prefers_the_player_when_no_device_is_usable(play_available):
    playback = sp._StreamerPlayback.__new__(sp._StreamerPlayback)
    playback._use_device = False
    captured: list = []
    playback._for_each_sentence = captured.append  # type: ignore[method-assign]
    playback._playback_worker()
    assert [getattr(fn, "__name__", fn) for fn in captured] == ["_play_sentence_via_player"]
