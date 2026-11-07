"""Tests for the local MLX Qwen3-TTS streaming provider."""

import struct
from unittest.mock import MagicMock, patch

import pytest

from tools.tts_streaming import Qwen3TTSMLXStreamer, _REGISTRY


class FakePopen:
    """Minimal subprocess.Popen stand-in with a scripted stdout."""

    def __init__(self, frames, rc=0):
        # Build frame bytes: [u32 len][pcm]...
        buf = b""
        for pcm in frames:
            buf += struct.pack("<I", len(pcm)) + pcm
        self._buf = buf
        self._rc = rc
        self.stdin = MagicMock()
        self.stdout = MagicMock()
        self.stderr = MagicMock()
        self.stdout.read = MagicMock(side_effect=self._read)
        self.stderr.read = MagicMock(return_value=b"")

    def _read(self, n):
        chunk, self._buf = self._buf[:n], self._buf[n:]
        return chunk

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        # Mirror subprocess.Popen.poll(): None while running, exit code once
        # the process has terminated (our fake terminates on wait()).
        return self._rc

    def kill(self):
        self._rc = -9


class TestQwen3TTSMLXStreamer:
    def test_registered(self):
        assert "qwen3tts-mlx" in _REGISTRY

    def test_available_false_when_venv_missing(self):
        with patch.object(Qwen3TTSMLXStreamer, "_VENV", "/nonexistent/venv/bin/python3"):
            assert Qwen3TTSMLXStreamer.available() is False

    def test_available_true_when_present(self):
        # The class defaults point at the real install; available() should
        # reflect filesystem existence (true on this dev machine).
        import os

        if os.path.exists(Qwen3TTSMLXStreamer._VENV):
            assert Qwen3TTSMLXStreamer.available() is True

    def test_stream_yields_frames_in_order(self):
        frames = [b"\x00\x01" * 100, b"\x02\x03" * 200]
        proc = FakePopen(frames)
        with patch("subprocess.Popen", return_value=proc), patch.object(
            Qwen3TTSMLXStreamer, "available", return_value=True
        ):
            s = Qwen3TTSMLXStreamer({"provider": "qwen3tts-mlx"}, {})
            out = list(s.stream("第一句。\n第二句。"))
        assert out == frames

    def test_stream_raises_on_worker_failure(self):
        proc = FakePopen([], rc=1)
        with patch("subprocess.Popen", return_value=proc), patch.object(
            Qwen3TTSMLXStreamer, "available", return_value=True
        ):
            s = Qwen3TTSMLXStreamer({"provider": "qwen3tts-mlx"}, {})
            with pytest.raises(RuntimeError, match="worker exited 1"):
                list(s.stream("hi"))

    def test_stream_uses_section_model_and_ref(self):
        proc = FakePopen([b"\x00\x01" * 10])
        with patch("subprocess.Popen", return_value=proc) as mock_popen, patch.object(
            Qwen3TTSMLXStreamer, "available", return_value=True
        ):
            s = Qwen3TTSMLXStreamer(
                {"provider": "qwen3tts-mlx"},
                {"model_dir": "/models/m", "ref_audio": "/ref.wav"},
            )
            list(s.stream("text"))
        cmd = mock_popen.call_args.args[0]
        assert "--model" in cmd and "/models/m" in cmd
        assert "--ref" in cmd and "/ref.wav" in cmd
