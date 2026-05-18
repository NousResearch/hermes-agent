"""Tests for TTS speed configuration across providers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "MINIMAX_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Edge TTS speed
# ---------------------------------------------------------------------------

class TestEdgeTtsSpeed:
    def _run(self, tts_config, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_edge = MagicMock()
        mock_edge.Communicate = MagicMock(return_value=mock_comm)

        with patch("tools.tts_tool._import_edge_tts", return_value=mock_edge):
            from tools.tts_tool import _generate_edge_tts
            asyncio.run(_generate_edge_tts("Hello", str(tmp_path / "out.mp3"), tts_config))
        return mock_edge.Communicate

    def test_default_no_rate_kwarg(self, tmp_path):
        """No speed config => no rate kwarg passed to Communicate."""
        comm_cls = self._run({}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert "rate" not in kwargs

    def test_global_speed_applied(self, tmp_path):
        """Global tts.speed used as fallback."""
        comm_cls = self._run({"speed": 1.5}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["rate"] == "+50%"

    def test_provider_speed_overrides_global(self, tmp_path):
        """tts.edge.speed takes precedence over tts.speed."""
        comm_cls = self._run({"speed": 1.5, "edge": {"speed": 2.0}}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["rate"] == "+100%"

    def test_speed_below_one(self, tmp_path):
        """Speed < 1.0 produces a negative rate string."""
        comm_cls = self._run({"speed": 0.5}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["rate"] == "-50%"

    def test_speed_exactly_one_no_rate(self, tmp_path):
        """Explicit speed=1.0 should not pass rate kwarg."""
        comm_cls = self._run({"speed": 1.0}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert "rate" not in kwargs


# ---------------------------------------------------------------------------
# OpenAI TTS speed
# ---------------------------------------------------------------------------

class TestOpenaiTtsSpeed:
    def _run(self, tts_config, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)):
            from tools.tts_tool import _generate_openai_tts
            _generate_openai_tts("Hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_client.audio.speech.create

    def test_default_no_speed_kwarg(self, tmp_path, monkeypatch):
        """No speed config => no speed kwarg in create call."""
        create = self._run({}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert "speed" not in kwargs

    def test_global_speed_applied(self, tmp_path, monkeypatch):
        """Global tts.speed used as fallback."""
        create = self._run({"speed": 1.5}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 1.5

    def test_provider_speed_overrides_global(self, tmp_path, monkeypatch):
        """tts.openai.speed takes precedence over tts.speed."""
        create = self._run({"speed": 1.5, "openai": {"speed": 2.0}}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 2.0

    def test_speed_clamped_low(self, tmp_path, monkeypatch):
        """Speed below 0.25 is clamped to 0.25."""
        create = self._run({"speed": 0.1}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 0.25

    def test_speed_clamped_high(self, tmp_path, monkeypatch):
        """Speed above 4.0 is clamped to 4.0."""
        create = self._run({"speed": 10.0}, tmp_path, monkeypatch)
        kwargs = create.call_args[1]
        assert kwargs["speed"] == 4.0


# ---------------------------------------------------------------------------
# MiniMax TTS (new API: raw audio, no speed/voice_setting)
# ---------------------------------------------------------------------------

class TestMinimaxTtsSpeed:
    def _run(self, tts_config, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "audio/mpeg"}
        mock_response.content = b"\x00\x01\x02\x03"

        # requests is imported locally inside _generate_minimax_tts
        with patch("requests.post", return_value=mock_response) as mock_post:
            from tools.tts_tool import _generate_minimax_tts
            output = _generate_minimax_tts("Hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_post, output

    def test_simple_payload(self, tmp_path, monkeypatch):
        """New API uses flat payload with model, text, voice_id."""
        mock_post, _ = self._run({}, tmp_path, monkeypatch)
        payload = mock_post.call_args[1]["json"]
        assert "model" in payload
        assert "text" in payload
        assert "voice_id" in payload
        assert "voice_setting" not in payload
        assert "audio_setting" not in payload
        assert "stream" not in payload

    def test_writes_raw_audio(self, tmp_path, monkeypatch):
        """New API returns raw bytes written directly to file."""
        _, output = self._run({}, tmp_path, monkeypatch)
        assert output == str(tmp_path / "out.mp3")
        with open(output, "rb") as f:
            assert f.read() == b"\x00\x01\x02\x03"
