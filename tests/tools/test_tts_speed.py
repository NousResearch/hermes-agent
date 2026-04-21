"""Tests for TTS speed configuration across providers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "MINIMAX_API_KEY",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_KEY",
        "ELEVENLABS_API_KEY",
    ):
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
# ElevenLabs credential resolution
# ---------------------------------------------------------------------------

class TestElevenLabsCredentials:
    def test_generate_elevenlabs_uses_resolved_api_key(self, tmp_path):
        chunks = [b"abc", b"def"]
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = chunks
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_elevenlabs", return_value=mock_cls), \
             patch("tools.tts_tool.resolve_elevenlabs_api_key", return_value="resolved-key"):
            from tools.tts_tool import _generate_elevenlabs

            out = tmp_path / "out.ogg"
            result = _generate_elevenlabs("Hello", str(out), {"elevenlabs": {}})

        assert result == str(out)
        assert out.read_bytes() == b"abcdef"
        assert mock_cls.call_args.kwargs["api_key"] == "resolved-key"

    def test_check_requirements_accepts_keychain_fallback(self):
        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", return_value=MagicMock()), \
             patch("tools.tts_tool.resolve_elevenlabs_api_key", return_value="resolved-key"):
            from tools.tts_tool import check_tts_requirements

            assert check_tts_requirements() is True


class TestSessionPlatformDetection:
    def test_prefers_explicit_session_platform(self, monkeypatch):
        from tools.tts_tool import _get_session_platform

        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:discord:dm:123")

        assert _get_session_platform() == "telegram"

    def test_falls_back_to_session_key_platform(self, monkeypatch):
        from tools.tts_tool import _get_session_platform

        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:12345")

        assert _get_session_platform() == "telegram"

    def test_telegram_session_key_defaults_elevenlabs_to_ogg(self, tmp_path, monkeypatch):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:12345")

        chunks = [b"abc", b"def"]
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = chunks
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "elevenlabs"}), \
             patch("tools.tts_tool._import_elevenlabs", return_value=mock_cls), \
             patch("tools.tts_tool.resolve_elevenlabs_api_key", return_value="resolved-key"):
            result = json.loads(text_to_speech_tool("Hello from Telegram", output_path=None))

        assert result["success"] is True
        assert result["provider"] == "elevenlabs"
        assert result["file_path"].endswith(".ogg")
        assert result["voice_compatible"] is True
        assert "[[audio_as_voice]]" in result["media_tag"]

    def test_telegram_custom_mp3_path_is_coerced_to_ogg_for_voice_delivery(self, tmp_path, monkeypatch):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:12345")

        chunks = [b"abc", b"def"]
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = chunks
        mock_cls = MagicMock(return_value=mock_client)

        requested = tmp_path / "voice-note-test.mp3"

        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "elevenlabs"}), \
             patch("tools.tts_tool._import_elevenlabs", return_value=mock_cls), \
             patch("tools.tts_tool.resolve_elevenlabs_api_key", return_value="resolved-key"):
            result = json.loads(text_to_speech_tool("Hello from Telegram", output_path=str(requested)))

        assert result["success"] is True
        assert result["provider"] == "elevenlabs"
        assert result["file_path"].endswith("voice-note-test.ogg")
        assert result["voice_compatible"] is True
        assert "[[audio_as_voice]]" in result["media_tag"]
        assert not requested.exists()


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
# MiniMax TTS speed (global fallback wired)
# ---------------------------------------------------------------------------

class TestMinimaxTtsSpeed:
    def _run(self, tts_config, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"audio": "deadbeef"},
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "extra_info": {"audio_size": 8},
        }

        # requests is imported locally inside _generate_minimax_tts
        with patch("requests.post", return_value=mock_response) as mock_post:
            from tools.tts_tool import _generate_minimax_tts
            _generate_minimax_tts("Hello", str(tmp_path / "out.mp3"), tts_config)
        return mock_post

    def test_global_speed_fallback(self, tmp_path, monkeypatch):
        """Global tts.speed used when minimax.speed not set."""
        mock_post = self._run({"speed": 1.5}, tmp_path, monkeypatch)
        payload = mock_post.call_args[1]["json"]
        assert payload["voice_setting"]["speed"] == 1.5

    def test_provider_speed_overrides_global(self, tmp_path, monkeypatch):
        """tts.minimax.speed takes precedence over tts.speed."""
        mock_post = self._run(
            {"speed": 1.5, "minimax": {"speed": 2.0}}, tmp_path, monkeypatch
        )
        payload = mock_post.call_args[1]["json"]
        assert payload["voice_setting"]["speed"] == 2.0
