"""OpenAI TTS opus handling for Telegram voice bubbles.

Real OpenAI encodes opus natively, but many OpenAI-*compatible* backends
(e.g. a self-hosted Speaches/Kokoro endpoint) only support mp3/flac/wav/pcm
and reject ``response_format="opus"``. ``_generate_openai_tts`` must try
native opus first and fall back to mp3 + local ffmpeg transcode so those
backends still deliver a playable OGG/Opus voice bubble.
"""
from unittest.mock import MagicMock, patch


def _client():
    mock_response = MagicMock()
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response
    return mock_client, MagicMock(return_value=mock_client)


def _formats(create):
    """response_format passed to each create() call, in order."""
    return [c.kwargs["response_format"] for c in create.call_args_list]


class TestOpenaiOpusFallback:
    def test_native_opus_no_transcode(self, tmp_path, monkeypatch):
        """An opus-capable backend gets a single opus request, no ffmpeg."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client, mock_cls = _client()
        out = str(tmp_path / "out.ogg")

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)), \
             patch("tools.tts_tool._convert_to_opus") as convert:
            from tools.tts_tool import _generate_openai_tts
            result = _generate_openai_tts("Hello", out, {})

        assert _formats(mock_client.audio.speech.create) == ["opus"]
        convert.assert_not_called()
        assert result == out

    def test_falls_back_to_mp3_and_transcodes(self, tmp_path, monkeypatch):
        """A backend that rejects opus is retried as mp3 then transcoded."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client, mock_cls = _client()
        # First call (opus) fails; second call (mp3) succeeds.
        mock_client.audio.speech.create.side_effect = [
            Exception("Unsupported response_format 'opus'"),
            MagicMock(),
        ]
        out = str(tmp_path / "out.ogg")
        converted = str(tmp_path / "out.ogg")

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)), \
             patch("tools.tts_tool._convert_to_opus", return_value=converted) as convert:
            from tools.tts_tool import _generate_openai_tts
            result = _generate_openai_tts("Hello", out, {})

        assert _formats(mock_client.audio.speech.create) == ["opus", "mp3"]
        convert.assert_called_once_with(str(tmp_path / "out.mp3"))
        assert result == converted

    def test_fallback_without_ffmpeg_keeps_mp3(self, tmp_path, monkeypatch):
        """No ffmpeg => return the mp3 so the caller still has audio."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client, mock_cls = _client()
        mock_client.audio.speech.create.side_effect = [
            Exception("Unsupported response_format 'opus'"),
            MagicMock(),
        ]
        out = str(tmp_path / "out.ogg")

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)), \
             patch("tools.tts_tool._convert_to_opus", return_value=None):
            from tools.tts_tool import _generate_openai_tts
            result = _generate_openai_tts("Hello", out, {})

        assert result == str(tmp_path / "out.mp3")

    def test_non_ogg_target_uses_mp3_directly(self, tmp_path, monkeypatch):
        """A plain .mp3 target never asks for opus."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client, mock_cls = _client()
        out = str(tmp_path / "out.mp3")

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)), \
             patch("tools.tts_tool._convert_to_opus") as convert:
            from tools.tts_tool import _generate_openai_tts
            result = _generate_openai_tts("Hello", out, {})

        assert _formats(mock_client.audio.speech.create) == ["mp3"]
        convert.assert_not_called()
        assert result == out
