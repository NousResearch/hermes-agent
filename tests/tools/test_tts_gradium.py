"""Tests for the Gradium TTS provider in tools/tts_tool.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("GRADIUM_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


def _minimal_wav_bytes() -> bytes:
    """Return a tiny but well-formed WAV (44-byte header + 1 sample of silence)."""
    import struct
    pcm = b"\x00\x00"
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 8000, 16000, 2, 16)
    data_chunk = struct.pack("<4sI", b"data", len(pcm)) + pcm
    riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_chunk), b"WAVE")
    return riff + fmt_chunk + data_chunk


@pytest.fixture
def mock_gradium_module():
    """Stub the gradium SDK so tests don't need the real package."""
    fake_client = MagicMock()
    fake_client.tts = AsyncMock(return_value=MagicMock(raw_data=_minimal_wav_bytes()))
    fake_client_cls = MagicMock(return_value=fake_client)
    fake_client_module = MagicMock(GradiumClient=fake_client_cls)
    fake_module = MagicMock(client=fake_client_module)
    with patch.dict(
        "sys.modules",
        {"gradium": fake_module, "gradium.client": fake_client_module},
    ):
        yield fake_client


class TestGenerateGradiumTts:
    def test_missing_api_key_raises_value_error(self, tmp_path, mock_gradium_module):
        from tools.tts_tool import _generate_gradium_tts

        with pytest.raises(ValueError, match="GRADIUM_API_KEY"):
            _generate_gradium_tts("Hello", str(tmp_path / "out.wav"), {})

    def test_writes_wav_directly(self, tmp_path, mock_gradium_module, monkeypatch):
        from tools.tts_tool import _generate_gradium_tts

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        out = tmp_path / "out.wav"

        result = _generate_gradium_tts("Hello", str(out), {})

        assert result == str(out)
        assert out.read_bytes() == _minimal_wav_bytes()
        mock_gradium_module.tts.assert_awaited_once()

    def test_default_voice_and_model_used_when_absent(
        self, tmp_path, mock_gradium_module, monkeypatch
    ):
        from tools.tts_tool import (
            DEFAULT_GRADIUM_TTS_MODEL,
            DEFAULT_GRADIUM_TTS_VOICE_ID,
            _generate_gradium_tts,
        )

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        _generate_gradium_tts("Hi", str(tmp_path / "out.wav"), {})

        call_kwargs = mock_gradium_module.tts.await_args.kwargs
        assert call_kwargs["setup"]["voice_id"] == DEFAULT_GRADIUM_TTS_VOICE_ID
        assert call_kwargs["setup"]["model_name"] == DEFAULT_GRADIUM_TTS_MODEL
        assert call_kwargs["setup"]["output_format"] == "wav"
        assert call_kwargs["text"] == "Hi"

    @pytest.mark.parametrize(
        "extension, expected_format",
        [(".wav", "wav"), (".ogg", "opus"), (".mp3", "wav")],
    )
    def test_output_format_picked_from_extension(
        self, tmp_path, mock_gradium_module, monkeypatch, extension, expected_format
    ):
        """`.ogg` → opus (native), `.wav` → wav, anything else → wav + transcode."""
        from tools.tts_tool import _generate_gradium_tts

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        # For .mp3 we go through ffmpeg; mock ffmpeg out so the test doesn't
        # depend on it being installed.
        with patch("tools.tts_tool.shutil.which", return_value=None):
            _generate_gradium_tts("Hi", str(tmp_path / f"out{extension}"), {})

        call_kwargs = mock_gradium_module.tts.await_args.kwargs
        assert call_kwargs["setup"]["output_format"] == expected_format

    def test_voice_and_model_from_config_override_defaults(
        self, tmp_path, mock_gradium_module, monkeypatch
    ):
        from tools.tts_tool import _generate_gradium_tts

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        config = {"gradium": {"voice_id": "custom-voice", "model": "premium"}}
        _generate_gradium_tts("Hi", str(tmp_path / "out.wav"), config)

        call_kwargs = mock_gradium_module.tts.await_args.kwargs
        assert call_kwargs["setup"]["voice_id"] == "custom-voice"
        assert call_kwargs["setup"]["model_name"] == "premium"

    def test_empty_audio_raises(self, tmp_path, mock_gradium_module, monkeypatch):
        from tools.tts_tool import _generate_gradium_tts

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        mock_gradium_module.tts.return_value = MagicMock(raw_data=b"")

        with pytest.raises(RuntimeError, match="empty audio"):
            _generate_gradium_tts("Hi", str(tmp_path / "out.wav"), {})

    def test_api_error_sanitized(self, tmp_path, mock_gradium_module, monkeypatch):
        from tools.tts_tool import _generate_gradium_tts

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        mock_gradium_module.tts.side_effect = RuntimeError("secret-key-in-error")

        with pytest.raises(RuntimeError, match="RuntimeError") as exc_info:
            _generate_gradium_tts("Hi", str(tmp_path / "out.wav"), {})
        assert "secret-key-in-error" not in str(exc_info.value)


class TestTtsDispatcherGradium:
    def test_dispatcher_routes_to_gradium(
        self, tmp_path, mock_gradium_module, monkeypatch
    ):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        out = str(tmp_path / "out.wav")
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "gradium"}):
            result = json.loads(text_to_speech_tool("Hello", output_path=out))

        assert result["success"] is True
        assert result["provider"] == "gradium"
        mock_gradium_module.tts.assert_awaited_once()

    def test_dispatcher_returns_error_when_sdk_not_installed(self, tmp_path, monkeypatch):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        with patch(
            "tools.tts_tool._import_gradium", side_effect=ImportError("no module")
        ), patch("tools.tts_tool._load_tts_config", return_value={"provider": "gradium"}):
            result = json.loads(
                text_to_speech_tool("Hello", output_path=str(tmp_path / "out.wav"))
            )

        assert result["success"] is False
        assert "gradium" in result["error"]


class TestImportGradiumLazyInstall:
    def test_calls_ensure_before_import(self, mock_gradium_module):
        from tools.tts_tool import _import_gradium

        with patch("tools.lazy_deps.ensure") as mock_ensure:
            _import_gradium()

        mock_ensure.assert_called_once_with("tts.gradium", prompt=False)

    def test_feature_unavailable_becomes_import_error(self, mock_gradium_module):
        from tools.lazy_deps import FeatureUnavailable
        from tools.tts_tool import _import_gradium

        unavailable = FeatureUnavailable(
            "tts.gradium", ("gradium==0.6.0",), "pip install failed"
        )
        with patch("tools.lazy_deps.ensure", side_effect=unavailable):
            with pytest.raises(ImportError, match="pip install failed"):
                _import_gradium()


class TestCheckTtsRequirementsGradium:
    def test_gradium_sdk_and_key_returns_true(self, mock_gradium_module, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "gradium"}):
            assert check_tts_requirements() is True

    def test_gradium_key_missing_returns_false(self, mock_gradium_module):
        from tools.tts_tool import check_tts_requirements

        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "gradium"}):
            assert check_tts_requirements() is False

    def test_gradium_sdk_missing_returns_false(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
        with patch("tools.tts_tool._import_gradium", side_effect=ImportError), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "gradium"}):
            assert check_tts_requirements() is False
