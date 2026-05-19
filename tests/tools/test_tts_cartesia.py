"""Tests for the Cartesia (Sonic) TTS provider in tools/tts_tool.py."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("CARTESIA_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_cartesia_module():
    """Stub the ``cartesia`` SDK so no network call is ever made.

    ``client.tts.generate`` returns a BinaryAPIResponse-like object whose
    ``write_to_file`` writes deterministic bytes to the requested path.
    """
    mock_client = MagicMock()

    def _write_to_file(path):
        with open(path, "wb") as f:
            f.write(b"fake-cartesia-audio")

    response = MagicMock()
    response.write_to_file.side_effect = _write_to_file
    mock_client.tts.generate.return_value = response

    mock_cartesia_cls = MagicMock(return_value=mock_client)
    fake_module = MagicMock()
    fake_module.Cartesia = mock_cartesia_cls
    # Patch lazy_deps.ensure too so _import_cartesia's lazy-install is a no-op.
    with patch.dict("sys.modules", {"cartesia": fake_module}), \
         patch("tools.lazy_deps.ensure", return_value=None):
        yield mock_client


class TestGenerateCartesia:
    def test_missing_api_key_raises_value_error(self, tmp_path, mock_cartesia_module):
        from tools.tts_tool import _generate_cartesia

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="CARTESIA_API_KEY"):
            _generate_cartesia("Hello", output_path, {})

    def test_successful_generation(self, tmp_path, mock_cartesia_module, monkeypatch):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        output_path = str(tmp_path / "test.mp3")
        result = _generate_cartesia("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.mp3").read_bytes() == b"fake-cartesia-audio"
        mock_cartesia_module.tts.generate.assert_called_once()
        kwargs = mock_cartesia_module.tts.generate.call_args[1]
        assert kwargs["transcript"] == "Hello world"
        assert kwargs["voice"]["mode"] == "id"

    def test_api_key_passed_to_client(self, tmp_path, mock_cartesia_module, monkeypatch):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "  spaced-key  ")
        _generate_cartesia("Hi", str(tmp_path / "test.mp3"), {})

        # Key is stripped before being handed to the SDK constructor.
        import sys

        sys.modules["cartesia"].Cartesia.assert_called_once_with(api_key="spaced-key")

    def test_default_model_and_voice(self, tmp_path, mock_cartesia_module, monkeypatch):
        from tools.tts_tool import (
            DEFAULT_CARTESIA_MODEL,
            DEFAULT_CARTESIA_VOICE_ID,
            _generate_cartesia,
        )

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        _generate_cartesia("Hi", str(tmp_path / "test.mp3"), {})

        kwargs = mock_cartesia_module.tts.generate.call_args[1]
        assert kwargs["model_id"] == DEFAULT_CARTESIA_MODEL
        assert kwargs["voice"]["id"] == DEFAULT_CARTESIA_VOICE_ID

    def test_config_overrides_model_and_voice(
        self, tmp_path, mock_cartesia_module, monkeypatch
    ):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        config = {"cartesia": {"model": "sonic-turbo", "voice_id": "my-voice-uuid"}}
        _generate_cartesia("Hi", str(tmp_path / "test.mp3"), config)

        kwargs = mock_cartesia_module.tts.generate.call_args[1]
        assert kwargs["model_id"] == "sonic-turbo"
        assert kwargs["voice"]["id"] == "my-voice-uuid"

    def test_mp3_output_format(self, tmp_path, mock_cartesia_module, monkeypatch):
        from tools.tts_tool import (
            DEFAULT_CARTESIA_BIT_RATE,
            DEFAULT_CARTESIA_SAMPLE_RATE,
            _generate_cartesia,
        )

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        _generate_cartesia("Hi", str(tmp_path / "test.mp3"), {})

        fmt = mock_cartesia_module.tts.generate.call_args[1]["output_format"]
        assert fmt["container"] == "mp3"
        assert fmt["sample_rate"] == DEFAULT_CARTESIA_SAMPLE_RATE
        assert fmt["bit_rate"] == DEFAULT_CARTESIA_BIT_RATE

    def test_wav_output_format(self, tmp_path, mock_cartesia_module, monkeypatch):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        config = {"cartesia": {"sample_rate": 24000}}
        _generate_cartesia("Hi", str(tmp_path / "test.wav"), config)

        fmt = mock_cartesia_module.tts.generate.call_args[1]["output_format"]
        assert fmt["container"] == "wav"
        assert fmt["encoding"] == "pcm_s16le"
        assert fmt["sample_rate"] == 24000
        assert "bit_rate" not in fmt

    def test_language_passed_only_when_configured(
        self, tmp_path, mock_cartesia_module, monkeypatch
    ):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")

        _generate_cartesia("Hi", str(tmp_path / "a.mp3"), {})
        assert "language" not in mock_cartesia_module.tts.generate.call_args[1]

        _generate_cartesia(
            "Hi", str(tmp_path / "b.mp3"), {"cartesia": {"language": "fr"}}
        )
        assert mock_cartesia_module.tts.generate.call_args[1]["language"] == "fr"

    def test_speed_routed_to_generation_config(
        self, tmp_path, mock_cartesia_module, monkeypatch
    ):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")

        # Default speed (1.0) must NOT add generation_config.
        _generate_cartesia("Hi", str(tmp_path / "a.mp3"), {})
        assert "generation_config" not in mock_cartesia_module.tts.generate.call_args[1]

        # Non-default speed goes through generation_config (v3 SDK norm).
        _generate_cartesia("Hi", str(tmp_path / "b.mp3"), {"cartesia": {"speed": 1.3}})
        kwargs = mock_cartesia_module.tts.generate.call_args[1]
        assert kwargs["generation_config"] == {"speed": 1.3}

    def test_response_read_shape(self, tmp_path, monkeypatch):
        """SDK variants returning a .read() body are handled."""
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        mock_client = MagicMock()
        response = MagicMock(spec=["read"])
        response.read.return_value = b"read-body-bytes"
        mock_client.tts.generate.return_value = response
        fake_module = MagicMock()
        fake_module.Cartesia = MagicMock(return_value=mock_client)

        out = str(tmp_path / "test.mp3")
        with patch.dict("sys.modules", {"cartesia": fake_module}), \
             patch("tools.lazy_deps.ensure", return_value=None):
            _generate_cartesia("Hi", out, {})

        assert (tmp_path / "test.mp3").read_bytes() == b"read-body-bytes"

    def test_response_bytes_shape(self, tmp_path, monkeypatch):
        """SDK variants returning raw bytes are handled."""
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.tts.generate.return_value = b"raw-bytes-body"
        fake_module = MagicMock()
        fake_module.Cartesia = MagicMock(return_value=mock_client)

        out = str(tmp_path / "test.mp3")
        with patch.dict("sys.modules", {"cartesia": fake_module}), \
             patch("tools.lazy_deps.ensure", return_value=None):
            _generate_cartesia("Hi", out, {})

        assert (tmp_path / "test.mp3").read_bytes() == b"raw-bytes-body"

    def test_response_chunk_iterator_shape(self, tmp_path, monkeypatch):
        """SDK variants yielding a chunk iterator (tts.bytes()) are handled."""
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.tts.generate.return_value = iter([b"chunk-1", b"chunk-2"])
        fake_module = MagicMock()
        fake_module.Cartesia = MagicMock(return_value=mock_client)

        out = str(tmp_path / "test.mp3")
        with patch.dict("sys.modules", {"cartesia": fake_module}), \
             patch("tools.lazy_deps.ensure", return_value=None):
            _generate_cartesia("Hi", out, {})

        assert (tmp_path / "test.mp3").read_bytes() == b"chunk-1chunk-2"

    def test_client_closed_after_generation(
        self, tmp_path, mock_cartesia_module, monkeypatch
    ):
        from tools.tts_tool import _generate_cartesia

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        _generate_cartesia("Hi", str(tmp_path / "test.mp3"), {})

        mock_cartesia_module.close.assert_called_once()


class TestTtsDispatcherCartesia:
    def test_dispatcher_success(self, tmp_path, mock_cartesia_module, monkeypatch):
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        output_path = str(tmp_path / "out.mp3")
        with patch(
            "tools.tts_tool._load_tts_config", return_value={"provider": "cartesia"}
        ):
            result = json.loads(
                text_to_speech_tool("Hello", output_path=output_path)
            )

        assert result["success"] is True
        assert result["provider"] == "cartesia"
        mock_cartesia_module.tts.generate.assert_called_once()

    def test_dispatcher_returns_error_when_sdk_not_installed(
        self, tmp_path, monkeypatch
    ):
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        with patch(
            "tools.tts_tool._import_cartesia", side_effect=ImportError("no module")
        ), patch(
            "tools.tts_tool._load_tts_config", return_value={"provider": "cartesia"}
        ):
            result = json.loads(
                text_to_speech_tool("Hello", output_path=str(tmp_path / "out.mp3"))
            )

        assert result["success"] is False
        assert "cartesia" in result["error"].lower()

    def test_dispatcher_missing_key_returns_error(
        self, tmp_path, mock_cartesia_module
    ):
        from tools.tts_tool import text_to_speech_tool

        with patch(
            "tools.tts_tool._load_tts_config", return_value={"provider": "cartesia"}
        ):
            result = json.loads(
                text_to_speech_tool("Hello", output_path=str(tmp_path / "out.mp3"))
            )

        assert result["success"] is False
        assert "CARTESIA_API_KEY" in result["error"]


class TestCartesiaIsBuiltinProvider:
    def test_cartesia_registered(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "cartesia" in BUILTIN_TTS_PROVIDERS

    def test_cartesia_lazy_dep_registered(self):
        from tools.lazy_deps import LAZY_DEPS

        assert "tts.cartesia" in LAZY_DEPS
        assert any("cartesia" in pkg for pkg in LAZY_DEPS["tts.cartesia"])
