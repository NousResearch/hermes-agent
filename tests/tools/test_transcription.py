"""Tests for transcription_tools.py — local (faster-whisper) and OpenAI providers.

Tests cover provider selection, config loading, validation, and transcription
dispatch.  All external dependencies (faster_whisper, openai) are mocked.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fake_faster_whisper_module(mock_model):
    return SimpleNamespace(WhisperModel=MagicMock(return_value=mock_model))


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestGetProvider:
    """_get_provider() picks the right backend based on config + availability."""

    def test_local_when_available(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "local"

    def test_explicit_local_no_cloud_fallback(self, monkeypatch):
        """Explicit local provider must not silently fall back to cloud."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.tool_backend_helpers.read_selection", return_value="local"):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "none"


    def test_disabled_config_returns_none(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"enabled": False, "provider": "openai"}) == "none"


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


class TestValidateAudioFile:

    def test_missing_file(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        result = _validate_audio_file(str(tmp_path / "nope.ogg"))
        assert result is not None
        assert "not found" in result["error"]


    def test_too_large(self, tmp_path):
        f = tmp_path / "big.ogg"
        f.write_bytes(b"x")
        from tools.transcription_tools import _validate_audio_file
        from tools.transcription_common import MAX_FILE_SIZE
        real_stat = f.stat()
        with patch.object(type(f), "stat", return_value=os.stat_result((
            real_stat.st_mode, real_stat.st_ino, real_stat.st_dev,
            real_stat.st_nlink, real_stat.st_uid, real_stat.st_gid,
            MAX_FILE_SIZE + 1,  # st_size
            real_stat.st_atime, real_stat.st_mtime, real_stat.st_ctime,
        ))):
            result = _validate_audio_file(str(f))
        assert result is not None
        assert "too large" in result["error"]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestLoadSttConfig:

    def test_merges_default_local_initial_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "stt:\n  local:\n    model: small\n",
            encoding="utf-8",
        )

        from tools.transcription_tools import _load_stt_config
        local_config = _load_stt_config()["local"]

        assert local_config["model"] == "small"
        assert local_config["initial_prompt"] == ""


# ---------------------------------------------------------------------------
# Local transcription
# ---------------------------------------------------------------------------


class TestTranscribeLocal:

    def test_successful_transcription(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 2.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        fake_fw = _fake_faster_whisper_module(mock_model)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch.dict("sys.modules", {"faster_whisper": fake_fw}), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio_file), "base")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"


    def test_not_installed(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local("/tmp/test.ogg", "base")
        assert result["success"] is False
        assert "not installed" in result["error"]


# ---------------------------------------------------------------------------
# OpenAI transcription
# ---------------------------------------------------------------------------


class TestTranscribeOpenAI:

    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        from tools.transcription_tools import _transcribe_openai
        result = _transcribe_openai("/tmp/test.ogg", "whisper-1")
        assert result["success"] is False
        assert "VOICE_TOOLS_OPENAI_KEY" in result["error"]


    def test_unset_language_omits_argument(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "Hello"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._load_stt_config", return_value={
                 "openai": {"language": ""},
             }), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(str(audio_file), "whisper-1")

        assert result["success"] is True
        assert "language" not in mock_client.audio.transcriptions.create.call_args.kwargs


    @pytest.mark.parametrize("language", ["en,fi", "en, fi", "en"])
    @pytest.mark.parametrize("source", ["provider", "global", "environment", "override", "hook"])
    def test_gpt_transcribe_legacy_hints_and_overrides(self, monkeypatch, tmp_path, language, source):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", language if source == "environment" else "")
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = {"text": "Hello"}
        config = {"provider": "openai", "cloud_trim_silence": False,
                  "language": language if source == "global" else "",
                  "openai": {"model": "gpt-transcribe"}}
        if source == "provider":
            config["openai"]["language"] = language
        elif source in ("override", "hook"):
            config["openai"].update({"language": "sv", "languages": ["sv"]})
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("hermes_cli.plugins.has_hook", return_value=source == "hook"), \
             patch("hermes_cli.plugins.invoke_hook", return_value=[{"language": language}]), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_cloud import _transcribe_openai
            from tools.transcription_tools import transcribe_audio
            result = (transcribe_audio(str(audio_file)) if source == "hook" else _transcribe_openai(
                str(audio_file), "gpt-transcribe",
                language=language if source == "override" else None,
            ))
        assert result["success"] is True, result
        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        expected = ["en"] if language == "en" else ["en", "fi"]
        assert kwargs["extra_body"] == {"languages": expected}
        assert "language" not in kwargs


    @pytest.mark.parametrize(("model", "settings", "expected"), [
        ("gpt-transcribe", {"language": "en,fi"}, {"languages[]": ["en", "fi"]}),
        ("gpt-transcribe", {"language": " en, fi, "}, {"languages[]": ["en", "fi"]}),
        ("gpt-transcribe", {"language": "en"}, {"languages[]": ["en"]}),
        ("gpt-transcribe", {"language": ""}, {}),
        ("whisper-1", {"language": "fi"}, {"language": ["fi"]}),
        ("gpt-4o-transcribe", {"language": "fi"}, {"language": ["fi"]}),
        pytest.param("gpt-transcribe", {"languages": ["en", "fi"]},
                     {"languages[]": ["en", "fi"]}, id="native-array"),
        pytest.param("gpt-transcribe", {"languages": [" en ", " fi "]},
                     {"languages[]": ["en", "fi"]}, id="trim-array"),
        pytest.param("gpt-transcribe", {"languages": ["fi"], "language": "sv"},
                     {"languages[]": ["fi"]}, id="array-precedence"),
        pytest.param("gpt-transcribe", {"languages": [], "language": "sv"},
                     {}, id="explicit-auto"),
        pytest.param("gpt-transcribe", {"languages": None, "language": "fi"},
                     {"languages[]": ["fi"]}, id="null-fallback"),
        pytest.param("whisper-1", {"languages": ["en", "fi"], "language": "sv"},
                     {"language": ["sv"]}, id="legacy-model"),
        pytest.param("gpt-transcribe", {"model": "whisper-large-v3", "languages": ["fi"]},
                     {"languages[]": ["fi"]}, id="autocorrect-array-precedence"),
        pytest.param("gpt-transcribe", {"model": "whisper-large-v3", "languages": []},
                     {}, id="autocorrect-explicit-auto"),
        pytest.param("gpt-transcribe", {"model": "whisper-large-v3", "languages": "fi"},
                     "stt.openai.languages must be an array of nonempty language-code strings",
                     id="autocorrect-invalid-array"),
        *[pytest.param("gpt-transcribe", {"languages": value, "language": "sv"},
                       "stt.openai.languages must be an array of nonempty language-code strings",
                       id=f"invalid-array-{index}")
          for index, value in enumerate(("en,fi", "", True, {"en": "fi"}, ["en", 42],
                                          [None], [" "], ["en,fi"]))],
    ])
    def test_language_hints_reach_multipart_request(self, monkeypatch, tmp_path, model, settings, expected):
        import wave
        from email import policy
        from email.parser import BytesParser

        import httpx
        import openai
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        if "model" in settings:
            # Mirror the import-time STT_OPENAI_MODEL default without reloading shared modules.
            monkeypatch.setattr("tools.transcription_cloud.DEFAULT_STT_MODEL", model)
        # Competing inherited hints must not override an explicit array (including []).
        inherited = "languages" in settings
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "pt" if inherited else "")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"stt": {
            "enabled": True, "provider": "openai", "language": "de" if inherited else "",
            "cloud_trim_silence": False,
            "openai": {"model": model, **settings},
        }}), encoding="utf-8")
        audio_file = tmp_path / "test.wav"
        with wave.open(str(audio_file), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x00\x00" * 16000)
        requests = []

        def respond(request):
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
                + request.read()
            )
            fields = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name != "file":
                    payload = part.get_payload(decode=True)
                    assert isinstance(payload, bytes)
                    fields.setdefault(name, []).append(payload.decode())
            requests.append(fields)
            if fields["response_format"] == ["text"]:
                return httpx.Response(200, text="test transcript")
            return httpx.Response(200, json={"text": "test transcript"})

        real_openai = openai.OpenAI
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: real_openai(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ))
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio(str(audio_file))

        if isinstance(expected, str):
            assert result["success"] is False, result
            assert expected in result["error"]
            assert requests == [], "Invalid config must fail before uploading audio"
            return
        assert result["success"] is True, result
        assert result["transcript"] == "test transcript"
        assert len(requests) == 1
        assert requests[0]["model"] == [model]
        actual = {key: values for key, values in requests[0].items()
                  if key in ("language", "languages", "languages[]")}
        assert actual == expected


# ---------------------------------------------------------------------------
# Main transcribe_audio() dispatch
# ---------------------------------------------------------------------------


class TestTranscribeAudio:

    def test_dispatches_to_local(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local", return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(audio_file))

        assert result["success"] is True
        mock_local.assert_called_once()


    def test_invalid_file_returns_error(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/file.ogg")
        assert result["success"] is False
        assert "not found" in result["error"]


class TestLocalFallback:

    def test_uses_installed_faster_whisper_without_changing_provider(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"provider": "openai", "local": {"model": "small"}},
        ), patch(
            "tools.transcription_tools._HAS_FASTER_WHISPER",
            True,
        ), patch(
            "tools.transcription_tools._transcribe_local",
            return_value={"success": True, "transcript": "local result"},
        ) as mock_local:
            from tools.transcription_tools import transcribe_audio_local_fallback

            result = transcribe_audio_local_fallback(str(audio_file))

        assert result["transcript"] == "local result"
        mock_local.assert_called_once_with(str(audio_file), "small")

    def test_does_not_install_when_no_local_backend_exists(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), patch(
            "tools.transcription_tools._has_local_command", return_value=False
        ):
            from tools.transcription_tools import transcribe_audio_local_fallback

            result = transcribe_audio_local_fallback(str(audio_file))

        assert result["success"] is False
        assert "installed local STT" in result["error"]


# ---------------------------------------------------------------------------
# Model name normalisation for local providers
# ---------------------------------------------------------------------------


class TestNormalizeLocalModel:
    """_normalize_local_model() maps cloud-only names to the local default."""

    def test_openai_model_name_maps_to_default(self):
        from tools.transcription_tools import _normalize_local_model, DEFAULT_LOCAL_MODEL
        assert _normalize_local_model("whisper-1") == DEFAULT_LOCAL_MODEL


    def test_local_transcribe_normalises_model(self):
        """transcribe_audio with local provider must not pass 'whisper-1' to WhisperModel."""
        import os
        from unittest.mock import MagicMock, patch

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(b"x")
            audio_file = f.name
        try:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (iter([]), MagicMock(language="en", duration=1.0))
            with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
                 patch("tools.transcription_tools._load_stt_config", return_value={
                     "enabled": True,
                     "provider": "local",
                     "local": {"model": "whisper-1"},
                 }), \
                 patch("tools.transcription_tools._local_model", None), \
                 patch("tools.transcription_tools._local_model_name", None), \
                 patch.dict("sys.modules", {"faster_whisper": _fake_faster_whisper_module(mock_model)}):
                mock_cls = __import__("faster_whisper").WhisperModel
                from tools.transcription_tools import transcribe_audio
                transcribe_audio(audio_file)
                # WhisperModel must NOT have been called with "whisper-1"
                call_args = mock_cls.call_args
                assert call_args is not None
                assert call_args[0][0] != "whisper-1", (
                    "WhisperModel was called with the cloud-only name 'whisper-1'"
                )
        finally:
            os.unlink(audio_file)
