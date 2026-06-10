"""Tests for the StepFun STT provider in transcription_tools.

Covers provider selection, dispatch, and the ``_transcribe_stepfun``
HTTP path. The real ``/v1/audio/asr/sse`` endpoint is not contacted —
we patch ``urllib.request.urlopen`` at the module boundary to keep these
tests network-free.

Mirrors the Moonshine provider shape: an opt-in provider that lives in
the dispatcher's elif chain but is intentionally NOT added to
``BUILTIN_STT_PROVIDERS`` or ``_BUILTIN_NAMES``. See the
``hermes-stt-provider`` skill's "deliberately-not-builtin" pattern.
"""

from unittest.mock import patch, MagicMock

import pytest


pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


# ---------------------------------------------------------------------------
# Provider selection — stepfun
# ---------------------------------------------------------------------------


class TestGetProviderStepfun:
    """``_get_provider`` honours an explicit ``provider: stepfun`` config."""

    def test_stepfun_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "stepfun"}) == "stepfun"

    def test_stepfun_when_api_key_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "stepfun"}) == "none"

    def test_stepfun_does_not_silently_fall_back_to_cloud(self, monkeypatch):
        """Explicit stepfun without an API key must not silently route elsewhere."""
        monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "stepfun"}) == "none"

    def test_stepfun_not_in_builtin_set(self):
        """StepFun is intentionally not a BUILTIN_STT_PROVIDER — keeps the
        registry/dispatcher sync invariant free of stepfun-specific entries."""
        from tools.transcription_tools import BUILTIN_STT_PROVIDERS
        assert "stepfun" not in BUILTIN_STT_PROVIDERS


# ---------------------------------------------------------------------------
# Dispatch — transcribe_audio() routes provider=stepfun to _transcribe_stepfun
# ---------------------------------------------------------------------------


class TestTranscribeAudioStepfunDispatch:
    """``transcribe_audio()`` dispatches to ``_transcribe_stepfun``."""

    def test_dispatch_routes_to_stepfun_handler(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "")  # no env override
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)  # any bytes; handler is mocked

        # Pytest's conftest redirects HERMES_HOME to a tempdir, so we have
        # to inject the stepfun config explicitly. The handler itself is
        # mocked so we don't need a real network call.
        stepfun_cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ko"},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=stepfun_cfg), \
             patch(
                 "tools.transcription_tools._transcribe_stepfun",
                 return_value={"success": True, "transcript": "안녕하세요", "provider": "stepfun"},
             ) as mock_handler:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(wav))

        assert result["success"] is True
        assert result["provider"] == "stepfun"
        assert result["transcript"] == "안녕하세요"
        # The handler was called exactly once with the audio path
        assert mock_handler.call_count == 1
        assert str(wav) in mock_handler.call_args.args[0]

    def test_dispatch_uses_configured_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ja"},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch(
                 "tools.transcription_tools._transcribe_stepfun",
                 return_value={"success": True, "transcript": "こんにちは", "provider": "stepfun"},
             ) as mock_handler:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(wav))

        args, _ = mock_handler.call_args
        assert args[0] == str(wav)
        assert args[1] == "stepaudio-2.5-asr"
        assert result["transcript"] == "こんにちは"

    def test_handler_not_called_when_provider_is_other(self, tmp_path, monkeypatch):
        """Regression guard: stepfun handler must not be invoked for non-stepfun providers."""
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        cfg = {"enabled": True, "provider": "local"}
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools._transcribe_stepfun") as mock_stepfun, \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hello", "provider": "local"}):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(wav))

        assert result["provider"] == "local"
        mock_stepfun.assert_not_called()


# ---------------------------------------------------------------------------
# Error envelope shape — mirrors the Moonshine / ElevenLabs / xai handlers
# ---------------------------------------------------------------------------


class TestTranscribeStepfunErrorEnvelope:
    """The handler must return the standard error envelope shape so the
    gateway/CLI can rely on ``{success, transcript, error}``."""

    def test_missing_api_key_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("tools.transcription_tools._load_stt_config", return_value={"stepfun": {}}):
            from tools.transcription_tools import _transcribe_stepfun
            result = _transcribe_stepfun(str(wav), "stepaudio-2.5-asr")

        assert result["success"] is False
        assert result["transcript"] == ""
        assert "STEPFUN_API_KEY" in result["error"]

    def test_missing_file_returns_error(self, tmp_path, monkeypatch):
        """Missing file is caught by the dispatcher-level _validate_audio_file
        (called before the handler), so the error envelope is returned before
        the stepfun handler is even invoked. This test verifies that contract."""
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        nonexistent = tmp_path / "does_not_exist.wav"

        cfg = {"enabled": True, "provider": "stepfun", "stepfun": {"model": "stepaudio-2.5-asr"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools._transcribe_stepfun") as mock_stepfun:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(nonexistent))

        assert result["success"] is False
        assert "not found" in result["error"].lower() or "not a file" in result["error"].lower()
        # The stepfun handler should never have been called — file validation
        # happens upstream in the dispatcher.
        mock_stepfun.assert_not_called()

    def test_step_plan_base_url_is_preserved(self, tmp_path, monkeypatch):
        """The step_plan base URL is PRESERVED (not stripped) so Step Plan
        subscribers consume plan quota instead of PAYG. This matches the
        TTS pattern in tools/tts_tool.py:_normalize_stepfun_tts_base_url."""
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/step_plan/v1")
        wav = tmp_path / "voice.ogg"
        wav.write_bytes(b"OggS" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ko"},
        }

        # Build a fake response that yields the SSE transcript stream
        fake_resp = MagicMock()
        fake_resp.__iter__ = lambda self: iter([
            b'data: {"type": "transcript.text.delta", "delta": "Hello"}',
            b'data: {"type": "transcript.text.delta", "delta": " world"}',
            b'data: [DONE]',
        ])
        fake_resp.__enter__ = lambda self: self
        fake_resp.__exit__ = lambda self, *a: None

        with patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools.urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            from tools.transcription_tools import _transcribe_stepfun
            result = _transcribe_stepfun(str(wav), "stepaudio-2.5-asr")

        # The urlopen call should hit /step_plan/v1/audio/asr/sse — NOT /v1/audio/asr/sse
        # Stripping /step_plan/ would silently route to PAYG and waste plan quota.
        called_url = mock_urlopen.call_args.args[0].full_url
        assert called_url == "https://api.stepfun.ai/step_plan/v1/audio/asr/sse", \
            f"Expected Step Plan URL preserved, got: {called_url}"

    def test_v1_base_url_works_for_payg_users(self, tmp_path, monkeypatch):
        """PAYG users with STEPFUN_BASE_URL=https://api.stepfun.ai/v1 still work."""
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/v1")
        wav = tmp_path / "voice.ogg"
        wav.write_bytes(b"OggS" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ko"},
        }

        fake_resp = MagicMock()
        fake_resp.__iter__ = lambda self: iter([
            b'data: {"type": "transcript.text.delta", "delta": "Hi"}',
            b'data: [DONE]',
        ])
        fake_resp.__enter__ = lambda self: self
        fake_resp.__exit__ = lambda self, *a: None

        with patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools.urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            from tools.transcription_tools import _transcribe_stepfun
            result = _transcribe_stepfun(str(wav), "stepaudio-2.5-asr")

        called_url = mock_urlopen.call_args.args[0].full_url
        assert called_url == "https://api.stepfun.ai/v1/audio/asr/sse"

    def test_bare_domain_is_anchored_to_v1(self):
        """`https://api.stepfun.ai` (no /v1) gets anchored to /v1."""
        from tools.transcription_tools import _normalize_stepfun_stt_base_url
        assert _normalize_stepfun_stt_base_url("https://api.stepfun.ai") == "https://api.stepfun.ai/v1"
        assert _normalize_stepfun_stt_base_url("https://api.stepfun.ai/") == "https://api.stepfun.ai/v1"
        assert _normalize_stepfun_stt_base_url("https://api.stepfun.ai/v2") == "https://api.stepfun.ai/v1"

    def test_default_base_url_is_step_plan(self):
        """When STEPFUN_BASE_URL is unset, default to /step_plan/v1 so the
        common case (subscriber) just works without configuration."""
        from tools.transcription_tools import _normalize_stepfun_stt_base_url
        assert _normalize_stepfun_stt_base_url("") == "https://api.stepfun.ai/step_plan/v1"
        assert _normalize_stepfun_stt_base_url(None) == "https://api.stepfun.ai/step_plan/v1"

    def test_unknown_model_logs_warning(self, tmp_path, caplog, monkeypatch):
        """Misconfigured model names (e.g. stepaudio-2-asr-pro, which the
        StepFun API 404s) should produce a logger.warning so the user sees
        the issue before burning a request. The handler still runs (we
        don't pre-validate server-side), but the warning is loud."""
        import logging
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/v1")
        wav = tmp_path / "voice.ogg"
        wav.write_bytes(b"OggS" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2-asr-pro", "language": "ko"},
        }

        fake_resp = MagicMock()
        fake_resp.__iter__ = lambda self: iter([b'data: [DONE]'])
        fake_resp.__enter__ = lambda self: self
        fake_resp.__exit__ = lambda self, *a: None

        with caplog.at_level(logging.WARNING, logger="tools.transcription_tools"), \
             patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools.urllib.request.urlopen", return_value=fake_resp):
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(str(wav))

        warnings = [r for r in caplog.records if "stepaudio-2-asr-pro" in r.getMessage()]
        assert len(warnings) >= 1, f"Expected warning about unknown model, got: {[r.getMessage() for r in caplog.records]}"
        assert "not in the known catalogue" in warnings[0].getMessage()

    def test_known_model_does_not_warn(self, tmp_path, caplog, monkeypatch):
        """The default stepaudio-2.5-asr must NOT trigger the unknown-model warning."""
        import logging
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.stepfun.ai/v1")
        wav = tmp_path / "voice.ogg"
        wav.write_bytes(b"OggS" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ko"},
        }

        fake_resp = MagicMock()
        fake_resp.__iter__ = lambda self: iter([
            b'data: {"type": "transcript.text.delta", "delta": "Hi"}',
            b'data: [DONE]',
        ])
        fake_resp.__enter__ = lambda self: self
        fake_resp.__exit__ = lambda self, *a: None

        with caplog.at_level(logging.WARNING, logger="tools.transcription_tools"), \
             patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools.urllib.request.urlopen", return_value=fake_resp):
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(str(wav))

        unknown_warnings = [r for r in caplog.records if "not in the known catalogue" in r.getMessage()]
        assert len(unknown_warnings) == 0, f"Got unexpected warning: {[r.getMessage() for r in unknown_warnings]}"

    def test_successful_transcription_collects_deltas(self, tmp_path, monkeypatch):
        """SSE transcript.text.delta events are concatenated into the final transcript."""
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "")  # use default
        wav = tmp_path / "voice.ogg"
        wav.write_bytes(b"OggS" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ko"},
        }

        fake_resp = MagicMock()
        fake_resp.__iter__ = lambda self: iter([
            b'data: {"type": "transcript.text.delta", "delta": "Hello "}',
            b'data: {"type": "transcript.text.delta", "delta": "world"}',
            b'data: {"type": "transcript.text.done"}',
        ])
        fake_resp.__enter__ = lambda self: self
        fake_resp.__exit__ = lambda self, *a: None

        with patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools.urllib.request.urlopen", return_value=fake_resp):
            from tools.transcription_tools import _transcribe_stepfun
            result = _transcribe_stepfun(str(wav), "stepaudio-2.5-asr")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"
        assert result["provider"] == "stepfun"

    def test_empty_transcript_returns_error(self, tmp_path, monkeypatch):
        """An SSE stream with no delta events returns a clean error envelope."""
        monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", "")
        wav = tmp_path / "voice.ogg"
        wav.write_bytes(b"OggS" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "stepfun",
            "stepfun": {"model": "stepaudio-2.5-asr", "language": "ko"},
        }

        fake_resp = MagicMock()
        fake_resp.__iter__ = lambda self: iter([b'data: [DONE]'])
        fake_resp.__enter__ = lambda self: self
        fake_resp.__exit__ = lambda self, *a: None

        with patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools.urllib.request.urlopen", return_value=fake_resp):
            from tools.transcription_tools import _transcribe_stepfun
            result = _transcribe_stepfun(str(wav), "stepaudio-2.5-asr")

        assert result["success"] is False
        assert "empty transcript" in result["error"]
