"""Tests for tools.transcription_tools — three-provider STT pipeline.

Covers the full provider matrix (local, groq, openai), fallback chains,
model auto-correction, config loading, validation edge cases, and
end-to-end dispatch.  All external dependencies are mocked.
"""

import os
import sys
import struct
import subprocess
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

if "faster_whisper" not in sys.modules:
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock(name="WhisperModel")
    # Set ``__spec__`` so ``importlib.util.find_spec("faster_whisper")``
    # doesn't raise ``ValueError: faster_whisper.__spec__ is None`` during
    # collection (used by skipif markers further down in this file).
    from importlib.machinery import ModuleSpec
    faster_whisper_stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = faster_whisper_stub


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(silence)

    return str(wav_path)


@pytest.fixture
def sample_ogg(tmp_path):
    """Create a fake OGG file for validation tests."""
    ogg_path = tmp_path / "test.ogg"
    ogg_path.write_bytes(b"fake audio data")
    return str(ogg_path)

@pytest.fixture
def sample_silk(tmp_path):
    """Create a fake WeChat .silk file for preprocessing tests."""
    silk_path = tmp_path / "voice.silk"
    silk_path.write_bytes(b"\x02#!SILK_V3fake")
    return str(silk_path)



@pytest.fixture
def oversized_wav(tmp_path):
    """Create a sparse WAV-shaped file just above the remote upload cap."""
    from tools.transcription_tools import MAX_FILE_SIZE

    wav_path = tmp_path / "oversized.wav"
    with wav_path.open("wb") as audio_file:
        audio_file.seek(MAX_FILE_SIZE)
        audio_file.write(b"\0")
    return str(wav_path)


pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real API keys leak into tests."""
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)


# ============================================================================
# _get_provider — full permutation matrix
# ============================================================================

class TestGetProviderGroq:
    """Groq-specific provider selection tests."""

    def test_groq_when_key_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "groq"

    def test_groq_explicit_no_fallback(self, monkeypatch):
        """Explicit groq with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "none"

    def test_groq_nothing_available(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "none"


class TestGetProviderFallbackPriority:
    """Auto-detect fallback priority and explicit provider behaviour."""

    def test_auto_detect_prefers_local(self):
        """Auto-detect prefers local over any cloud provider."""
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"

    def test_auto_detect_prefers_groq_over_openai(self, monkeypatch):
        """Auto-detect: groq (free) is preferred over openai (paid)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "groq"

    def test_explicit_openai_no_key_returns_none(self, monkeypatch):
        """Explicit openai with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "none"

    def test_unknown_provider_passed_through(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "custom-endpoint"}) == "custom-endpoint"

    def test_empty_config_defaults_to_local(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"


# ============================================================================
# Explicit provider config respected  (GH-1774)
# ============================================================================

class TestExplicitProviderRespected:
    """When stt.provider is explicitly set, that choice is authoritative.
    No silent fallback to a different cloud provider."""

    def test_explicit_local_no_fallback_to_openai(self, monkeypatch):
        """GH-1774: provider=local must not silently fall back to openai
        even when an OpenAI API key is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "***")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "none", f"Expected 'none' but got {result!r}"

    def test_explicit_local_no_fallback_to_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "none"

    def test_explicit_local_uses_local_command_fallback(self, monkeypatch):
        """Local-to-local_command fallback is fine — both are local."""
        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --output_dir {output_dir} --language {language}",
        )
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "local"})
            assert result == "local_command"

    def test_explicit_groq_no_fallback_to_openai(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "groq"})
            assert result == "none"

    def test_explicit_openai_no_fallback_to_groq(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({"provider": "openai"})
            assert result == "none"

    def test_auto_detect_still_falls_back_to_cloud(self, monkeypatch):
        """When no provider is explicitly set, auto-detect cloud fallback works."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            # Empty dict = no explicit provider, uses DEFAULT_PROVIDER auto-detect
            result = _get_provider({})
            assert result == "openai"

    def test_auto_detect_prefers_groq_over_openai(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            result = _get_provider({})
            assert result == "groq"


# ============================================================================
# _transcribe_groq
# ============================================================================

class TestTranscribeGroq:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_groq
        result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "GROQ_API_KEY" in result["error"]

    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert result["provider"] == "groq"
        mock_client.close.assert_called_once()

    def test_whitespace_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello world  \n"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["transcript"] == "hello world"

    def test_uses_groq_base_url(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client) as mock_openai_cls:
            from tools.transcription_tools import _transcribe_groq, GROQ_BASE_URL
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        call_kwargs = mock_openai_cls.call_args
        assert call_kwargs.kwargs["base_url"] == GROQ_BASE_URL

    def test_api_error_returns_failure(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is False
        assert "API error" in result["error"]
        mock_client.close.assert_called_once()

    def test_permission_error(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = PermissionError("denied")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    def test_language_hint_omitted_when_unset(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hi"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in kwargs

    def test_language_hint_from_config(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hola"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch(
                 "tools.transcription_tools._load_stt_config",
                 return_value={"groq": {"language": "es"}},
             ):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["language"] == "es"

    def test_language_hint_from_env_when_config_missing(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "hu")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "szia"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["language"] == "hu"

    def test_language_config_overrides_env(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "hu")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch(
                 "tools.transcription_tools._load_stt_config",
                 return_value={"groq": {"language": "en"}},
             ):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["language"] == "en"

    def test_language_whitespace_treated_as_unset(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hi"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch(
                 "tools.transcription_tools._load_stt_config",
                 return_value={"groq": {"language": "   "}},
             ):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in kwargs

    def test_null_groq_subsection_is_safe(self, monkeypatch, sample_wav):
        """`stt.groq: null` in YAML yields None; must not raise, auto-detect stays intact."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hi"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch(
                 "tools.transcription_tools._load_stt_config",
                 return_value={"groq": None},
             ):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is True
        kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert "language" not in kwargs


# ============================================================================
# _transcribe_openai — additional tests
# ============================================================================

class TestTranscribeOpenAIExtended:
    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai("/tmp/test.ogg", "whisper-1")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_uses_openai_base_url(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client) as mock_openai_cls:
            from tools.transcription_tools import _transcribe_openai, OPENAI_BASE_URL
            _transcribe_openai(sample_wav, "whisper-1")

        call_kwargs = mock_openai_cls.call_args
        assert call_kwargs.kwargs["base_url"] == OPENAI_BASE_URL

    def test_whitespace_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello  \n"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1")

        assert result["transcript"] == "hello"
        mock_client.close.assert_called_once()

    def test_permission_error(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = PermissionError("denied")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1")

        assert result["success"] is False
        assert "Permission denied" in result["error"]
        mock_client.close.assert_called_once()


class TestTranscribeLocalCommand:
    def test_command_provider_uses_sanitized_child_env(self, monkeypatch):
        """Salvage of #56332: command STT must not inherit Hermes secrets."""
        monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-vision")
        monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("MY_SAFE_STT_VAR", "keep")

        captured = {}

        class _Stream:
            def read(self, size):
                return ""

        class Proc:
            returncode = 0
            stdout = _Stream()
            stderr = _Stream()

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return Proc()

        monkeypatch.setattr("tools.transcription_tools.subprocess.Popen", fake_popen)

        from tools.transcription_tools import _run_command_stt

        result = _run_command_stt("echo hi", timeout=1)

        assert result.returncode == 0
        env = captured["env"]
        assert "AUXILIARY_VISION_API_KEY" not in env
        assert "GATEWAY_RELAY_SECRET" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["MY_SAFE_STT_VAR"] == "keep"

    def test_local_whisper_subprocess_uses_sanitized_env(
        self, monkeypatch, sample_wav, tmp_path
    ):
        """Sibling path: local whisper subprocess.run also scrubbed (#56332 gap)."""
        monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-vision")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("MY_SAFE_LOCAL_STT", "keep")
        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --model {model} --output_dir {output_dir} --language {language}",
        )

        captured = {}
        out_dir = tmp_path / "local-out"
        out_dir.mkdir()
        (out_dir / "transcript.txt").write_text("hello", encoding="utf-8")

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self_inner):
                    return str(out_dir)

                def __exit__(self_inner, *exc):
                    return False

            return _TempDir()

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs.get("env")
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr("tools.transcription_tools.tempfile.TemporaryDirectory", fake_tempdir)
        monkeypatch.setattr("tools.transcription_tools.subprocess.run", fake_run)
        monkeypatch.setattr(
            "tools.transcription_tools._prepare_local_audio",
            lambda *a, **k: (str(sample_wav), None),
        )

        from tools.transcription_tools import _transcribe_local_command

        result = _transcribe_local_command(str(sample_wav), "base")
        assert result["success"] is True
        env = captured["env"]
        assert env is not None
        assert "AUXILIARY_VISION_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["MY_SAFE_LOCAL_STT"] == "keep"

    def test_auto_detects_local_whisper_binary(self, monkeypatch):
        monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
        monkeypatch.setattr("tools.transcription_tools._find_whisper_binary", lambda: "/opt/homebrew/bin/whisper")

        from tools.transcription_tools import _get_local_command_template

        template = _get_local_command_template()

        assert template is not None
        assert template.startswith("/opt/homebrew/bin/whisper ")
        assert "{model}" in template
        assert "{output_dir}" in template

    def test_command_fallback_with_template(self, monkeypatch, sample_ogg, tmp_path):
        out_dir = tmp_path / "local-out"
        out_dir.mkdir()

        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "whisper {input_path} --model {model} --output_dir {output_dir} --language {language}",
        )
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "en")

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self_inner):
                    return str(out_dir)

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _TempDir()

        def fake_run(cmd, *args, **kwargs):
            assert isinstance(cmd, list)
            (out_dir / "test.txt").write_text("hello from local command\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("tools.transcription_tools.tempfile.TemporaryDirectory", fake_tempdir)
        monkeypatch.setattr("tools.transcription_tools._find_ffmpeg_binary", lambda: "/opt/homebrew/bin/ffmpeg")
        monkeypatch.setattr("tools.transcription_tools.subprocess.run", fake_run)

        from tools.transcription_tools import _transcribe_local_command

        result = _transcribe_local_command(sample_ogg, "base")

        assert result["success"] is True
        assert result["transcript"] == "hello from local command"
        assert result["provider"] == "local_command"


# ============================================================================
# _transcribe_local — additional tests
# ============================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faster_whisper"),
    reason="faster_whisper not installed",
)
class TestTranscribeLocalExtended:
    def test_model_reuse_on_second_call(self, tmp_path):
        """Second call with same model should NOT reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "base")

        # WhisperModel should be created only once
        assert mock_whisper_cls.call_count == 1

    def test_model_reloaded_on_change(self, tmp_path):
        """Switching model name should reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "small")

        assert mock_whisper_cls.call_count == 2

    def test_exception_returns_failure(self, tmp_path):
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_whisper_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "large-v3")

        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]

    def test_config_device_and_compute_type_passed_to_whisper(self, tmp_path):
        """User-configured device and compute_type should be forwarded to WhisperModel.

        Regression test for #8319: these values were hardcoded to "auto".
        """
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        fake_config = {
            "local": {
                "device": "cpu",
                "compute_type": "float32",
            }
        }

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools._load_stt_config", return_value=fake_config):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        mock_whisper_cls.assert_called_once_with("base", device="cpu", compute_type="float32")

    def test_config_defaults_to_auto_when_not_set(self, tmp_path):
        """Without config, device and compute_type should default to "auto"."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")

        mock_whisper_cls.assert_called_once_with("base", device="auto", compute_type="auto")

    def test_multiple_segments_joined(self, tmp_path):
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg1 = MagicMock()
        seg1.text = "Hello"
        seg2 = MagicMock()
        seg2.text = " world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", return_value=mock_model), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"

    def test_apple_silicon_forces_cpu_without_auto_probe(self, tmp_path):
        """Apple Silicon/Rosetta should skip device='auto' to avoid SIGABRT."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "safe"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([seg], info)
        mock_whisper_cls = MagicMock(return_value=cpu_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._should_force_faster_whisper_cpu", return_value=True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "safe"
        mock_whisper_cls.assert_called_once_with("base", device="cpu", compute_type="int8")

    def test_force_cpu_detects_rosetta_on_apple_silicon(self):
        from tools.transcription_tools import _should_force_faster_whisper_cpu

        with patch("tools.transcription_tools.platform.system", return_value="Darwin"), \
             patch("tools.transcription_tools.platform.machine", return_value="x86_64"), \
             patch("tools.transcription_tools._sysctl_value", side_effect=lambda key: {
                 "sysctl.proc_translated": "1",
                 "hw.optional.arm64": "1",
             }.get(key, "")):
            assert _should_force_faster_whisper_cpu() is True

    def test_force_cpu_false_on_intel_macos(self):
        from tools.transcription_tools import _should_force_faster_whisper_cpu

        with patch("tools.transcription_tools.platform.system", return_value="Darwin"), \
             patch("tools.transcription_tools.platform.machine", return_value="x86_64"), \
             patch("tools.transcription_tools._sysctl_value", return_value="0"):
            assert _should_force_faster_whisper_cpu() is False

    def test_load_time_cuda_lib_failure_falls_back_to_cpu(self, tmp_path):
        """Missing libcublas at load time → reload on CPU, succeed."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "hi"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([seg], info)

        call_args = []

        def fake_whisper(model_name, device, compute_type):
            call_args.append((device, compute_type))
            if device == "auto":
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            return cpu_model

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._should_force_faster_whisper_cpu", return_value=False), \
             patch("faster_whisper.WhisperModel", side_effect=fake_whisper), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "hi"
        assert call_args == [("auto", "auto"), ("cpu", "int8")]

    def test_runtime_cuda_lib_failure_evicts_cache_and_retries_on_cpu(self, tmp_path):
        """libcublas dlopen fails at transcribe() → evict cache, reload CPU, retry."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "recovered"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        # First model loads fine (auto), but transcribe() blows up on dlopen
        gpu_model = MagicMock()
        gpu_model.transcribe.side_effect = RuntimeError(
            "Library libcublas.so.12 is not found or cannot be loaded"
        )
        # Second model (forced CPU) works
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([seg], info)

        models = [gpu_model, cpu_model]
        call_args = []

        def fake_whisper(model_name, device, compute_type):
            call_args.append((device, compute_type))
            return models.pop(0)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._should_force_faster_whisper_cpu", return_value=False), \
             patch("faster_whisper.WhisperModel", side_effect=fake_whisper), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "recovered"
        # First load is auto, retry forces CPU.
        assert call_args == [("auto", "auto"), ("cpu", "int8")]
        # Cached-bad-model eviction: the broken GPU model was called once,
        # then discarded; the CPU model served the retry.
        assert gpu_model.transcribe.call_count == 1
        assert cpu_model.transcribe.call_count == 1

    def test_cublas_status_not_supported_retries_on_cpu(self, tmp_path):
        """Blackwell cuBLAS unsupported errors should use the CPU fallback path."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "blackwell fallback"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        gpu_model = MagicMock()
        gpu_model.transcribe.side_effect = RuntimeError(
            "cuBLAS failed with status CUBLAS_STATUS_NOT_SUPPORTED"
        )
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = ([seg], info)

        models = [gpu_model, cpu_model]
        call_args = []

        def fake_whisper(model_name, device, compute_type):
            call_args.append((device, compute_type))
            return models.pop(0)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", side_effect=fake_whisper), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "blackwell fallback"
        assert call_args == [("auto", "auto"), ("cpu", "int8")]

    def test_cuda_out_of_memory_does_not_trigger_cpu_fallback(self, tmp_path):
        """'CUDA out of memory' is a real error, not a missing lib — surface it."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_whisper_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        # Single call — no CPU retry, because OOM isn't a missing-lib symptom.
        assert mock_whisper_cls.call_count == 1
        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]


# ============================================================================
# Model auto-correction
# ============================================================================

class TestModelAutoCorrection:
    def test_groq_corrects_openai_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "whisper-1")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_groq_corrects_gpt4o_transcribe(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "gpt-4o-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_openai_corrects_groq_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai, DEFAULT_STT_MODEL
            _transcribe_openai(sample_wav, "whisper-large-v3-turbo")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_openai_corrects_distil_whisper(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai, DEFAULT_STT_MODEL
            _transcribe_openai(sample_wav, "distil-whisper-large-v3-en")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_compatible_groq_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-large-v3"

    def test_compatible_openai_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "gpt-4o-mini-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini-transcribe"

    def test_gpt_transcribe_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "gpt-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-transcribe"
        assert call_kwargs.kwargs["response_format"] == "json"

    def test_gpt_transcribe_language_hint_uses_languages_list(self, monkeypatch, sample_wav):
        """gpt-transcribe rejects the singular ``language`` field; the hint
        must be sent as a ``languages`` list via extra_body instead."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch("tools.transcription_tools._resolve_stt_language", return_value="fr"):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "gpt-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert "language" not in call_kwargs.kwargs
        assert call_kwargs.kwargs["extra_body"] == {"languages": ["fr"]}

    def test_legacy_openai_model_language_hint_uses_singular_field(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch("tools.transcription_tools._resolve_stt_language", return_value="fr"):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "gpt-4o-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["language"] == "fr"
        assert "extra_body" not in call_kwargs.kwargs

    def test_gpt_transcribe_rejected_on_groq(self, monkeypatch, sample_wav):
        """gpt-transcribe is OpenAI-only and must be auto-corrected on Groq."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "gpt-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_unknown_model_passes_through_groq(self, monkeypatch, sample_wav):
        """A model not in either known set should not be overridden."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"

    def test_unknown_model_passes_through_openai(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"


# ============================================================================
# _load_stt_config
# ============================================================================

class TestLoadSttConfig:
    def test_returns_dict_when_import_fails(self):
        with patch("tools.transcription_tools._load_stt_config") as mock_load:
            mock_load.return_value = {}
            from tools.transcription_tools import _load_stt_config
            assert _load_stt_config() == {}

    def test_real_load_returns_dict(self):
        """_load_stt_config should always return a dict, even on import error."""
        with patch.dict("sys.modules", {"hermes_cli": None, "hermes_cli.config": None}):
            from tools.transcription_tools import _load_stt_config
            result = _load_stt_config()
        assert isinstance(result, dict)


# ============================================================================
# _validate_audio_file — edge cases
# ============================================================================

class TestValidateAudioFileEdgeCases:
    def test_directory_is_not_a_file(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        # tmp_path itself is a directory with an .ogg-ish name? No.
        # Create a directory with a valid audio extension
        d = tmp_path / "audio.ogg"
        d.mkdir()
        result = _validate_audio_file(str(d))
        assert result is not None
        assert "not a file" in result["error"]

    def test_symlink_with_supported_extension_is_rejected(self, tmp_path):
        if not hasattr(os, "symlink"):
            pytest.skip("symlinks are not supported on this platform")

        target = tmp_path / "target.txt"
        target.write_bytes(b"not audio")
        link = tmp_path / "linked.wav"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        from tools.transcription_tools import _validate_audio_file
        result = _validate_audio_file(str(link))
        assert result is not None
        assert "symbolic link" in result["error"]

    def test_stat_oserror(self, tmp_path):
        f = tmp_path / "test.ogg"
        f.write_bytes(b"data")
        from tools.transcription_tools import _validate_audio_file

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", side_effect=OSError("disk error")):
            result = _validate_audio_file(str(f))

        assert result is not None
        assert "Failed to access" in result["error"]

    def test_all_supported_formats_accepted(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file, SUPPORTED_FORMATS
        for fmt in SUPPORTED_FORMATS:
            f = tmp_path / f"test{fmt}"
            f.write_bytes(b"data")
            assert _validate_audio_file(str(f)) is None, f"Format {fmt} should be accepted"

    def test_telegram_oga_and_opus_accepted(self, tmp_path):
        # Telegram delivers voice notes as .oga (OGG/Opus); .opus is the
        # bare-codec sibling. Both must pass validation or every inbound
        # voice note fails before reaching any STT backend.
        from tools.transcription_tools import _validate_audio_file
        for fmt in (".oga", ".opus"):
            f = tmp_path / f"voice{fmt}"
            f.write_bytes(b"data")
            assert _validate_audio_file(str(f)) is None

    def test_case_insensitive_extension(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        f = tmp_path / "test.MP3"
        f.write_bytes(b"data")
        assert _validate_audio_file(str(f)) is None


# ============================================================================
# transcribe_audio — end-to-end dispatch
# ============================================================================

class TestTranscribeAudioDispatch:
    def test_dispatches_to_groq(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "groq"}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi", "provider": "groq"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "groq"
        mock_groq.assert_called_once()

    def test_dispatches_to_local(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_local.assert_called_once()

    def test_oversized_local_file_reaches_dispatcher(self, oversized_wav):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(oversized_wav)

        assert result["success"] is True
        mock_local.assert_called_once()

    def test_oversized_local_command_file_reaches_dispatcher(self, oversized_wav):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "local_command"}), \
             patch("tools.transcription_tools._get_provider", return_value="local_command"), \
             patch("tools.transcription_tools._transcribe_local_command",
                   return_value={"success": True, "transcript": "hi"}) as mock_command:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(oversized_wav)

        assert result["success"] is True
        mock_command.assert_called_once()

    def test_oversized_remote_file_is_rejected_before_dispatch(self, oversized_wav):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai") as mock_openai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(oversized_wav)

        assert result["success"] is False
        assert "File too large" in result["error"]
        mock_openai.assert_not_called()

    def test_dispatches_to_openai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai",
                   return_value={"success": True, "transcript": "hi", "provider": "openai"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_openai.assert_called_once()

    def test_no_provider_returns_error(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]
        assert "faster-whisper" in result["error"]
        assert "GROQ_API_KEY" in result["error"]

    def test_explicit_openai_no_key_returns_error(self, monkeypatch, sample_ogg):
        """Explicit provider=openai with no key returns an error, not a fallback."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]

    def test_invalid_file_short_circuits(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/audio.wav")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_model_override_passed_to_groq(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="whisper-large-v3")

        _, kwargs = mock_groq.call_args
        assert kwargs.get("model_name") or mock_groq.call_args[0][1] == "whisper-large-v3"

    def test_model_override_passed_to_local(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="large-v3")

        assert mock_local.call_args[0][1] == "large-v3"

    def test_converts_silk_before_dispatch(self, sample_silk):
        with patch("tools.transcription_tools._prepare_audio_for_transcription",
                   return_value=("/tmp/converted.wav", "/tmp/hermes-silk-123", None),
                   create=True) as mock_prepare, \
             patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local, \
             patch("tools.transcription_tools.shutil.rmtree") as mock_rmtree:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_silk)

        assert result["success"] is True
        mock_prepare.assert_called_once_with(sample_silk)
        mock_local.assert_called_once_with("/tmp/converted.wav", "base")
        mock_rmtree.assert_called_once_with("/tmp/hermes-silk-123", ignore_errors=True)

    def test_silk_symlink_is_rejected_before_preprocessing(self, tmp_path):
        """A Silk symlink must not reach the decoder before path safety validation."""
        if not hasattr(os, "symlink"):
            pytest.skip("symlinks are not supported on this platform")

        target = tmp_path / "voice.silk"
        target.write_bytes(b"\x02#!SILK_V3fake")
        link = tmp_path / "linked.silk"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        with patch(
            "tools.transcription_tools._prepare_audio_for_transcription", create=True
        ) as mock_prepare:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(link))

        assert result["success"] is False
        assert "symbolic link" in result["error"]
        mock_prepare.assert_not_called()

    def test_oversized_silk_is_rejected_before_preprocessing(self, tmp_path):
        """A Silk source over the upload limit must not reach the decoder."""
        silk_path = tmp_path / "oversized.silk"
        from tools.transcription_tools import MAX_FILE_SIZE
        with silk_path.open("wb") as audio_file:
            audio_file.truncate(MAX_FILE_SIZE + 1)

        with patch(
            "tools.transcription_tools._prepare_audio_for_transcription", create=True
        ) as mock_prepare:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(silk_path))

        assert result["success"] is False
        assert "File too large" in result["error"]
        mock_prepare.assert_not_called()

    def test_default_model_used_when_none(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio, DEFAULT_GROQ_STT_MODEL
            transcribe_audio(sample_ogg, model=None)

        assert mock_groq.call_args[0][1] == DEFAULT_GROQ_STT_MODEL

    def test_config_local_model_used(self, sample_ogg):
        config = {"local": {"model": "small"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_local.call_args[0][1] == "small"

    def test_config_openai_model_used(self, sample_ogg):
        config = {"openai": {"model": "gpt-4o-transcribe"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai",
                   return_value={"success": True, "transcript": "hi"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_openai.call_args[0][1] == "gpt-4o-transcribe"

    def test_command_provider_falls_back_to_local_when_configured(self, sample_ogg):
        config = {
            "provider": "parakeet",
            "providers": {
                "parakeet": {
                    "type": "command",
                    "command": "parakeet-transcribe {input}",
                    "fallback_provider": "local",
                }
            },
            "local": {"model": "small"},
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="parakeet"), \
             patch("tools.transcription_tools._transcribe_command_stt",
                   return_value={"success": False, "error": "missing binary"}), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "fallback text", "provider": "local"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["transcript"] == "fallback text"
        assert result["fallback_from"] == "parakeet"
        assert result["fallback_reason"] == "missing binary"
        assert "fell back to local faster-whisper" in result["warning"]
        mock_local.assert_called_once_with(sample_ogg, "small")

    def test_command_provider_local_fallback_accepts_null_local_config(self, sample_ogg):
        config = {
            "provider": "parakeet",
            "providers": {
                "parakeet": {
                    "type": "command",
                    "command": "parakeet-transcribe {input}",
                    "fallback_provider": "local",
                }
            },
            "local": None,
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="parakeet"), \
             patch("tools.transcription_tools._transcribe_command_stt",
                   return_value={"success": False, "error": "missing binary"}), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "fallback text", "provider": "local"}) as mock_local:
            from tools.transcription_tools import transcribe_audio, DEFAULT_LOCAL_MODEL
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_local.assert_called_once_with(sample_ogg, DEFAULT_LOCAL_MODEL)

    def test_command_provider_model_override_does_not_leak_into_local_fallback(
        self, sample_ogg
    ):
        config = {
            "provider": "parakeet",
            "providers": {
                "parakeet": {
                    "type": "command",
                    "command": "parakeet-transcribe {input}",
                    "fallback_provider": "local",
                }
            },
            "local": {"model": "small"},
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="parakeet"), \
             patch("tools.transcription_tools._transcribe_command_stt",
                   return_value={"success": False, "error": "missing binary"}), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "fallback text", "provider": "local"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg, model="parakeet-mlx-large")

        assert result["success"] is True
        mock_local.assert_called_once_with(sample_ogg, "small")

    def test_command_provider_returns_original_failure_without_fallback(self, sample_ogg):
        config = {
            "provider": "parakeet",
            "providers": {
                "parakeet": {
                    "type": "command",
                    "command": "parakeet-transcribe {input}",
                }
            },
        }
        expected = {"success": False, "error": "missing binary"}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="parakeet"), \
             patch("tools.transcription_tools._transcribe_command_stt", return_value=expected), \
             patch("tools.transcription_tools._transcribe_local") as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result == expected
        mock_local.assert_not_called()


# ============================================================================
# _transcribe_mistral
# ============================================================================


@pytest.fixture
def mock_mistral_module():
    """Inject a fake mistralai module into sys.modules for testing."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_mistral_cls = MagicMock(return_value=mock_client)
    fake_module = MagicMock()
    fake_module.Mistral = mock_mistral_cls
    with patch.dict("sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}):
        yield mock_client


class TestTranscribeMistral:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral("/tmp/test.ogg", "voxtral-mini-latest")
        assert result["success"] is False
        assert "MISTRAL_API_KEY" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        mock_result = MagicMock()
        mock_result.text = "hello from mistral"
        mock_mistral_module.audio.transcriptions.complete.return_value = mock_result

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is True
        assert result["transcript"] == "hello from mistral"
        assert result["provider"] == "mistral"
        mock_mistral_module.audio.transcriptions.complete.assert_called_once()
        mock_mistral_module.__exit__.assert_called_once()

    def test_api_error_returns_failure(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.transcriptions.complete.side_effect = RuntimeError("secret-key-leaked")

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is False
        assert "RuntimeError" in result["error"]
        assert "secret-key-leaked" not in result["error"]

    def test_permission_error(self, monkeypatch, sample_ogg, mock_mistral_module):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.transcriptions.complete.side_effect = PermissionError("denied")

        from tools.transcription_tools import _transcribe_mistral
        result = _transcribe_mistral(sample_ogg, "voxtral-mini-latest")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ============================================================================
# _get_provider — Mistral
# ============================================================================

class TestGetProviderMistral:
    """Mistral-specific provider selection tests."""

    def test_mistral_when_key_and_sdk_available(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "mistral"

    def test_mistral_explicit_no_key_returns_none(self, monkeypatch):
        """Explicit mistral with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "none"

    def test_mistral_explicit_no_sdk_returns_none(self, monkeypatch):
        """Explicit mistral with key but no SDK returns none."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "mistral"}) == "none"

    def test_auto_detect_mistral_after_openai(self, monkeypatch):
        """Auto-detect: mistral is tried after openai when both are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "mistral"

    def test_auto_detect_openai_preferred_over_mistral(self, monkeypatch):
        """Auto-detect: openai is preferred over mistral (both paid, openai more common)."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "openai"

    def test_auto_detect_groq_preferred_over_mistral(self, monkeypatch):
        """Auto-detect: groq (free) is preferred over mistral (paid)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "groq"

    def test_auto_detect_skips_mistral_without_sdk(self, monkeypatch):
        """Auto-detect: mistral skipped when key is set but SDK is not installed."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "none"


# ============================================================================
# transcribe_audio — Mistral dispatch
# ============================================================================

class TestTranscribeAudioMistralDispatch:
    def test_dispatches_to_mistral(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "mistral"}), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi", "provider": "mistral"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "mistral"
        mock_mistral.assert_called_once()

    def test_config_mistral_model_used(self, sample_ogg):
        config = {"provider": "mistral", "mistral": {"model": "voxtral-mini-2602"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_mistral.call_args[0][1] == "voxtral-mini-2602"

    def test_model_override_passed_to_mistral(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="voxtral-mini-2602")

        assert mock_mistral.call_args[0][1] == "voxtral-mini-2602"


# ============================================================================
# _transcribe_xai
# ============================================================================


@pytest.fixture
def mock_xai_http_module():
    """Inject a fake tools.xai_http module for testing."""
    fake_module = MagicMock()
    fake_module.hermes_xai_user_agent = MagicMock(return_value="hermes-xai/test")
    with patch.dict("sys.modules", {"tools.xai_http": fake_module}):
        yield fake_module


class TestTranscribeXAI:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_xai
        result = _transcribe_xai("/tmp/test.ogg", "grok-stt")
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "bonjour le monde",
            "language": "fr",
            "duration": 3.2,
        }

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is True
        assert result["transcript"] == "bonjour le monde"
        assert result["provider"] == "xai"

    def test_whitespace_stripped(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "  hello world  \n"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["transcript"] == "hello world"

    def test_api_error_returns_failure(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid audio format"}}
        mock_response.text = '{"error": {"message": "Invalid audio format"}}'

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "HTTP 400" in result["error"]
        assert "Invalid audio format" in result["error"]

    @pytest.mark.parametrize("rejected_status", [401, 403])
    def test_retries_auth_rejection_with_refreshed_oauth_credentials(
        self, sample_ogg, mock_xai_http_module, rejected_status
    ):
        mock_xai_http_module.resolve_xai_http_credentials.side_effect = [
            {
                "api_key": "stale-oauth-token",
                "base_url": "https://api.x.ai/v1",
                "provider": "xai-oauth",
            },
            {
                "api_key": "fresh-oauth-token",
                "base_url": "https://api.x.ai/v1",
                "provider": "xai-oauth",
            },
        ]

        rejected = MagicMock()
        rejected.status_code = rejected_status
        rejected.json.return_value = {
            "error": {"message": "OAuth2 access token could not be validated"}
        }
        accepted = MagicMock()
        accepted.status_code = 200
        accepted.json.return_value = {
            "text": "fleet speech transcription proof",
            "language": "en",
            "duration": 2.1,
        }

        stt_config = {"provider": "xai"}
        with patch("tools.transcription_tools._load_stt_config", return_value=stt_config), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("requests.post", side_effect=[rejected, accepted]) as mock_post:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result == {
            "success": True,
            "transcript": "fleet speech transcription proof",
            "provider": "xai",
        }
        assert mock_post.call_count == 2
        assert mock_post.call_args_list[0].kwargs["headers"]["Authorization"] == (
            "Bearer stale-oauth-token"
        )
        assert mock_post.call_args_list[1].kwargs["headers"]["Authorization"] == (
            "Bearer fresh-oauth-token"
        )
        assert mock_xai_http_module.resolve_xai_http_credentials.call_args_list == [
            call(),
            call(force_refresh=True, api_key_hint="stale-oauth-token"),
        ]

    def test_empty_transcript_returns_failure(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "   "}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "empty transcript" in result["error"]
        assert result["no_speech"] is True  # live voice loops treat this as silence

    def test_permission_error(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("builtins.open", side_effect=PermissionError("denied")):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    def test_network_error_returns_failure(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", side_effect=ConnectionError("timeout")):
            from tools.transcription_tools import _transcribe_xai
            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is False
        assert "timeout" in result["error"]

    def test_sends_language_and_format(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        # Explicitly set language via env to exercise the override chain
        # (config > env > DEFAULT_LOCAL_STT_LANGUAGE)
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "fr")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "fr", "duration": 1.0}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
        assert data.get("language") == "fr"
        assert data.get("format") == "true"

    def test_custom_base_url(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        monkeypatch.setenv("XAI_STT_BASE_URL", "https://custom.x.ai/v1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "en", "duration": 1.0}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "custom.x.ai" in url

    def test_oauth_credentials_ignore_stt_base_url_override(
        self,
        monkeypatch,
        sample_ogg,
        mock_xai_http_module,
    ):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("XAI_STT_BASE_URL", "https://attacker.example/v1")
        mock_xai_http_module.resolve_xai_http_credentials.return_value = {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "en", "duration": 1.0}

        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"xai": {"base_url": "https://attacker.example/config"}},
        ), patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai

            result = _transcribe_xai(sample_ogg, "grok-stt")

        assert result["success"] is True
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url == "https://api.x.ai/v1/stt"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer oauth-bearer-token"

    def test_diarize_sent_when_configured(self, monkeypatch, sample_ogg, mock_xai_http_module):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test", "language": "fr", "duration": 1.0}

        config = {"xai": {"diarize": True}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_xai
            _transcribe_xai(sample_ogg, "grok-stt")

        data = mock_post.call_args.kwargs.get("data", mock_post.call_args[1].get("data", {}))
        assert data.get("diarize") == "true"


# ============================================================================
# _get_provider — xAI
# ============================================================================

class TestGetProviderXAI:
    """xAI-specific provider selection tests."""

    def test_xai_when_key_set(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "xai"}) == "xai"

    def test_xai_explicit_no_key_returns_none(self, monkeypatch):
        """Explicit xai with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "xai"}) == "none"

    def test_auto_detect_xai_after_mistral(self, monkeypatch):
        """Auto-detect: xai is tried after mistral when all above are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "xai"

    def test_auto_detect_mistral_preferred_over_xai(self, monkeypatch):
        """Auto-detect: mistral is preferred over xai."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "mistral"

    def test_auto_detect_no_key_returns_none(self, monkeypatch):
        """Auto-detect: xai skipped when no key is set."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "none"


# ============================================================================
# transcribe_audio — xAI dispatch
# ============================================================================

class TestTranscribeAudioXAIDispatch:
    def test_dispatches_to_xai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "xai"}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi", "provider": "xai"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "xai"
        mock_xai.assert_called_once()

    def test_model_default_is_grok_stt(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "xai"}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_xai.call_args[0][1] == "grok-stt"

    def test_model_override_passed_to_xai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="custom-stt")

        assert mock_xai.call_args[0][1] == "custom-stt"


# ============================================================================
# _transcribe_elevenlabs
# ============================================================================

class TestTranscribeElevenLabs:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_elevenlabs
        result = _transcribe_elevenlabs("/tmp/test.ogg", "scribe_v2")
        assert result["success"] is False
        assert "ELEVENLABS_API_KEY" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_ogg):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "hello from elevenlabs"}

        config = {
            "elevenlabs": {
                "language_code": "eng",
                "tag_audio_events": True,
                "diarize": True,
            }
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("requests.post", return_value=mock_response) as mock_post:
            from tools.transcription_tools import _transcribe_elevenlabs
            result = _transcribe_elevenlabs(sample_ogg, "scribe_v2")

        assert result["success"] is True
        assert result["transcript"] == "hello from elevenlabs"
        assert result["provider"] == "elevenlabs"
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["xi-api-key"] == "eleven-test-key"
        assert call_kwargs["data"]["model_id"] == "scribe_v2"
        assert call_kwargs["data"]["language_code"] == "eng"
        assert call_kwargs["data"]["tag_audio_events"] == "true"
        assert call_kwargs["data"]["diarize"] == "true"

    def test_api_error_returns_failure(self, monkeypatch, sample_ogg):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": {"message": "Invalid API key"}}
        mock_response.text = '{"detail": {"message": "Invalid API key"}}'

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_elevenlabs
            result = _transcribe_elevenlabs(sample_ogg, "scribe_v2")

        assert result["success"] is False
        assert "HTTP 401" in result["error"]
        assert "Invalid API key" in result["error"]

    def test_empty_transcript_returns_failure(self, monkeypatch, sample_ogg):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "   "}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            from tools.transcription_tools import _transcribe_elevenlabs
            result = _transcribe_elevenlabs(sample_ogg, "scribe_v2")

        assert result["success"] is False
        assert "empty transcript" in result["error"]
        assert result["no_speech"] is True  # live voice loops treat this as silence


# ============================================================================
# _get_provider — ElevenLabs
# ============================================================================

class TestGetProviderElevenLabs:
    """ElevenLabs-specific provider selection tests."""

    def test_elevenlabs_when_key_set(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test")
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "elevenlabs"}) == "elevenlabs"

    def test_elevenlabs_explicit_no_key_returns_none(self, monkeypatch):
        """Explicit elevenlabs with no key returns none — no cross-provider fallback."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "elevenlabs"}) == "none"

    def test_auto_detect_elevenlabs_after_xai(self, monkeypatch):
        """Auto-detect: elevenlabs is tried after xai when all above are unavailable."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "elevenlabs"

    def test_auto_detect_xai_preferred_over_elevenlabs(self, monkeypatch):
        """Auto-detect: xai is preferred over elevenlabs."""
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._HAS_MISTRAL", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "xai"


# ============================================================================
# transcribe_audio — ElevenLabs dispatch
# ============================================================================

class TestTranscribeAudioElevenLabsDispatch:
    def test_dispatches_to_elevenlabs(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "elevenlabs"}), \
             patch("tools.transcription_tools._get_provider", return_value="elevenlabs"), \
             patch("tools.transcription_tools._transcribe_elevenlabs",
                   return_value={"success": True, "transcript": "hi", "provider": "elevenlabs"}) as mock_elevenlabs:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "elevenlabs"
        mock_elevenlabs.assert_called_once()

    def test_config_elevenlabs_model_used(self, sample_ogg):
        config = {"provider": "elevenlabs", "elevenlabs": {"model_id": "scribe_v1"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="elevenlabs"), \
             patch("tools.transcription_tools._transcribe_elevenlabs",
                   return_value={"success": True, "transcript": "hi"}) as mock_elevenlabs:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_elevenlabs.call_args[0][1] == "scribe_v1"

    def test_model_override_passed_to_elevenlabs(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="elevenlabs"), \
             patch("tools.transcription_tools._transcribe_elevenlabs",
                   return_value={"success": True, "transcript": "hi"}) as mock_elevenlabs:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="scribe_v2")

        assert mock_elevenlabs.call_args[0][1] == "scribe_v2"


# ============================================================================
# _extract_transcript_text
# ============================================================================

class TestExtractTranscriptText:
    def test_strips_qwen3_asr_language_envelope(self):
        from tools.transcription_tools import _extract_transcript_text

        result = _extract_transcript_text(
            "language zh\n<audio_language>zh</audio_language>\n<asr_text>你好，世界",
        )

        assert result == "你好，世界"

    def test_keeps_non_envelope_marker_literal(self):
        from tools.transcription_tools import _extract_transcript_text

        result = _extract_transcript_text(
            "The user literally said <asr_text> while reading markup.",
        )

        assert result == "The user literally said <asr_text> while reading markup."

    def test_keeps_language_sentence_with_marker_literal(self):
        from tools.transcription_tools import _extract_transcript_text

        result = _extract_transcript_text(
            "Language teachers may say <asr_text> when discussing markup.",
        )

        assert result == "Language teachers may say <asr_text> when discussing markup."


# Shell safety — shlex.split on auto-detected templates
# ============================================================================
class TestShellSafety:
    def test_auto_detected_template_is_shlex_safe(self, monkeypatch):
        """Auto-detected whisper command should be safely splittable."""
        import shlex
        monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
        monkeypatch.setattr(
            "tools.transcription_tools._find_whisper_binary",
            lambda: "/usr/bin/whisper",
        )
        from tools.transcription_tools import _get_local_command_template
        template = _get_local_command_template()
        assert template is not None
        cmd = template.format(
            input_path=shlex.quote("/tmp/test.wav"),
            output_dir=shlex.quote("/tmp/out"),
            language=shlex.quote("en"),
            model=shlex.quote("base"),
        )
        parts = shlex.split(cmd)
        assert parts[0] == "/usr/bin/whisper"
        assert "/tmp/test.wav" in parts

    def test_env_var_template_metacharacters_are_literal_argv(
        self, monkeypatch, sample_wav, tmp_path
    ):
        from tools.transcription_tools import (
            LOCAL_STT_COMMAND_ENV,
            _transcribe_local_command,
            windows_hide_flags,
        )

        output_dir = tmp_path / "transcript-output"
        output_dir.mkdir()
        monkeypatch.setenv(
            LOCAL_STT_COMMAND_ENV,
            (
                "whisper {input_path} ; printf injected | tee log.txt "
                "&& echo $(id) `whoami` --output_dir {output_dir}"
            ),
        )

        def fake_tempdir(prefix=None):
            class _TempDir:
                def __enter__(self):
                    return str(output_dir)

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _TempDir()

        invocation = {}

        def fake_run(command, **kwargs):
            invocation["command"] = command
            invocation["kwargs"] = kwargs
            (output_dir / "transcript.txt").write_text("safe", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(
            "tools.transcription_tools.tempfile.TemporaryDirectory", fake_tempdir
        )
        monkeypatch.setattr("tools.transcription_tools.subprocess.run", fake_run)

        result = _transcribe_local_command(sample_wav, "base")

        assert result["transcript"] == "safe"
        assert invocation["command"] == [
            "whisper",
            sample_wav,
            ";",
            "printf",
            "injected",
            "|",
            "tee",
            "log.txt",
            "&&",
            "echo",
            "$(id)",
            "`whoami`",
            "--output_dir",
            str(output_dir),
        ]
        assert invocation["kwargs"].pop("env") is not None
        assert invocation["kwargs"] == {
            "check": True,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 300,
            "stdin": subprocess.DEVNULL,
            "creationflags": windows_hide_flags(),
        }

    def test_no_env_var_uses_list_mode(self, monkeypatch):
        """When no env var is set, use_shell should be False."""
        import os
        from tools.transcription_tools import LOCAL_STT_COMMAND_ENV
        monkeypatch.delenv(LOCAL_STT_COMMAND_ENV, raising=False)
        use_shell = bool(os.getenv(LOCAL_STT_COMMAND_ENV, "").strip())
        assert use_shell is False


class TestLocalModelLock:
    """#24767 — concurrent first-use must not double-load the whisper model."""

    def test_lock_exists_and_is_a_lock(self):
        import threading
        from tools import transcription_tools
        assert isinstance(transcription_tools._local_model_lock, type(threading.Lock()))

    def test_concurrent_transcribe_loads_model_once(self, tmp_path):
        import threading
        from tools import transcription_tools
        from tools.transcription_tools import _transcribe_local

        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg = MagicMock()
        seg.text = "hello"
        info = MagicMock()
        info.language = "en"
        info.duration = 1.0

        load_count = 0
        load_started = threading.Event()

        def slow_load(model_name, device="auto", compute_type="auto"):
            nonlocal load_count
            load_count += 1
            load_started.set()
            import time
            time.sleep(0.05)
            model = MagicMock()
            model.transcribe.return_value = ([seg], info)
            return model

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._load_local_whisper_model", side_effect=slow_load), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            threads = [
                threading.Thread(target=_transcribe_local, args=(str(audio), "base"))
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert load_count == 1


class TestLocalBaseUrlNoApiKey:
    """#25193 — empty api_key with a local base_url should not raise."""

    def test_local_base_url_returns_placeholder_key(self):
        from tools.transcription_tools import _resolve_openai_audio_client_config
        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"openai": {"base_url": "http://localhost:8504/v1"}},
        ):
            api_key, base_url = _resolve_openai_audio_client_config()
        assert api_key == "not-needed"
        assert base_url == "http://localhost:8504/v1"

    def test_private_ip_base_url_returns_placeholder_key(self):
        from tools.transcription_tools import _resolve_openai_audio_client_config
        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"openai": {"base_url": "http://192.168.1.10:8000/v1"}},
        ):
            api_key, base_url = _resolve_openai_audio_client_config()
        assert api_key == "not-needed"

    def test_public_base_url_still_requires_key(self):
        from tools.transcription_tools import _resolve_openai_audio_client_config
        with patch(
            "tools.transcription_tools._load_stt_config",
            return_value={"openai": {"base_url": "https://api.example.com/v1"}},
        ), patch(
            "tools.transcription_tools.resolve_openai_audio_api_key", return_value="",
        ), patch(
            "tools.transcription_tools.resolve_managed_tool_gateway", return_value=None,
        ), patch(
            "tools.transcription_tools.managed_nous_tools_enabled", return_value=False,
        ):
            with pytest.raises(ValueError):
                _resolve_openai_audio_client_config()

    def test_is_local_or_private_url(self):
        from tools.transcription_tools import _is_local_or_private_url
        assert _is_local_or_private_url("http://localhost:8504/v1")
        assert _is_local_or_private_url("http://127.0.0.1:9000")
        assert _is_local_or_private_url("http://10.0.0.5/v1")
        assert _is_local_or_private_url("http://stt.internal/v1")
        assert not _is_local_or_private_url("https://api.openai.com/v1")
        assert not _is_local_or_private_url("")


# =====================================================================


# CAF (iMessage voice note) conversion tests
# ============================================================================

class TestCafConversion:
    """Tests for _convert_caf_to_wav and CAF dispatch in transcribe_audio."""

    def test_convert_caf_with_ffmpeg(self, tmp_path, monkeypatch):
        """_convert_caf_to_wav uses ffmpeg when available."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)
        wav_path = str(tmp_path / "voice.wav")

        def fake_run(cmd, **kwargs):
            Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00")
            return MagicMock(returncode=0)

        monkeypatch.setattr(
            "tools.transcription_tools._find_ffmpeg_binary",
            lambda: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        from tools.transcription_tools import _convert_caf_to_wav
        result = _convert_caf_to_wav(str(caf_path))
        assert result == wav_path
        assert Path(result).exists()

    def test_convert_caf_fallback_to_afconvert(self, tmp_path, monkeypatch):
        """When ffmpeg is not found, falls back to afconvert (macOS)."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)
        wav_path = str(tmp_path / "voice.wav")

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if cmd[0] == "/usr/bin/ffmpeg":
                raise subprocess.CalledProcessError(1, cmd)
            Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00")
            return MagicMock(returncode=0)

        monkeypatch.setattr(
            "tools.transcription_tools._find_ffmpeg_binary",
            lambda: "/usr/bin/ffmpeg",
        )
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(
            "tools.transcription_tools.shutil.which",
            lambda name: "/usr/bin/afconvert" if name == "afconvert" else None,
        )

        from tools.transcription_tools import _convert_caf_to_wav
        result = _convert_caf_to_wav(str(caf_path))
        assert result == wav_path
        assert call_count["n"] == 2

    def test_convert_caf_all_converters_fail(self, tmp_path, monkeypatch):
        """When both ffmpeg and afconvert are unavailable, returns None."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)

        monkeypatch.setattr(
            "tools.transcription_tools._find_ffmpeg_binary", lambda: None
        )
        monkeypatch.setattr(
            "tools.transcription_tools.shutil.which", lambda name: None
        )

        from tools.transcription_tools import _convert_caf_to_wav
        result = _convert_caf_to_wav(str(caf_path))
        assert result is None

    def test_transcribe_caf_converted_before_groq(self, tmp_path, monkeypatch):
        """transcribe_audio converts .caf to .wav before dispatching to Groq."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)
        wav_path = str(tmp_path / "voice.wav")

        def fake_convert(file_path):
            Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00")
            return wav_path

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "groq"}), \
             patch("tools.transcription_tools._get_provider",
                   return_value="groq"), \
             patch("tools.transcription_tools._convert_caf_to_wav",
                   side_effect=fake_convert) as mock_convert, \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hello",
                                 "provider": "groq"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(caf_path))

        assert result["success"] is True
        mock_convert.assert_called_once_with(str(caf_path))
        mock_groq.assert_called_once()
        call_args = mock_groq.call_args
        sent_path = call_args[0][0] if call_args[0] else call_args[1].get("file_path")
        assert sent_path == wav_path

    def test_transcribe_caf_conversion_failure_returns_error(
        self, tmp_path, monkeypatch
    ):
        """When CAF conversion fails, transcribe_audio returns an error."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "groq"}), \
             patch("tools.transcription_tools._get_provider",
                   return_value="groq"), \
             patch("tools.transcription_tools._convert_caf_to_wav",
                   return_value=None):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(caf_path))

        assert result["success"] is False
        assert "could not be converted" in result["error"]

    def test_transcribe_caf_not_converted_for_local(self, tmp_path, monkeypatch):
        """CAF conversion is skipped for local provider (native handling)."""
        caf_path = tmp_path / "voice.caf"
        caf_path.write_bytes(b"caff\x00" * 20)

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider",
                   return_value="local"), \
             patch("tools.transcription_tools._convert_caf_to_wav") as mock_convert, \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(caf_path))

        assert result["success"] is True
        mock_convert.assert_not_called()


class TestTranscribeCredentialReadGuard:
    """transcribe_audio must refuse credential/secret stores before dispatch."""

    def test_transcribe_audio_blocks_credential_read(self, tmp_path):
        """A ``.env`` (secret-bearing) file is refused up front, so its
        plaintext is never shipped to an external STT provider — mirroring the
        read guard added to image-gen (587be5b5b) and xAI video-gen
        (104232979)."""
        from tools.transcription_tools import transcribe_audio
        from agent.file_safety import get_read_block_error

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-secret\n")

        expected = get_read_block_error(str(env_file))
        assert expected, "test setup: a .env file should be read-blocked"

        result = transcribe_audio(str(env_file))

        assert result["success"] is False
        # The error is the shared read-guard message, not an audio-validation
        # or provider error — proving the guard fired before dispatch.
        assert result["error"] == expected


class TestRunCommandSttIdleTimeout:
    """_run_command_stt uses a progress-based idle timeout (mirrors TTS runner)."""

    @staticmethod
    def _shell_command(*args):
        import shlex
        if os.name == "nt":
            return subprocess.list2cmdline(list(args))
        return " ".join(shlex.quote(str(arg)) for arg in args)

    def test_stderr_progress_extends_beyond_timeout(self, tmp_path):
        """A slow-but-alive command that keeps emitting output survives an
        idle timeout shorter than its total runtime."""
        from tools.transcription_tools import _run_command_stt

        script = tmp_path / "progress_then_exit.py"
        script.write_text(
            "\n".join([
                "import sys, time",
                "for idx in range(4):",
                "    print(f'tick {idx}', file=sys.stderr, flush=True)",
                "    time.sleep(0.15)",
                "print('done', flush=True)",
            ]),
            encoding="utf-8",
        )

        result = _run_command_stt(
            self._shell_command(sys.executable, "-u", str(script)),
            timeout=0.25,
        )

        assert result.returncode == 0
        assert "tick 3" in result.stderr
        assert "done" in result.stdout

    def test_silent_stall_still_times_out(self, tmp_path):
        """A silently stalled command is killed once the idle window elapses,
        and pre-stall output is preserved on the TimeoutExpired."""
        from tools.transcription_tools import _run_command_stt

        script = tmp_path / "progress_then_hang.py"
        script.write_text(
            "\n".join([
                "import sys, time",
                "print('starting pass 1', file=sys.stderr, flush=True)",
                "time.sleep(1.0)",
            ]),
            encoding="utf-8",
        )

        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _run_command_stt(
                self._shell_command(sys.executable, "-u", str(script)),
                timeout=0.2,
            )

        assert "starting pass 1" in (excinfo.value.stderr or "")
