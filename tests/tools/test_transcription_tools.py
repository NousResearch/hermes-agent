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

class TestGetProviderFallbackPriority:
    """Auto-detect fallback priority and explicit provider behaviour."""

    def test_auto_detect_prefers_local(self):
        """Auto-detect prefers local over any cloud provider."""
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"

    def test_unknown_provider_passed_through(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "custom-endpoint"}) == "custom-endpoint"

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
             patch("tools.transcription_tools._load_stt_config", return_value=fake_config), \
             patch("tools.transcription_tools._should_force_faster_whisper_cpu", return_value=True):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        mock_whisper_cls.assert_called_once_with("base", device="cpu", compute_type="float32")

    def test_apple_silicon_auto_config_still_forces_cpu_int8(self):
        """Apple Silicon safety still overrides unsafe device autodetection."""
        mock_whisper_cls = MagicMock()

        with patch("faster_whisper.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._should_force_faster_whisper_cpu", return_value=True):
            from tools.transcription_tools import _load_local_whisper_model
            _load_local_whisper_model("base", device="auto", compute_type="auto")

        mock_whisper_cls.assert_called_once_with("base", device="cpu", compute_type="int8")


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


    def test_all_supported_formats_accepted(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file, SUPPORTED_FORMATS
        for fmt in SUPPORTED_FORMATS:
            f = tmp_path / f"test{fmt}"
            f.write_bytes(b"data")
            assert _validate_audio_file(str(f)) is None, f"Format {fmt} should be accepted"

# ============================================================================
# transcribe_audio — end-to-end dispatch
# ============================================================================

class TestTranscribeAudioDispatch:
    def test_oversized_local_file_reaches_dispatcher(self, oversized_wav):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(oversized_wav)

        assert result["success"] is True
        mock_local.assert_called_once()


    def test_no_provider_returns_error(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]
        assert "faster-whisper" in result["error"]
        assert "GROQ_API_KEY" in result["error"]


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


    def test_config_local_model_used(self, sample_ogg):
        config = {"local": {"model": "small"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_local.call_args[0][1] == "small"

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

# ============================================================================
# _get_provider — Mistral
# ============================================================================

class TestGetProviderMistral:
    """Mistral-specific provider selection tests."""

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

# ============================================================================
# transcribe_audio — Mistral dispatch
# ============================================================================

class TestTranscribeAudioMistralDispatch:
    def test_config_mistral_model_used(self, sample_ogg):
        config = {"provider": "mistral", "mistral": {"model": "voxtral-mini-2602"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="mistral"), \
             patch("tools.transcription_tools._transcribe_mistral",
                   return_value={"success": True, "transcript": "hi"}) as mock_mistral:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

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


    @pytest.mark.parametrize("rejected_status", [401])
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

# ============================================================================
# _get_provider — xAI
# ============================================================================

class TestGetProviderXAI:
    """xAI-specific provider selection tests."""

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

# ============================================================================
# transcribe_audio — xAI dispatch
# ============================================================================

class TestTranscribeAudioXAIDispatch:
    def test_model_default_is_grok_stt(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "xai"}), \
             patch("tools.transcription_tools._get_provider", return_value="xai"), \
             patch("tools.transcription_tools._transcribe_xai",
                   return_value={"success": True, "transcript": "hi"}) as mock_xai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_xai.call_args[0][1] == "grok-stt"

# ============================================================================
# _transcribe_elevenlabs
# ============================================================================

class TestTranscribeElevenLabs:
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

# ============================================================================
# _get_provider — ElevenLabs
# ============================================================================

class TestGetProviderElevenLabs:
    """ElevenLabs-specific provider selection tests."""

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

# ============================================================================
# transcribe_audio — ElevenLabs dispatch
# ============================================================================

class TestTranscribeAudioElevenLabsDispatch:
    def test_config_elevenlabs_model_used(self, sample_ogg):
        config = {"provider": "elevenlabs", "elevenlabs": {"model_id": "scribe_v1"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="elevenlabs"), \
             patch("tools.transcription_tools._transcribe_elevenlabs",
                   return_value={"success": True, "transcript": "hi"}) as mock_elevenlabs:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_elevenlabs.call_args[0][1] == "scribe_v1"

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


class TestLocalModelLock:
    """#24767 — concurrent first-use must not double-load the whisper model."""

    def test_concurrent_transcribe_loads_model_once(self, tmp_path):
        import threading
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
                "    time.sleep(0.04)",
                "print('done', flush=True)",
            ]),
            encoding="utf-8",
        )

        result = _run_command_stt(
            self._shell_command(sys.executable, "-u", str(script)),
            timeout=0.1,
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
                "time.sleep(30)",
            ]),
            encoding="utf-8",
        )

        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _run_command_stt(
                self._shell_command(sys.executable, "-u", str(script)),
                timeout=0.1,
            )

        assert "starting pass 1" in (excinfo.value.stderr or "")
