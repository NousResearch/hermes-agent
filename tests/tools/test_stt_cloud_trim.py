"""Tests for the cloud STT pre-upload silence trim.

Local faster-whisper gets Silero VAD (``build_local_transcribe_kwargs``);
cloud providers upload the raw file. ``_trim_silence_for_cloud_stt``
closes that gap: it collapses long pauses with ffmpeg before upload so
silence isn't uploaded, billed per audio-minute, or hallucinated on.

Contract under test:

1. Trim runs only for built-in CLOUD providers — never local/local_command,
   never command-type or plugin providers.
2. Best-effort semantics: disabled config, missing ffmpeg/ffprobe, trim
   failure, mostly-silence result, or <10% saving all mean "upload the
   original untouched" (return None) — the transcription NEVER fails
   because of the trim.
3. The dispatcher passes the trimmed file to the provider and cleans up
   the temp dir afterwards.
4. E2E (real ffmpeg): a WAV with long silent stretches gets measurably
   shorter; a fully-silent WAV falls back to the original.
"""

import shutil
import struct
import sys
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if "faster_whisper" not in sys.modules:
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock(name="WhisperModel")
    from importlib.machinery import ModuleSpec
    faster_whisper_stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = faster_whisper_stub

from tools.transcription_tools import (
    CLOUD_STT_PROVIDERS,
    BUILTIN_STT_PROVIDERS,
    _cloud_trim_settings,
    _CLOUD_TRIM_KEEP_MS_DEFAULT,
    _CLOUD_TRIM_THRESHOLD_DB_DEFAULT,
    _trim_silence_for_cloud_stt,
)
import tools.transcription_tools as tt_module

# The E2E fixtures below must be past the short-clip input gate.
_GATE = tt_module._CLOUD_TRIM_MIN_INPUT_SECONDS

_HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


# ============================================================================
# Helpers
# ============================================================================


def _write_wav(path: Path, segments) -> str:
    """Write a 16 kHz mono WAV from (kind, seconds) segments.

    kind is "tone" (audible square-ish wave) or "silence".
    """
    rate = 16000
    frames = bytearray()
    for kind, seconds in segments:
        n = int(rate * seconds)
        if kind == "tone":
            # 400 Hz square wave at strong amplitude — unambiguous speech-band energy.
            samples = [12000 if (i // 20) % 2 == 0 else -12000 for i in range(n)]
        else:
            samples = [0] * n
        frames.extend(struct.pack(f"<{n}h", *samples))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return str(path)


# ============================================================================
# Provider gating
# ============================================================================


class TestProviderGating:
    def test_cloud_set_excludes_local_providers(self):
        assert "local" not in CLOUD_STT_PROVIDERS
        assert "local_command" not in CLOUD_STT_PROVIDERS

    def test_cloud_set_covers_every_remote_builtin(self):
        # Invariant: every built-in that is not local-ish uploads audio and
        # must get the trim. New built-ins are cloud unless proven otherwise.
        assert CLOUD_STT_PROVIDERS == BUILTIN_STT_PROVIDERS - {"local", "local_command"}

    def test_local_provider_never_trims(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "local", "enabled": True}), \
             patch("tools.transcription_tools._trim_silence_for_cloud_stt") as trim, \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "ok"}):
            from tools.transcription_tools import _transcribe_prepared_audio
            result = _transcribe_prepared_audio(wav)
        assert result["success"] is True
        trim.assert_not_called()

    def test_cloud_provider_trims_and_forwards_trimmed_path(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        trimmed_dir = tmp_path / "trim-work"
        trimmed_dir.mkdir()
        trimmed = _write_wav(trimmed_dir / "a-trimmed.wav", [("tone", 1)])
        seen = {}

        def fake_groq(file_path, model_name, *, language=None, prompt=None):
            seen["path"] = file_path
            return {"success": True, "transcript": "hi", "provider": "groq"}

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "groq", "enabled": True}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._trim_silence_for_cloud_stt",
                   return_value=trimmed), \
             patch("tools.transcription_tools._transcribe_groq", side_effect=fake_groq):
            from tools.transcription_tools import _transcribe_prepared_audio
            result = _transcribe_prepared_audio(wav)

        assert result["success"] is True
        assert seen["path"] == trimmed
        # Dispatcher owns the cleanup of the trim temp dir.
        assert not trimmed_dir.exists()

    def test_trim_returning_none_uploads_original(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        seen = {}

        def fake_groq(file_path, model_name, *, language=None, prompt=None):
            seen["path"] = file_path
            return {"success": True, "transcript": "hi", "provider": "groq"}

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"provider": "groq", "enabled": True}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._trim_silence_for_cloud_stt",
                   return_value=None), \
             patch("tools.transcription_tools._transcribe_groq", side_effect=fake_groq):
            from tools.transcription_tools import _transcribe_prepared_audio
            result = _transcribe_prepared_audio(wav)

        assert result["success"] is True
        assert seen["path"] == wav

    def test_command_provider_never_trims(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        cfg = {
            "provider": "mywhisper",
            "enabled": True,
            "providers": {"mywhisper": {"type": "command", "command": "true"}},
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools._trim_silence_for_cloud_stt") as trim, \
             patch("tools.transcription_tools._transcribe_command_stt",
                   return_value={"success": True, "transcript": "ok"}):
            from tools.transcription_tools import _transcribe_prepared_audio
            _transcribe_prepared_audio(wav)
        trim.assert_not_called()


# ============================================================================
# Settings resolution
# ============================================================================


class TestCloudTrimSettings:
    def test_defaults(self):
        enabled, threshold, keep = _cloud_trim_settings({})
        assert enabled is True
        assert threshold == _CLOUD_TRIM_THRESHOLD_DB_DEFAULT
        assert keep == _CLOUD_TRIM_KEEP_MS_DEFAULT

    def test_disable(self):
        enabled, _, _ = _cloud_trim_settings({"cloud_trim_silence": False})
        assert enabled is False

    def test_yaml_string_false_disables(self):
        # Config strings must be normalized like every other stt boolean
        # (is_truthy_value) — "false" from YAML/env must not mean enabled.
        enabled, _, _ = _cloud_trim_settings({"cloud_trim_silence": "false"})
        assert enabled is False

    def test_none_means_default_on(self):
        enabled, _, _ = _cloud_trim_settings({"cloud_trim_silence": None})
        assert enabled is True

    def test_custom_values(self):
        enabled, threshold, keep = _cloud_trim_settings(
            {"cloud_trim_threshold_db": -30, "cloud_trim_keep_ms": 500}
        )
        assert enabled is True
        assert threshold == -30
        assert keep == 500

    def test_garbage_falls_back(self):
        _, threshold, keep = _cloud_trim_settings(
            {"cloud_trim_threshold_db": "loud", "cloud_trim_keep_ms": None}
        )
        assert threshold == _CLOUD_TRIM_THRESHOLD_DB_DEFAULT
        assert keep == _CLOUD_TRIM_KEEP_MS_DEFAULT

    def test_negative_keep_clamped(self):
        _, _, keep = _cloud_trim_settings({"cloud_trim_keep_ms": -100})
        assert keep == 0

    def test_non_dict_config(self):
        enabled, threshold, keep = _cloud_trim_settings(None)
        assert enabled is True
        assert threshold == _CLOUD_TRIM_THRESHOLD_DB_DEFAULT


# ============================================================================
# Best-effort fallbacks (all must return None, never raise)
# ============================================================================


class TestTrimFallbacks:
    def test_disabled_returns_none(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        assert _trim_silence_for_cloud_stt(wav, {"cloud_trim_silence": False}) is None

    def test_missing_ffmpeg_returns_none(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value=None):
            assert _trim_silence_for_cloud_stt(wav, {}) is None

    def test_missing_ffprobe_returns_none(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value="/bin/ffmpeg"), \
             patch("tools.transcription_tools._find_ffprobe_binary", return_value=None):
            assert _trim_silence_for_cloud_stt(wav, {}) is None

    def test_ffmpeg_failure_returns_none_and_cleans_up(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        import subprocess as sp

        def probe(path):
            return 60.0  # past the short-clip gate so the encode is attempted

        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value="/bin/ffmpeg"), \
             patch("tools.transcription_tools._probe_audio_duration", side_effect=probe), \
             patch("tools.transcription_tools.subprocess.run",
                   side_effect=sp.CalledProcessError(1, "ffmpeg")):
            assert _trim_silence_for_cloud_stt(wav, {}) is None

    def test_unprobeable_source_returns_none(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1)])
        with patch("tools.transcription_tools._find_ffmpeg_binary", return_value="/bin/ffmpeg"), \
             patch("tools.transcription_tools._probe_audio_duration", return_value=None):
            assert _trim_silence_for_cloud_stt(wav, {}) is None


# ============================================================================
# E2E with real ffmpeg
# ============================================================================


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestTrimE2E:
    def test_long_pauses_are_collapsed(self, tmp_path):
        # 2s speech + 6s silence + 2s speech + 4s trailing silence = 14s,
        # ~10s of it silence. The trim must save well over 10%.
        wav = _write_wav(
            tmp_path / "pauses.wav",
            [("tone", 2), ("silence", 6), ("tone", 2), ("silence", 4)],
        )
        from tools.transcription_tools import _probe_audio_duration
        trimmed = _trim_silence_for_cloud_stt(wav, {})
        assert trimmed is not None
        try:
            original = _probe_audio_duration(wav)
            result = _probe_audio_duration(trimmed)
            assert result is not None and original is not None
            assert result < original * 0.6  # >40% shorter
            assert result > 3.5  # both speech chunks survived
        finally:
            shutil.rmtree(Path(trimmed).parent, ignore_errors=True)

    def test_dense_speech_untouched(self, tmp_path):
        # Continuous tone (past the short-clip gate) — nothing to trim,
        # saving <10% → return None.
        wav = _write_wav(tmp_path / "dense.wav", [("tone", 14)])
        assert _trim_silence_for_cloud_stt(wav, {}) is None

    def test_all_silence_falls_back_to_original(self, tmp_path):
        # Pure silence (past the short-clip gate) collapses to ~nothing; the
        # provider must decide "no speech", not a client-side dB heuristic
        # → return None.
        wav = _write_wav(tmp_path / "silence.wav", [("silence", 14)])
        assert _trim_silence_for_cloud_stt(wav, {}) is None

    def test_short_clip_skips_trim_entirely(self, tmp_path):
        # Below the input-duration gate the encode pipeline must not run at
        # all — savings can't matter on short clips and several providers
        # bill a per-request minimum anyway.
        wav = _write_wav(
            tmp_path / "short.wav", [("tone", 2), ("silence", 4), ("tone", 2)]
        )
        with patch.object(tt_module, "_run_ffmpeg_stt_encode") as mock_encode:
            assert _trim_silence_for_cloud_stt(wav, {}) is None
        mock_encode.assert_not_called()

    def test_disabled_config_uploads_original(self, tmp_path):
        wav = _write_wav(
            tmp_path / "pauses.wav", [("tone", 2), ("silence", 6), ("tone", 2)]
        )
        assert _trim_silence_for_cloud_stt(wav, {"cloud_trim_silence": False}) is None
