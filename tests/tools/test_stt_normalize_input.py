"""Tests for the opt-in pre-STT input gain normalization (``stt.normalize_input``).

A phone voice note routinely carries a 25-35 dB crest factor: the speech sits 30-40 dB
below one handling click. Whisper levels against that peak, so the speech is effectively
inaudible and the decoder emits an outro it learned from YouTube transcripts
("Merci d'avoir regardé cette vidéo", "Sous-titrage ST' 501"). ``_normalize_stt_input``
closes that gap: compress the crest factor, then level to -16 LUFS, before any backend
runs.

Contract under test:

1. Off unless ``stt.normalize_input`` says otherwise (bool, int or string spelling).
2. Applies to every backend — it runs in ``transcribe_audio`` before the provider
   dispatch, and therefore before the cloud silence trim, which would otherwise cut the
   quiet speech it is meant to rescue.
3. Best-effort: missing ffmpeg, a failing pass, or an unreadable input all mean
   "transcribe the original untouched" — normalization NEVER fails a transcription.
4. E2E (real ffmpeg): a WAV whose speech sits far below a loud transient comes out with
   its dynamic range compressed — the quiet segment's RMS rises while the peak holds.
5. The dispatcher hands the normalized file to the provider and removes the temp dir.
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

from tools.transcription_audio import (
    _STT_NORMALIZE_FILTER,
    _normalize_stt_input,
    _normalize_stt_input_enabled,
)

_HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))
_requires_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


# ============================================================================
# Helpers
# ============================================================================

def _write_crest_wav(path: Path, *, quiet: int = 200, peak: int = 30000,
                     speech_seconds: float = 3.0, rate: int = 16000) -> str:
    """A WAV whose speech sits far below a single loud transient — the failing input class.

    ``quiet`` is the speech amplitude, ``peak`` the transient. Default ratio is ~43 dB of
    crest, which is what a far-from-mouth phone capture looks like.
    """
    frames = bytearray()
    n = int(rate * speech_seconds)
    # 400 Hz square wave at the quiet amplitude: unambiguous speech-band energy.
    frames.extend(struct.pack(f"<{n}h", *[quiet if (i // 20) % 2 == 0 else -quiet for i in range(n)]))
    # 20 ms transient at the peak amplitude (a handling click).
    click = int(rate * 0.02)
    frames.extend(struct.pack(f"<{click}h", *[peak if i % 2 == 0 else -peak for i in range(click)]))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return str(path)


def _wav_metrics(path: str) -> dict:
    """(rate, channels, peak, rms_of_first_80_percent) — the speech part, click excluded."""
    with wave.open(path, "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    assert width == 2, f"expected 16-bit PCM, got {width * 8}-bit"
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    peak = max(abs(s) for s in samples) if samples else 0
    speech = samples[: int(len(samples) * 0.8)]
    rms = (sum(s * s for s in speech) / max(len(speech), 1)) ** 0.5
    return {"rate": rate, "channels": channels, "peak": peak, "rms": rms}


# ============================================================================
# Config gating
# ============================================================================

class TestConfigGate:

    def test_off_when_unset(self):
        assert _normalize_stt_input_enabled({}) is False

    def test_off_when_false(self):
        assert _normalize_stt_input_enabled({"normalize_input": False}) is False

    @pytest.mark.parametrize("value", [True, 1, "true", "TRUE", "yes", "on", "enabled", " 1 "])
    def test_truthy_spellings(self, value):
        assert _normalize_stt_input_enabled({"normalize_input": value}) is True

    @pytest.mark.parametrize("value", [False, 0, "false", "no", "off", "", None, [], {}])
    def test_falsey_spellings(self, value):
        assert _normalize_stt_input_enabled({"normalize_input": value}) is False

    def test_non_mapping_config_does_not_crash(self):
        assert _normalize_stt_input_enabled(None) is False
        assert _normalize_stt_input_enabled("nonsense") is False

    def test_disabled_never_invokes_ffmpeg(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        with patch("tools.transcription_audio._run_quiet") as run:
            path, work_dir = _normalize_stt_input(wav, {})
        assert (path, work_dir) == (None, None)
        run.assert_not_called()


# ============================================================================
# Best-effort semantics
# ============================================================================

class TestBestEffort:

    def test_missing_ffmpeg_keeps_the_original(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        with patch("tools.transcription_audio._find_ffmpeg_binary", return_value=None):
            path, work_dir = _normalize_stt_input(wav, {"normalize_input": True})
        assert (path, work_dir) == (None, None)

    def test_failing_pass_keeps_the_original_and_cleans_up(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        with patch("tools.transcription_audio._run_quiet",
                   side_effect=RuntimeError("ffmpeg exploded")) as run:
            path, work_dir = _normalize_stt_input(wav, {"normalize_input": True})
        assert (path, work_dir) == (None, None)
        run.assert_called_once()

    def test_empty_output_is_a_failure(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")

        def fake_run(argv, timeout=None):
            Path(argv[-1]).write_bytes(b"")  # ffmpeg "succeeds" but writes nothing

        with patch("tools.transcription_audio._run_quiet", side_effect=fake_run):
            path, work_dir = _normalize_stt_input(wav, {"normalize_input": True})
        assert (path, work_dir) == (None, None)


# ============================================================================
# E2E: the dynamic range actually gets compressed
# ============================================================================

@_requires_ffmpeg
class TestNormalizationE2E:

    def test_quiet_speech_is_lifted_relative_to_the_peak(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        before = _wav_metrics(wav)

        path, work_dir = _normalize_stt_input(wav, {"normalize_input": True})
        try:
            assert path is not None and Path(path).is_file()
            after = _wav_metrics(path)
        finally:
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

        # Delivered as 16 kHz mono PCM so every provider and local backend accepts it.
        assert after["rate"] == 16000
        assert after["channels"] == 1
        # The point of the pass: the speech is no longer ~43 dB under the transient.
        crest_before = before["peak"] / before["rms"]
        crest_after = after["peak"] / after["rms"]
        assert crest_after < crest_before / 4, (
            f"crest factor barely moved: {crest_before:.1f} -> {crest_after:.1f}")

    def test_filter_chain_compresses_before_it_levels(self):
        # Regression guard: ``loudnorm`` alone only moves the peak, so a chain that dropped
        # the compressor would pass every other test while failing on the real input class.
        assert "acompressor" in _STT_NORMALIZE_FILTER
        assert _STT_NORMALIZE_FILTER.index("acompressor") < _STT_NORMALIZE_FILTER.index("loudnorm")
        assert "highpass" in _STT_NORMALIZE_FILTER

    def test_work_dir_is_removable_by_the_caller(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        path, work_dir = _normalize_stt_input(wav, {"normalize_input": True})
        assert path and Path(path).is_file()
        assert work_dir and Path(work_dir).is_dir()
        shutil.rmtree(work_dir, ignore_errors=True)
        assert not Path(work_dir).exists()


# ============================================================================
# Dispatcher wiring: runs before the backend, before the cloud trim
# ============================================================================

class TestDispatcherWiring:

    def test_normalized_path_reaches_the_provider_and_temp_dir_is_removed(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        norm_dir = tmp_path / "norm-work"
        norm_dir.mkdir()
        normalized = _write_crest_wav(norm_dir / "a-norm.wav", quiet=3000)
        seen = {}

        def fake_dispatch(file_path, model=None, source=None):
            seen["path"] = file_path
            return {"success": True, "transcript": "ok", "provider": "groq"}

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"normalize_input": True}), \
             patch("tools.transcription_tools._normalize_stt_input",
                   return_value=(normalized, str(norm_dir))) as norm, \
             patch("tools.transcription_tools._transcribe_prepared_audio", side_effect=fake_dispatch):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(wav)

        assert result["success"] is True
        norm.assert_called_once()
        assert seen["path"] == normalized
        assert not norm_dir.exists(), "dispatcher must own the normalization temp dir"

    def test_disabled_leaves_the_original_in_place(self, tmp_path):
        wav = _write_crest_wav(tmp_path / "a.wav")
        seen = {}

        def fake_dispatch(file_path, model=None, source=None):
            seen["path"] = file_path
            return {"success": True, "transcript": "ok"}

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._transcribe_prepared_audio", side_effect=fake_dispatch):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(wav)

        assert result["success"] is True
        assert seen["path"] == wav

    @_requires_ffmpeg
    def test_normalization_runs_before_the_cloud_silence_trim(self, tmp_path):
        """The trim cuts anything under -40 dB; on a quiet capture that is the speech itself.
        Normalization must therefore hand the trim an already-compressed file."""
        wav = _write_crest_wav(tmp_path / "a.wav")
        trim_inputs = []

        def fake_trim(file_path, stt_config):
            trim_inputs.append(file_path)

        with patch("tools.transcription_tools._load_stt_config",
                   return_value={"normalize_input": True, "provider": "groq"}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._trim_silence_for_cloud_stt", side_effect=fake_trim), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "ok"}):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(wav)

        assert result["success"] is True
        assert trim_inputs, "cloud trim never ran"
        assert "-norm" in Path(trim_inputs[0]).name, (
            f"the trim saw the raw file ({trim_inputs[0]}), so quiet speech is cut before "
            "normalization can rescue it")
