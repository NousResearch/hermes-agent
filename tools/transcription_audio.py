"""Audio preprocessing for STT.

Binary discovery, the shared ffmpeg m4a encode (transcode + silence trim),
source/format validation, WeChat .silk decoding, CAF conversion and the
best-effort cloud pre-upload silence trim. Facade-owned state (``_HAS_PILK``,
``_safe_find_spec``) is read lazily from ``tools.transcription_tools``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from utils import is_truthy_value
from tools.transcription_common import (
    COMMON_LOCAL_BIN_DIRS, LOCAL_NATIVE_AUDIO_FORMATS, MAX_FILE_SIZE, SUPPORTED_FORMATS,
    _config_number, _error_result, _lazy_ensure_quietly, _process_error_detail)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")


def _find_binary(binary_name: str) -> Optional[str]:
    """Find a local binary, checking common Homebrew/local prefixes as well as PATH."""
    for directory in COMMON_LOCAL_BIN_DIRS:
        candidate = Path(directory) / binary_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(binary_name)


_find_ffmpeg_binary = partial(_find_binary, "ffmpeg")
_find_ffprobe_binary = partial(_find_binary, "ffprobe")
_find_whisper_binary = partial(_find_binary, "whisper")


def _run_quiet(command: list, *, timeout: float, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """``subprocess.run`` for STT helper binaries: checked, captured, utf-8 text, no stdin, hidden window."""
    return subprocess.run(
        command, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL, env=env, creationflags=windows_hide_flags())


# Shared encode profile for every STT-bound m4a (transcode and silence-trim):
# 16 kHz mono 32 kbps AAC, faststart. One owner so codec/bitrate never drift.
_STT_M4A_ENCODE_ARGS = ("-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k", "-movflags", "+faststart")


def _run_ffmpeg_stt_encode(ffmpeg: str, input_path: str, output_path: str, *, audio_filter: Optional[str] = None) -> None:
    """Run the shared STT m4a encode, optionally with an ``-af`` filter. Raises on failure; callers own the semantics."""
    filter_args = ["-af", audio_filter] if audio_filter else []
    _run_quiet([ffmpeg, "-y", "-i", input_path, *filter_args, *_STT_M4A_ENCODE_ARGS, output_path],
               timeout=120)


def _transcode_audio_for_stt(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Transcode to a compact 16 kHz mono AAC/m4a for STT upload; ``(converted_path, None)`` or ``(None, error)``.
    Newer OpenAI models reject containers ``whisper-1`` accepted (notably Ogg/Opus voice notes) and
    gateway downloads may carry a misleading extension."""
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "audio needs transcoding for the STT API, but ffmpeg was not found"
    converted_path = os.path.join(work_dir, f"{Path(file_path).stem or 'audio'}-stt.m4a")
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, converted_path)
        return converted_path, None
    except subprocess.CalledProcessError as exc:
        details = _process_error_detail(exc)
        logger.error("ffmpeg STT transcode failed for %s: %s", file_path, details)
        return None, f"failed to transcode audio for the STT API: {details}"
    except Exception as exc:  # noqa: BLE001 - transcode is best-effort
        logger.error("unexpected STT transcode failure for %s: %s", file_path, exc, exc_info=True)
        return None, f"failed to transcode audio for the STT API: {exc}"


def _validate_audio_file_size(audio_path: Path, *, enforce_size_limit: bool = True) -> Optional[Dict[str, Any]]:
    """Return an error when *audio_path* is inaccessible or (if enforced) exceeds the remote upload cap."""
    try:
        file_size = audio_path.stat().st_size
    except OSError as e:
        return _error_result(f"Failed to access file: {e}")
    if enforce_size_limit and file_size > MAX_FILE_SIZE:
        return _error_result(f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024):.0f}MB)")
    return None


def _validate_audio_source_file(file_path: str, *, enforce_size_limit: bool = True) -> Optional[Dict[str, Any]]:
    """Validate source path safety (and optionally size) before any decoder runs."""
    audio_path = Path(file_path)
    if os.path.islink(audio_path):
        return _error_result(f"Path is a symbolic link: {file_path}")
    if not audio_path.exists():
        return _error_result(f"Audio file not found: {file_path}")
    if not audio_path.is_file():
        return _error_result(f"Path is not a file: {file_path}")
    return _validate_audio_file_size(audio_path, enforce_size_limit=enforce_size_limit)


def _validate_audio_file(file_path: str, *, enforce_size_limit: bool = True) -> Optional[Dict[str, Any]]:
    """Validate a supported, decoder-safe audio file."""
    source_error = _validate_audio_source_file(file_path, enforce_size_limit=enforce_size_limit)
    suffix = Path(file_path).suffix
    if source_error or suffix.lower() in SUPPORTED_FORMATS:
        return source_error
    return _error_result(f"Unsupported format: {suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")


def _prepare_audio_for_transcription(file_path: str) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Convert a decoder-safe .silk source to a temporary supported WAV file."""
    from tools.transcription_tools import _HAS_PILK, _safe_find_spec
    audio_path = Path(file_path)
    if audio_path.suffix.lower() != ".silk":
        return file_path, None, None
    if not _HAS_PILK:
        # pilk is a tiny silk-v3 codec binding — lazy-installed on first .silk voice note.
        _lazy_ensure_quietly("stt.silk")
        if not _safe_find_spec("pilk"):
            return None, None, _error_result(
                "Unsupported format: .silk. Install the optional 'pilk' dependency to enable WeChat voice transcription."
            )
    temp_dir = tempfile.mkdtemp(prefix="hermes-silk-")
    converted_path = os.path.join(temp_dir, f"{audio_path.stem}.wav")
    try:
        import pilk
        pilk.silk_to_wav(file_path, converted_path)
        if not Path(converted_path).is_file() or Path(converted_path).stat().st_size == 0:
            raise RuntimeError("pilk did not produce a readable WAV file")
        return converted_path, temp_dir, None
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error("Failed to convert .silk audio %s: %s", file_path, exc, exc_info=True)
        return None, None, _error_result(f"Failed to convert .silk audio for transcription: {exc}")


def _prepare_local_audio(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Normalize audio for local CLI STT when needed."""
    audio_path = Path(file_path)
    if audio_path.suffix.lower() in LOCAL_NATIVE_AUDIO_FORMATS:
        return file_path, None
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "Local STT fallback requires ffmpeg for non-WAV inputs, but ffmpeg was not found"
    converted_path = os.path.join(work_dir, f"{audio_path.stem}.wav")
    try:
        _run_quiet([ffmpeg, "-y", "-i", file_path, converted_path], timeout=300)
        return converted_path, None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out for %s", file_path)
        return None, "Audio conversion for local STT timed out"
    except subprocess.CalledProcessError as e:
        details = _process_error_detail(e)
        logger.error("ffmpeg conversion failed for %s: %s", file_path, details)
        return None, f"Failed to convert audio for local STT: {details}"


def _convert_caf_to_wav(file_path: str) -> Optional[str]:
    """Convert CAF to WAV using ffmpeg or afconvert (macOS)."""
    audio_path = Path(file_path)
    wav_path = os.path.join(audio_path.parent, f"{audio_path.stem}.wav")
    ffmpeg = _find_ffmpeg_binary()
    afconvert = shutil.which("afconvert")
    candidates = (
        ("ffmpeg", [ffmpeg, "-y", "-i", file_path, wav_path] if ffmpeg else None),
        ("afconvert", [afconvert, file_path, wav_path, "-d", "LEI16", "-f", "WAVE"] if afconvert else None),
    )
    for label, command in ((label, cmd) for label, cmd in candidates if cmd):
        try:
            _run_quiet(command, timeout=300)
            return wav_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("%s CAF to WAV failed for %s: %s", label, file_path, e)
    return None


# ---- Cloud pre-upload silence trim --------------------------------------
#
# Local faster-whisper gets Silero VAD; cloud providers get the raw file, so silence
# is paid for twice (upload + per-minute billing) and cloud Whisper hallucinates on
# it. Before uploading to a built-in cloud provider we collapse long pauses with
# ffmpeg's silenceremove, keeping ``stt.cloud_trim_keep_ms`` of each pause so word
# boundaries survive. Purely best-effort — ANY of these uploads the original:
# ``stt.cloud_trim_silence: false``, ffmpeg/ffprobe missing, trim failure/timeout, a
# ~empty result (the provider, not a dB heuristic, decides "no speech"), or <10%
# saving. Command-type and plugin providers are NOT trimmed: they may wrap local
# CLIs that want the original bytes.

_CLOUD_TRIM_THRESHOLD_DB_DEFAULT = -40  # audio below this level counts as silence
_CLOUD_TRIM_KEEP_MS_DEFAULT = 300  # how much of each pause survives the trim
_CLOUD_TRIM_MIN_SAVING = 0.10  # use the trimmed file only when >=10% shorter
_CLOUD_TRIM_MIN_RESULT_SECONDS = 0.3  # all-silence guard floor: never upload ~empty audio
# Below this the trim can't pay for itself (several providers bill a 10s minimum)
# and the encode would sit on the synchronous voice-note path.
_CLOUD_TRIM_MIN_INPUT_SECONDS = 12.0


def _probe_audio_duration(file_path: str) -> Optional[float]:
    """Return the audio duration in seconds via ffprobe, or None. Canonical sync probe;
    ``gateway/run.py._probe_audio_duration`` and the Telegram adapter carry local variants — keep
    the command shape in sync."""
    ffprobe = _find_ffprobe_binary()
    if not ffprobe:
        return None
    try:
        probe = _run_quiet([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", file_path], timeout=30)
        return float(probe.stdout.strip())
    except Exception:  # noqa: BLE001 - probe is best-effort
        return None


def _cloud_trim_settings(stt_config: Dict[str, Any]) -> tuple[bool, int, int]:
    """Resolve (enabled, threshold_db, keep_ms) for the cloud silence trim."""
    cfg = stt_config if isinstance(stt_config, dict) else {}
    # is_truthy_value: a YAML string "false" must disable, exactly like is_stt_enabled.
    enabled = is_truthy_value(cfg.get("cloud_trim_silence", True), default=True)
    threshold_db = _config_number(cfg, "cloud_trim_threshold_db", _CLOUD_TRIM_THRESHOLD_DB_DEFAULT, int)
    keep_ms = _config_number(cfg, "cloud_trim_keep_ms", _CLOUD_TRIM_KEEP_MS_DEFAULT, int)
    return enabled, threshold_db, max(keep_ms, 0)


def _trim_silence_for_cloud_stt(file_path: str, stt_config: Dict[str, Any]) -> Optional[str]:
    """Return a silence-trimmed copy of *file_path* for cloud upload, or None (= upload the original).
    On success the caller owns deleting the returned file's parent directory."""
    enabled, threshold_db, keep_ms = _cloud_trim_settings(stt_config)
    if not enabled:
        return None
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        logger.debug("Cloud STT silence trim skipped: ffmpeg not found")
        return None
    original_duration = _probe_audio_duration(file_path)
    if not original_duration or original_duration <= 0:
        logger.debug("Cloud STT silence trim skipped: could not probe %s", file_path)
        return None
    name = Path(file_path).name
    if original_duration < _CLOUD_TRIM_MIN_INPUT_SECONDS:
        logger.debug("Cloud STT silence trim skipped for %s: %.1fs is below the %.0fs gate",
                     name, original_duration, _CLOUD_TRIM_MIN_INPUT_SECONDS)
        return None
    keep_seconds = keep_ms / 1000.0
    # start_periods=1 strips leading silence; stop_periods=-1 collapses every interior/trailing silence.
    filter_expr = (
        f"silenceremove="
        f"start_periods=1:start_threshold={threshold_db}dB:start_silence={keep_seconds}:"
        f"stop_periods=-1:stop_threshold={threshold_db}dB:stop_silence={keep_seconds}")
    work_dir = tempfile.mkdtemp(prefix="hermes-stt-trim-")
    trimmed_path = os.path.join(work_dir, f"{Path(file_path).stem or 'audio'}-trimmed.m4a")
    # Scale the all-silence guard with keep_ms: output that is solely kept pause must never upload as "speech".
    min_result_seconds = max(_CLOUD_TRIM_MIN_RESULT_SECONDS, 2 * keep_seconds)
    keep_result = False
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, trimmed_path, audio_filter=filter_expr)
        trimmed_duration = _probe_audio_duration(trimmed_path)
        if not trimmed_duration or trimmed_duration < min_result_seconds:
            logger.debug("Cloud STT silence trim discarded for %s: trimmed result ~empty (%.2fs)",
                         name, trimmed_duration or 0.0)
            return None
        if trimmed_duration > original_duration * (1 - _CLOUD_TRIM_MIN_SAVING):
            logger.debug("Cloud STT silence trim discarded for %s: saves <%.0f%% (%.1fs -> %.1fs)",
                         name, _CLOUD_TRIM_MIN_SAVING * 100, original_duration, trimmed_duration)
            return None
        logger.info("Trimmed silence from %s before cloud STT upload (%.1fs -> %.1fs, -%d%%)", name,
                    original_duration, trimmed_duration, round((1 - trimmed_duration / original_duration) * 100))
        keep_result = True
        return trimmed_path
    except Exception as exc:  # noqa: BLE001 - trim is best-effort
        logger.debug("Cloud STT silence trim failed for %s: %s", file_path, exc)
        return None
    finally:
        if not keep_result:
            shutil.rmtree(work_dir, ignore_errors=True)
