"""Per-provider upload limits and long-audio splitting for cloud STT.

Every cloud transcription endpoint caps one request by size, and some models also cap it by
duration. ``transcribe_audio`` asks :func:`exceeds_upload_limit` before dispatch. An
oversized input is re-encoded to the compact STT m4a first, which is the vendors' own advice
and fits most recordings into a single request with no seams. When it still does not fit,
the file is cut into segments at silences close to the limit. Each segment is dispatched on
its own and the transcripts are joined in order.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.transcription_audio import (
    _find_ffmpeg_binary, _probe_audio_duration, _run_quiet, _transcode_audio_for_stt)
from tools.transcription_common import MAX_FILE_SIZE, _error_result, _ok_result

logger = logging.getLogger("tools.transcription_tools")

_MB = 1024 * 1024
# Vendor-documented per-request upload caps. Providers not listed here (DeepInfra's
# OpenAI-compatible endpoint, command and plugin providers) get the OpenAI-shaped default.
_PROVIDER_MAX_BYTES = {
    "openai": 25 * _MB,  # developers.openai.com/api/docs/guides/speech-to-text: "Files can be up to 25 MB"
    "groq": 25 * _MB,  # console.groq.com/docs/speech-to-text: 25 MB free tier (100 MB dev tier)
    "mistral": 500 * _MB,  # docs.mistral.ai/resources/known-limitations: 500 MB, 60 minutes
    "xai": 500 * _MB,  # docs.x.ai speech-to-text: "Max 500 MB" (413 above it)
    "elevenlabs": 5 * 1024 * _MB - 1,  # elevenlabs.io speech-to-text convert: "must be less than 5.0GB"
}
_PROVIDER_MAX_SECONDS = {"mistral": 3600.0}
# OpenAI per-model duration caps, all measured live (the vendor docs give none in seconds):
# - gpt-4o-transcribe / gpt-4o-mini-transcribe have a documented 2,000 max output tokens. 1322 s of
#   dense speech came back cut off at output_tokens=2048, and the API refuses more than 1400 s
#   outright. 450 s keeps fast speech (~200 wpm) under the token ceiling.
# - whisper-1 answers in ~3% of real time (600 s -> 17.8 s, 1200 s -> 55 s), so 600 s segments
#   finish well inside the default 60 s ``stt.openai.timeout``.
# - gpt-transcribe returned 1322 s complete in 16.7 s and accepted 1558 s, so it gets no cap.
_OPENAI_MODEL_MAX_SECONDS = {"gpt-4o-transcribe": 450.0, "gpt-4o-mini-transcribe": 450.0, "whisper-1": 600.0}
# Margin under each cap so container overhead and duration rounding never tip a segment over.
_SEGMENT_HEADROOM = 0.9
# How far back from the ideal cut a silence may sit and still be used as the cut point.
_MAX_SILENCE_LOOKBACK_SECONDS = 60.0


def upload_limit(provider: str, model: Optional[str]) -> Tuple[int, Optional[float]]:
    """``(max_bytes, max_seconds or None)`` for one request to *provider* with *model*."""
    max_seconds = _PROVIDER_MAX_SECONDS.get(provider)
    if provider == "openai":
        max_seconds = _OPENAI_MODEL_MAX_SECONDS.get(model or "")
    return _PROVIDER_MAX_BYTES.get(provider, MAX_FILE_SIZE), max_seconds


def _fits(path: str, limit: Tuple[int, Optional[float]], duration: Optional[float]) -> bool:
    max_bytes, max_seconds = limit
    if os.path.getsize(path) > max_bytes:
        return False
    # An unknown duration (no ffprobe) is left for the vendor to judge.
    return max_seconds is None or duration is None or duration <= max_seconds


def exceeds_upload_limit(path: str, limit: Tuple[int, Optional[float]]) -> bool:
    """True when *path* is over the byte cap, or over a duration cap ffprobe can measure."""
    if os.path.getsize(path) > limit[0]:
        return True
    return limit[1] is not None and not _fits(path, limit, _probe_audio_duration(path))


def _too_large(path: str, max_bytes: int, reason: str) -> Dict[str, Any]:
    size_mb = os.path.getsize(path) / _MB
    return _error_result(
        f"File too large: {size_mb:.1f}MB (max {max_bytes / _MB:.0f}MB) and it could not be split: {reason}")


def _silence_midpoints(ffmpeg: str, path: str) -> List[float]:
    """Midpoints (seconds) of the pauses ffmpeg's silencedetect finds; [] on any failure."""
    try:
        result = _run_quiet([ffmpeg, "-hide_banner", "-nostats", "-i", path,
                             "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"], timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:  # silence-aware cuts are an optimisation
        logger.debug("silencedetect failed for %s: %s", path, exc)
        return []
    starts = [float(v) for v in re.findall(r"silence_start: (-?[\d.]+)", result.stderr)]
    ends = [float(v) for v in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    return [(max(start, 0.0) + end) / 2 for start, end in zip(starts, ends)]


def _cut_points(duration: float, target: float, silences: List[float]) -> List[float]:
    """Cut times giving segments no longer than *target*, each at the latest pause before the ideal cut."""
    cuts: List[float] = []
    position = 0.0
    lookback = min(_MAX_SILENCE_LOOKBACK_SECONDS, target / 4)
    while duration - position > target:
        ideal = position + target
        nearby = [s for s in silences if ideal - lookback <= s <= ideal and s > position]
        position = max(nearby) if nearby else ideal
        cuts.append(position)
    return cuts


def _split(ffmpeg: str, path: str, work_dir: str, cuts: List[float]) -> List[str]:
    pattern = os.path.join(work_dir, "part%03d.m4a")
    _run_quiet([ffmpeg, "-y", "-loglevel", "error", "-i", path, "-vn", "-c", "copy", "-f", "segment",
                "-segment_times", ",".join(f"{c:.3f}" for c in cuts), "-reset_timestamps", "1", pattern],
               timeout=300)
    return sorted(os.path.join(work_dir, name) for name in os.listdir(work_dir) if name.startswith("part"))


def transcribe_oversized(
    path: str, limit: Tuple[int, Optional[float]], dispatch: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit *path* under *limit* (re-encode, else split at silences) and run *dispatch* per piece."""
    from tools.voice_mode_transcript import is_whisper_hallucination

    max_bytes, max_seconds = limit
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return _too_large(path, max_bytes, "ffmpeg was not found")
    with TemporaryDirectory(prefix="hermes-stt-split-", ignore_cleanup_errors=True) as work_dir:
        compact, error = _transcode_audio_for_stt(path, work_dir)
        if error or not compact:
            return _too_large(path, max_bytes, error or "re-encoding produced no file")
        duration = _probe_audio_duration(compact)
        if _fits(compact, limit, duration):
            logger.info("Re-encoded oversized audio %s to %.1fMB for a single STT upload",
                        os.path.basename(path), os.path.getsize(compact) / _MB)
            return dispatch(compact)
        if not duration:
            return _too_large(path, max_bytes, "its duration could not be read")
        target = max_bytes * _SEGMENT_HEADROOM * duration / os.path.getsize(compact)
        if max_seconds:
            target = min(target, max_seconds * _SEGMENT_HEADROOM)
        parts = _split(ffmpeg, compact, work_dir, _cut_points(duration, target, _silence_midpoints(ffmpeg, compact)))
        logger.info("Transcribing %s (%.0fs) in %d segments of <= %.0fs",
                    os.path.basename(path), duration, len(parts), math.ceil(target))
        transcripts: List[str] = []
        provider = ""
        for index, part in enumerate(parts, start=1):
            result = dispatch(part)
            provider = result.get("provider") or provider
            if result.get("no_speech"):
                continue
            if not result.get("success"):
                return _error_result(f"Segment {index}/{len(parts)} failed: {result.get('error', 'unknown error')}",
                                     provider=provider)
            text = (result.get("transcript") or "").strip()
            if text and not is_whisper_hallucination(text):
                transcripts.append(text)
    if not transcripts:
        return _error_result("Transcription returned an empty transcript", no_speech=True, provider=provider)
    return {**_ok_result(" ".join(transcripts), provider), "segments": len(parts)}
