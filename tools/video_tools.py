#!/usr/bin/env python3
"""
Video analysis helpers for gateway media enrichment.

The gateway receives videos as cached local file paths. This module turns those
files into a compact text digest by sampling a small number of frames, sending
them through the existing vision helper, and transcribing any extractable audio.
"""

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.transcription_tools import transcribe_audio
from tools.vision_tools import vision_analyze_tool


logger = logging.getLogger(__name__)

DEFAULT_MAX_VIDEO_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_DURATION_SECONDS = 5 * 60
DEFAULT_MAX_FRAMES = 6
DEFAULT_SECONDS_PER_FRAME = 5.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 45
DEFAULT_AUDIO_SECONDS = 3 * 60

COMMON_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


@dataclass(frozen=True)
class SampledFrame:
    timestamp_seconds: float
    path: str


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _find_binary(binary_name: str) -> Optional[str]:
    for directory in COMMON_BIN_DIRS:
        candidate = Path(directory) / binary_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(binary_name)


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _run_command(command: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _probe_duration(video_path: str, ffprobe: str, timeout: int) -> Optional[float]:
    result = _run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        timeout,
    )
    if result.returncode != 0:
        logger.info("ffprobe failed for %s: %s", video_path, result.stderr.strip()[:300])
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _sample_timestamps(
    duration_seconds: Optional[float],
    *,
    max_duration_seconds: float,
    max_frames: int,
    seconds_per_frame: float,
) -> List[float]:
    if duration_seconds is None or duration_seconds <= 0:
        return [0.0]

    bounded_duration = min(duration_seconds, max_duration_seconds)
    if bounded_duration <= 1:
        return [0.0]

    frame_count = min(max_frames, max(1, math.ceil(bounded_duration / seconds_per_frame)))
    if frame_count == 1:
        return [min(0.5, bounded_duration / 2)]

    start = 0.5
    end = max(start, bounded_duration - 0.5)
    step = (end - start) / (frame_count - 1)
    return [start + (step * i) for i in range(frame_count)]


def _extract_frames(
    video_path: str,
    *,
    ffmpeg: str,
    timestamps: List[float],
    output_dir: Path,
    timeout: int,
) -> List[SampledFrame]:
    frames: List[SampledFrame] = []
    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = output_dir / f"frame_{index:02d}.jpg"
        result = _run_command(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(frame_path),
            ],
            timeout,
        )
        if result.returncode != 0:
            logger.info(
                "ffmpeg frame extraction failed at %.2fs for %s: %s",
                timestamp,
                video_path,
                result.stderr.strip()[:300],
            )
            continue
        if frame_path.exists() and frame_path.stat().st_size > 0:
            frames.append(SampledFrame(timestamp_seconds=timestamp, path=str(frame_path)))
    return frames


def _extract_audio_track(
    video_path: str,
    *,
    ffmpeg: str,
    output_dir: Path,
    duration_seconds: Optional[float],
    max_audio_seconds: float,
    timeout: int,
) -> Optional[str]:
    audio_path = output_dir / "audio.wav"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-map",
        "0:a:0?",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if duration_seconds is not None:
        command.extend(["-t", f"{min(duration_seconds, max_audio_seconds):.3f}"])
    else:
        command.extend(["-t", f"{max_audio_seconds:.3f}"])
    command.append(str(audio_path))

    result = _run_command(command, timeout)
    if result.returncode != 0:
        logger.info("ffmpeg audio extraction failed for %s: %s", video_path, result.stderr.strip()[:300])
        return None
    if not audio_path.exists() or audio_path.stat().st_size <= 44:
        return None
    return str(audio_path)


async def _analyze_frames(frames: List[SampledFrame]) -> List[str]:
    summaries: List[str] = []
    for frame in frames:
        timestamp = _format_timestamp(frame.timestamp_seconds)
        prompt = (
            "You are analyzing one sampled frame from a user-sent video for a "
            f"conversational assistant. This frame is at approximately {timestamp}. "
            "Describe the visible scene, people, actions, on-screen text, UI, "
            "objects, and any clues about what is happening. Be concise but specific."
        )
        try:
            result_json = await vision_analyze_tool(
                image_url=frame.path,
                user_prompt=prompt,
            )
            result = json.loads(result_json)
        except Exception as exc:
            logger.warning("Video frame vision analysis failed for %s: %s", frame.path, exc)
            summaries.append(f"- {timestamp}: frame analysis failed ({exc})")
            continue

        if result.get("success"):
            analysis = str(result.get("analysis") or "").strip()
            summaries.append(f"- {timestamp}: {analysis or 'No visual details returned.'}")
        else:
            error = str(result.get("analysis") or result.get("error") or "unknown error").strip()
            summaries.append(f"- {timestamp}: frame analysis unavailable ({error})")
    return summaries


async def _transcribe_video_audio(audio_path: Optional[str]) -> Optional[str]:
    if not audio_path:
        return None
    result = await asyncio.to_thread(transcribe_audio, audio_path)
    if result.get("success"):
        transcript = str(result.get("transcript") or "").strip()
        return transcript or None
    error = str(result.get("error") or "unknown error").strip()
    return f"Audio transcription unavailable ({error})"


async def analyze_video_file(
    video_path: str,
    *,
    max_frames: Optional[int] = None,
    max_duration_seconds: Optional[float] = None,
    seconds_per_frame: Optional[float] = None,
    max_audio_seconds: Optional[float] = None,
    command_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a compact visual/audio digest for a local video file."""
    path = Path(os.path.expanduser(video_path))
    if not path.is_file():
        return {
            "success": False,
            "analysis": f"The video file was not found at {video_path}.",
            "error": "file_not_found",
        }

    max_bytes = _env_int("HERMES_VIDEO_MAX_BYTES", DEFAULT_MAX_VIDEO_BYTES)
    file_size = path.stat().st_size
    if file_size > max_bytes:
        return {
            "success": False,
            "analysis": (
                f"The user sent a video at {video_path}, but it is too large to analyze "
                f"automatically ({file_size} bytes, max {max_bytes} bytes)."
            ),
            "error": "video_too_large",
        }

    ffmpeg = _find_binary("ffmpeg")
    ffprobe = _find_binary("ffprobe")
    if not ffmpeg or not ffprobe:
        missing = ", ".join(name for name, found in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not found)
        return {
            "success": False,
            "analysis": (
                f"The user sent a video at {video_path}, but I could not analyze it "
                f"because {missing} is not available."
            ),
            "error": "ffmpeg_missing",
        }

    resolved_max_frames = max_frames or _env_int("HERMES_VIDEO_MAX_FRAMES", DEFAULT_MAX_FRAMES)
    resolved_max_duration = max_duration_seconds or _env_float(
        "HERMES_VIDEO_MAX_DURATION_SECONDS",
        DEFAULT_MAX_DURATION_SECONDS,
    )
    resolved_seconds_per_frame = seconds_per_frame or _env_float(
        "HERMES_VIDEO_SECONDS_PER_FRAME",
        DEFAULT_SECONDS_PER_FRAME,
    )
    resolved_max_audio = max_audio_seconds or _env_float(
        "HERMES_VIDEO_MAX_AUDIO_SECONDS",
        DEFAULT_AUDIO_SECONDS,
    )
    resolved_timeout = command_timeout_seconds or _env_int(
        "HERMES_VIDEO_COMMAND_TIMEOUT_SECONDS",
        DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )

    try:
        duration = _probe_duration(str(path), ffprobe, resolved_timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("Unable to probe video duration for %s: %s", path, exc)
        duration = None

    timestamps = _sample_timestamps(
        duration,
        max_duration_seconds=resolved_max_duration,
        max_frames=resolved_max_frames,
        seconds_per_frame=resolved_seconds_per_frame,
    )

    with tempfile.TemporaryDirectory(prefix="hermes-video-") as temp_dir:
        temp_path = Path(temp_dir)
        try:
            frames = _extract_frames(
                str(path),
                ffmpeg=ffmpeg,
                timestamps=timestamps,
                output_dir=temp_path,
                timeout=resolved_timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.info("Unable to extract video frames for %s: %s", path, exc)
            frames = []

        analyzed_duration = min(duration, resolved_max_duration) if duration is not None else None
        try:
            audio_path = _extract_audio_track(
                str(path),
                ffmpeg=ffmpeg,
                output_dir=temp_path,
                duration_seconds=analyzed_duration,
                max_audio_seconds=resolved_max_audio,
                timeout=resolved_timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.info("Unable to extract video audio for %s: %s", path, exc)
            audio_path = None

        frame_summaries, transcript = await asyncio.gather(
            _analyze_frames(frames),
            _transcribe_video_audio(audio_path),
        )

    lines = [f"Video file: {path}"]
    if duration is not None:
        duration_line = f"Duration: {_format_timestamp(duration)}"
        if duration > resolved_max_duration:
            duration_line += f" (analyzed the first {_format_timestamp(resolved_max_duration)})"
        lines.append(duration_line)
    else:
        lines.append("Duration: unknown")

    if frame_summaries:
        lines.append("Sampled visual frames:")
        lines.extend(frame_summaries)
    else:
        lines.append("Sampled visual frames: no usable frames could be extracted.")

    if transcript:
        lines.append(f"Audio transcript: {transcript}")
    else:
        lines.append("Audio transcript: no usable audio track found or transcription returned no text.")

    if duration is not None and duration > resolved_max_duration:
        lines.append("Note: analysis was bounded for latency and cost, so later video content was not inspected.")

    success = bool(frame_summaries or transcript)
    analysis = "\n".join(lines)
    return {
        "success": success,
        "analysis": analysis,
        "duration_seconds": duration,
        "sampled_frame_count": len(frame_summaries),
        "error": None if success else "video_analysis_empty",
    }
