"""Bounded chronological JPEG extraction for video-analysis fallback."""

from __future__ import annotations

import base64
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_FRAME_FALLBACK_BYTES = 16 * 1024 * 1024
_DEFAULT_FRAME_COUNT = 12
_DEFAULT_FRAME_MAX_SIDE = 768


class VideoFrameExtractionError(RuntimeError):
    """Raised when a provider needs image frames but none can be produced."""


@dataclass(frozen=True)
class VideoFrame:
    """One chronologically sampled JPEG frame."""

    timestamp_seconds: float
    data_url: str


def _probe_video_duration(video_path: Path, ffprobe_path: str) -> float:
    """Return a finite, positive video duration in seconds."""
    completed = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown error").strip()
        raise VideoFrameExtractionError(
            f"ffprobe could not read the video duration: {detail[:240]}"
        )

    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        raise VideoFrameExtractionError(
            "ffprobe returned no duration for the first video stream"
        )
    try:
        duration = float(lines[0])
    except ValueError as exc:
        raise VideoFrameExtractionError(
            f"ffprobe returned an invalid duration: {lines[0]!r}"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise VideoFrameExtractionError(
            f"ffprobe returned a non-positive duration: {duration!r}"
        )
    return duration


def _frame_timestamps(duration: float, max_frames: int) -> list[float]:
    """Choose bounded midpoint samples that cover the complete timeline."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    frame_count = min(max_frames, max(1, math.ceil(duration)))
    return [
        duration * (index + 0.5) / frame_count
        for index in range(frame_count)
    ]


def _jpeg_data_url(frame_path: Path) -> str:
    data = frame_path.read_bytes()
    if not data.startswith(b"\xff\xd8\xff"):
        raise VideoFrameExtractionError(
            f"ffmpeg produced a non-JPEG frame at {frame_path.name}"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_video_frames(
    video_path: Path,
    *,
    max_frames: int = _DEFAULT_FRAME_COUNT,
    max_side: int = _DEFAULT_FRAME_MAX_SIDE,
) -> list[VideoFrame]:
    """Extract bounded chronological JPEG samples with ffmpeg/ffprobe.

    Subprocesses receive argv vectors, never shell commands. Their stdin is
    disabled, execution is time-bounded, and all temporary files are closed
    before read/delete so the fallback behaves consistently on Windows.
    """
    if max_side < 64:
        raise ValueError("max_side must be at least 64 pixels")

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    missing = [
        name
        for name, resolved in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path))
        if not resolved
    ]
    if missing:
        raise VideoFrameExtractionError(
            "Frame fallback requires ffmpeg and ffprobe on PATH; missing: "
            + ", ".join(missing)
        )

    duration = _probe_video_duration(video_path, ffprobe_path)
    timestamps = _frame_timestamps(duration, max_frames)
    frames: list[VideoFrame] = []
    extraction_errors: list[str] = []
    encoded_bytes = 0

    with tempfile.TemporaryDirectory(prefix="hermes-video-frames-") as temp_dir:
        output_dir = Path(temp_dir)
        for index, timestamp in enumerate(timestamps):
            output_path = output_dir / f"frame-{index:03d}.jpg"
            completed = subprocess.run(
                [
                    ffmpeg_path,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-frames:v",
                    "1",
                    "-threads",
                    "1",
                    "-vf",
                    (
                        f"scale={max_side}:{max_side}:"
                        "force_original_aspect_ratio=decrease:"
                        "force_divisible_by=2"
                    ),
                    "-q:v",
                    "3",
                    "-y",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                timeout=45,
                stdin=subprocess.DEVNULL,
            )
            if completed.returncode != 0 or not output_path.is_file():
                detail_value = completed.stderr or completed.stdout or b"unknown error"
                if isinstance(detail_value, bytes):
                    detail = detail_value.decode("utf-8", errors="replace")
                else:
                    detail = str(detail_value)
                extraction_errors.append(
                    f"{timestamp:.3f}s: {detail.strip()[:160]}"
                )
                continue

            try:
                data_url = _jpeg_data_url(output_path)
            except VideoFrameExtractionError as exc:
                extraction_errors.append(f"{timestamp:.3f}s: {exc}")
                continue

            next_size = encoded_bytes + len(data_url.encode("ascii"))
            if next_size > _MAX_FRAME_FALLBACK_BYTES:
                logger.warning(
                    "Stopping video frame fallback at %d frames: encoded payload "
                    "would exceed %.0f MB",
                    len(frames),
                    _MAX_FRAME_FALLBACK_BYTES / (1024 * 1024),
                )
                break
            frames.append(VideoFrame(timestamp_seconds=timestamp, data_url=data_url))
            encoded_bytes = next_size

    if not frames:
        detail = extraction_errors[0] if extraction_errors else "no frames produced"
        raise VideoFrameExtractionError(
            "ffmpeg could not extract any decodable JPEG frames: " + detail
        )
    if extraction_errors:
        logger.warning(
            "Video frame fallback produced %d/%d frames; first failure: %s",
            len(frames),
            len(timestamps),
            extraction_errors[0],
        )
    return frames


async def extract_video_frames_async(video_path: Path) -> list[VideoFrame]:
    """Keep ffmpeg work off event loops and inside the vision CPU budget."""
    from tools import vision_tools

    return await vision_tools._run_encode_on_cpu_executor(
        extract_video_frames,
        video_path,
    )
