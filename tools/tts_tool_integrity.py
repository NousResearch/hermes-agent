"""Completeness check for synthesized TTS chunks (``tools.tts_tool``).

A provider's HTTP body can end early without raising (a proxy idle cut, a dropped
connection): the OpenAI SDK's ``stream_to_file`` writes whatever arrived, so synthesis
"succeeds" with a few seconds of a long reply. Every chunk is checked once after synthesis,
whichever provider wrote it: an Ogg stream must end on a byte-complete page carrying the
end-of-stream flag, and long text must yield a plausible fraction of its expected duration.
A truncated chunk is synthesized once more as a fresh request; a second truncation fails the
chunk rather than delivering a clipped voice note.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import subprocess
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Optional

from tools.tts_tool_delivery import _origin, _section, _sniff_audio_container

logger = logging.getLogger("tools.tts_tool")

# capture pattern, version, header type, granule position, serial, sequence, CRC, segment count
_OGG_PAGE_HEADER = struct.Struct("<4sBBqIIIB")
_OGG_CAPTURE = b"OggS"
_OGG_END_OF_STREAM = 0x04
_OPUS_GRANULE_RATE = 48000  # Opus granule positions always count 48 kHz samples
_OPUS_HEAD_MAGIC = b"OpusHead"
_OPUS_HEAD_SIZE = 19  # identification header length (RFC 7845 section 5.1)
_OPUS_PRE_SKIP = struct.Struct("<H")  # little-endian uint16 at byte 10 of the header
_OPUS_PRE_SKIP_OFFSET = 10

# Duration plausibility is checked only on text long enough for a clipped reply to be obvious,
# against a deliberately fast speaking rate (typical speech is ~13-14 chars/s), and flags only
# audio under a quarter of that, so a normal reply at any configured speed never trips it.
_MIN_CHARS_FOR_DURATION_CHECK = 400
_FAST_SPEECH_CHARS_PER_SECOND = 20.0
_MIN_DURATION_RATIO = 0.25


@dataclass(frozen=True)
class _OggScan:
    complete: bool  # the file ends exactly on a page boundary
    end_of_stream: bool  # the last page carries the EOS flag
    duration: Optional[float]  # seconds from the last Opus granule; None when not Opus


@dataclass
class _OggWalk:
    end_of_stream: bool = False
    granule: int = -1
    first_page_read: bool = False
    opus_pre_skip: Optional[int] = None  # None when the first page is not an OpusHead


def _opus_pre_skip(first_body: bytes) -> Optional[int]:
    """Pre-skip samples from an ``OpusHead`` packet, or None when the stream is not Opus."""
    if first_body.startswith(_OPUS_HEAD_MAGIC) and len(first_body) >= _OPUS_HEAD_SIZE:
        return _OPUS_PRE_SKIP.unpack_from(first_body, _OPUS_PRE_SKIP_OFFSET)[0]
    return None


def _read_ogg_page(fh: BinaryIO, offset: int, size: int, walk: _OggWalk) -> Optional[int]:
    """Read the page at *offset* into *walk* -> the next page's offset; ``size + 1`` when the page
    runs past the end of the file; None when the bytes there are not an Ogg page."""
    fh.seek(offset)
    header = fh.read(_OGG_PAGE_HEADER.size)
    if not header.startswith(_OGG_CAPTURE[:len(header)]):
        return None
    if len(header) < _OGG_PAGE_HEADER.size:
        return size + 1
    _capture, _version, header_type, granule, _serial, _sequence, _crc, segments = (
        _OGG_PAGE_HEADER.unpack(header))
    lacing = fh.read(segments)
    end = offset + _OGG_PAGE_HEADER.size + segments + sum(lacing)
    if len(lacing) < segments or end > size:
        return size + 1
    if not walk.first_page_read:
        walk.first_page_read = True
        walk.opus_pre_skip = _opus_pre_skip(fh.read(min(sum(lacing), _OPUS_HEAD_SIZE)))
    if granule != -1:
        walk.granule = granule
    walk.end_of_stream = bool(header_type & _OGG_END_OF_STREAM)
    return end


def _scan_ogg(path: str) -> Optional[_OggScan]:
    """Walk the pages of the Ogg file at *path*; None when it is not a walkable Ogg stream
    (unreadable, or bytes that are not Ogg pages), which the caller treats as unverifiable."""
    try:
        size = os.path.getsize(path)
        walk, offset = _OggWalk(), 0
        with open(path, "rb") as fh:
            while offset < size:
                next_offset = _read_ogg_page(fh, offset, size, walk)
                if next_offset is None:
                    return None
                offset = next_offset
    except OSError:
        return None
    duration = None
    if walk.opus_pre_skip is not None and walk.granule >= 0:
        duration = max(0.0, (walk.granule - walk.opus_pre_skip) / _OPUS_GRANULE_RATE)
    complete = offset == size
    return _OggScan(complete=complete, end_of_stream=complete and walk.end_of_stream, duration=duration)


def _probe_duration(path: str) -> Optional[float]:
    """Seconds of audio actually present, summed over the demuxed packets (no decode); None when
    ffprobe is missing or cannot read the file (headerless PCM). The container duration
    (``format=duration``) is not used: an MP3 Xing/Info header keeps reporting the full length
    after the stream was cut."""
    from tools.transcription_audio import _find_ffprobe_binary, _run_quiet
    ffprobe = _find_ffprobe_binary()
    if not ffprobe:
        return None
    try:
        probe = _run_quiet([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "packet=duration_time", "-of", "csv=p=0", path], timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    durations = [seconds for line in probe.stdout.split() if (seconds := _finite_seconds(line)) is not None]
    return sum(durations) if durations else None


def _finite_seconds(value: str) -> Optional[float]:
    """A packet's ``duration_time`` as a finite float; None for ffprobe's ``N/A`` (unknown)."""
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def _speech_speed(tts_config: Any, provider: str) -> float:
    """Configured speed (``tts.<provider>.speed`` > ``tts.speed``) clamped to 1.0-4.0; slower
    speech only lengthens audio, so it never tightens the plausibility floor."""
    default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
    try:
        speed = float(_section(tts_config, provider).get("speed", default))
    except (TypeError, ValueError):
        return 1.0
    return min(4.0, max(1.0, speed))


def _describe(path: str, duration: Optional[float], text: str) -> str:
    audio = f"{duration:.1f} s of audio, " if duration is not None else ""
    return f"{os.path.getsize(path):,} bytes, {audio}{len(text):,} characters of text"


def _audio_truncation_problem(path: str, text: str, tts_config: Any, provider: str) -> Optional[str]:
    """Why the audio at *path* looks cut short, or None when it looks complete or can't be checked."""
    duration = None
    if _sniff_audio_container(path) == "ogg" and (scan := _scan_ogg(path)) is not None:
        if not scan.complete:
            return f"Ogg stream stops partway through a page ({_describe(path, scan.duration, text)})"
        if not scan.end_of_stream:
            return f"final Ogg page has no end-of-stream flag ({_describe(path, scan.duration, text)})"
        duration = scan.duration
    if len(text) < _MIN_CHARS_FOR_DURATION_CHECK:
        return None
    if duration is None:
        duration = _probe_duration(path)
    if duration is None:
        return None
    floor = len(text) / (_FAST_SPEECH_CHARS_PER_SECOND * _speech_speed(tts_config, provider))
    floor *= _MIN_DURATION_RATIO
    if duration >= floor:
        return None
    return f"audio far shorter than its text ({_describe(path, duration, text)}; expected at least {floor:.0f} s)"


def _synthesize_complete_chunk(
    synthesize: Callable[[], tuple[dict[str, Any], str]], text: str, *, index: int, provider: str,
    tts_config: Any,
) -> tuple[dict[str, Any], str]:
    """Run *synthesize* (one fresh provider request -> ``(result, audio_path)``) and check the audio;
    a truncated chunk is retried once, and a second truncation raises the facade's ``_ChunkFailed``."""
    result, path = synthesize()
    problem = _audio_truncation_problem(path, text, tts_config, provider)
    if problem is None:
        return result, path
    logger.warning("TTS chunk %d from %s looks truncated (%s); retrying with a fresh request",
                   index, provider, problem)
    result, path = synthesize()
    retry_problem = _audio_truncation_problem(path, text, tts_config, provider)
    if retry_problem is None:
        return result, path
    raise _origin()._ChunkFailed(
        f"TTS chunk {index} failed ({provider}): the provider's audio stream ended early and the "
        f"audio is incomplete ({problem}); a fresh retry was also incomplete ({retry_problem}).")
