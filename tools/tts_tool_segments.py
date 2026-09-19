"""Paragraph boundaries and PCM assembly for the three local TTS engines.

Speech and pauses travel separately: provider text never contains control markers.
The planner retains pauses across request caps; the writer inserts them before encoding.
This module has no eager optional dependencies and is also used by neutts_synth.py.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOCAL_TTS_PROVIDERS = frozenset({"piper", "kittentts", "neutts"})
DEFAULT_PARAGRAPH_PAUSE_MS = 600
MAX_PARAGRAPH_PAUSE_MS = 10000


def _validate_pause(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_PARAGRAPH_PAUSE_MS:
        raise ValueError(f"paragraph_pause_ms must be an integer from 0 to {MAX_PARAGRAPH_PAUSE_MS}")
    return value


def local_paragraph_pause_ms(provider: str, config: dict) -> int:
    """Local default is 600 ms; zero keeps the legacy single-string synthesis path."""
    if provider not in LOCAL_TTS_PROVIDERS:
        return 0
    section = config.get(provider)
    section = section if isinstance(section, dict) else {}
    return _validate_pause(section.get("paragraph_pause_ms", DEFAULT_PARAGRAPH_PAUSE_MS))


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    pause_before_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("A speech segment must contain non-empty text")
        _validate_pause(self.pause_before_ms)


@dataclass(frozen=True)
class SpeechChunk:
    segments: tuple[SpeechSegment, ...]

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)


def plan_speech_chunks(text: str, max_chars: int, pause_ms: int) -> list[SpeechChunk] | list[str]:
    """Pack cleaned paragraphs without erasing boundaries or pausing mid-paragraph.

    A pause belongs to the following segment, including at the beginning of a new
    request. Single-paragraph and disabled inputs use the original chunker unchanged.
    """
    from tools.tts_tool_delivery import FALLBACK_MAX_TEXT_LENGTH, _split_text_for_tts

    _validate_pause(pause_ms)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not pause_ms or len(paragraphs) < 2:
        return _split_text_for_tts(text, max_chars)
    if max_chars <= 0:
        max_chars = FALLBACK_MAX_TEXT_LENGTH
    chunks: list[SpeechChunk] = []
    current: list[SpeechSegment] = []
    length = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        for piece_index, piece in enumerate(_split_text_for_tts(paragraph, max_chars)):
            pause = pause_ms if paragraph_index and piece_index == 0 else 0
            if current and length + 1 + len(piece) > max_chars:
                chunks.append(SpeechChunk(tuple(current)))
                current, length = [], 0
            length += len(piece) + bool(current)
            current.append(SpeechSegment(piece, pause))
    if current:
        chunks.append(SpeechChunk(tuple(current)))
    return chunks


def encode_segments(segments: tuple[SpeechSegment, ...]) -> str:
    return json.dumps([{"text": s.text, "pause_before_ms": s.pause_before_ms} for s in segments],
                      ensure_ascii=False)


def decode_segments(payload: str) -> tuple[SpeechSegment, ...]:
    """Validate the stdin protocol before the subprocess loads a model."""
    rows = json.loads(payload)
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected a non-empty list of speech segments")
    try:
        return tuple(SpeechSegment(**row) for row in rows)
    except TypeError as exc:
        raise ValueError("Invalid speech segment fields") from exc


@dataclass(frozen=True)
class PCMBlock:
    data: bytes
    sample_rate: int
    sample_width: int = 2
    channels: int = 1


def pcm_from_wav(data: bytes) -> PCMBlock:
    with wave.open(io.BytesIO(data), "rb") as source:
        return PCMBlock(source.readframes(source.getnframes()), source.getframerate(),
                        source.getsampwidth(), source.getnchannels())


def pcm_from_samples(samples, sample_rate: int = 24000) -> PCMBlock:
    """Mono model output to signed little-endian PCM16 (numpy is already an engine dependency)."""
    import numpy as np
    samples = np.asarray(samples)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("TTS output must contain finite mono samples")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    return PCMBlock(pcm.tobytes(), sample_rate)


def write_pcm_segments(path: str, audio: Iterable[tuple[SpeechSegment, PCMBlock]]) -> None:
    """Stream compatible PCM blocks and bounded silence to one atomically published WAV.

    A failed segment or format change leaves no partial output. Only one paragraph's
    waveform is held at a time; silence is written in bounded blocks, not a giant buffer.
    """
    target = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".wav",
                                         dir=target.parent, delete=False) as temp:
            temporary = temp.name
        audio_format = None
        with wave.open(temporary, "wb") as output:
            # A generator may fail before its first block; keep close() from masking it.
            output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            for segment, block in audio:
                _validate_pause(segment.pause_before_ms)
                fmt = (block.channels, block.sample_width, block.sample_rate)
                if (any(type(value) is not int or value <= 0 for value in fmt)
                        or block.sample_width > 4):
                    raise ValueError("Invalid PCM format")
                frame_size = block.channels * block.sample_width
                if not block.data or len(block.data) % frame_size:
                    raise ValueError("TTS produced empty or incomplete PCM frames")
                if audio_format is None:
                    audio_format = fmt
                    output.setnchannels(block.channels)
                    output.setsampwidth(block.sample_width)
                    output.setframerate(block.sample_rate)
                elif audio_format != fmt:
                    raise ValueError("TTS segments have incompatible PCM formats")
                frames = block.sample_rate * segment.pause_before_ms // 1000
                zero_frame = (b"\x80" if block.sample_width == 1 else b"\x00" * block.sample_width) * block.channels
                while frames:
                    count = min(frames, 4096)
                    output.writeframesraw(zero_frame * count)
                    frames -= count
                output.writeframesraw(block.data)
            if audio_format is None:
                raise ValueError("TTS produced no audio segments")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
