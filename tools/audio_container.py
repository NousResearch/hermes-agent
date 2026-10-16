"""Shared magic-byte audio/AV container detection — the ONE sniffer for the codebase.

Outbound (``tools/tts_tool.py``): TTS backends silently ignore the requested
format (Edge emits MP3, Piper WAV), so the file is sniffed and its ``.ogg``
extension repaired. Inbound (``gateway/platforms/base.py`` cache_audio_*,
``gateway/platforms/signal.py``): adapters pass wrong/guessed voice-note
extensions (Telegram ``.oga``, iOS M4A-branded MP4), so the cache sniffs the
real container for STT and players. Only audio/AV containers are claimed:
RIFF/WEBP and other images return ``None`` so callers check images first.
"""

from __future__ import annotations

from typing import Optional

# Container id -> canonical file extension.
CONTAINER_TO_EXT = {c: f".{c}" for c in ("m4a", "mp4", "ogg", "flac", "wav", "mp3", "aac", "webm")}

# MP4 ftyp brands that mean "this is audio" (iOS voice notes use M4A ).
_MP4_AUDIO_BRANDS = (b"m4a ", b"m4b ")

# Unambiguous fixed prefixes (checked after ftyp/RIFF, which need bytes 8-11).
_PREFIX_CONTAINERS = ((b"OggS", "ogg"), (b"fLaC", "flac"), (b"ID3", "mp3"))


def sniff_container(data: bytes) -> Optional[str]:
    """Return a CONTAINER_TO_EXT key from magic bytes, or ``None`` when unknown."""
    if len(data) >= 8 and data[4:8] == b"ftyp":
        # Brand at bytes 8-11: "M4A "/"M4B " are voice notes/audiobooks;
        # everything else (isom/mp42/avc1/qt) is video.
        if len(data) >= 12 and data[8:12].lower() in _MP4_AUDIO_BRANDS:
            return "m4a"
        return "mp4"
    for prefix, container in _PREFIX_CONTAINERS:
        if data.startswith(prefix):
            return container
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        # ``0xFF 0xFx`` sync word is shared by MP3 and ADTS AAC; bits 3-1 of
        # byte 1 disambiguate: ADTS has ID=0, layer=00 (mask 0xF6 -> 0xF0).
        return "aac" if (data[1] & 0xF6) == 0xF0 else "mp3"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    return None


def sniff_audio_ext(data: bytes, fallback_ext: str = ".ogg") -> str:
    """Return a container-matching extension, or ``fallback_ext`` when unknown.
    Callers *claim* audio, so generic MP4 maps to ``.m4a`` (payload is AAC regardless
    of brand; STT accepts both but voice-bubble routing keys off audio extensions)."""
    fallback = fallback_ext if fallback_ext.startswith(".") else f".{fallback_ext}"
    container = sniff_container(data)
    if container is None:
        return fallback
    return ".m4a" if container == "mp4" else CONTAINER_TO_EXT[container]
