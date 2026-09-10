"""Long-form chunking, ffmpeg encoding, container repair and delivery packing (``tools.tts_tool``).

Provider-agnostic post-processing: split text under a per-request cap, wrap raw PCM as WAV,
convert WAV/MP3 to the target container, sniff/repair mislabelled ``.ogg`` files, and combine
final-encoded chunks under a destination platform's upload limit. Also home of the sibling
helpers ``_origin`` / ``_section`` / ``_remove_quietly``.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS, DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH, _get_named_provider_config,
    _is_command_provider_config)

logger = logging.getLogger("tools.tts_tool")


def _origin():
    """``tools.tts_tool``, resolved per call so seams monkeypatched there still apply."""
    from tools import tts_tool
    return tts_tool


def _section(tts_config: Any, key: str) -> Dict[str, Any]:
    """``tts.<key>`` as a dict (``null``/non-dict sections read as empty)."""
    section = tts_config.get(key) if isinstance(tts_config, dict) else None
    return section if isinstance(section, dict) else {}


FALLBACK_MAX_TEXT_LENGTH = 4000  # provider not recognised at all

# Per-provider input-character caps (official docs); override: ``tts.<provider>.max_text_length``.
PROVIDER_MAX_TEXT_LENGTH: Dict[str, int] = {
    "edge": 5000,         # edge-tts practical sync limit
    "openai": 4096,       # https://platform.openai.com/docs/guides/text-to-speech
    "xai": 15000,         # https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
    "minimax": 10000,     # https://platform.minimax.io/docs/api-reference/speech-t2a-http (sync)
    "mistral": 4000,      # conservative; no published per-request cap
    "gemini": 32000,      # 32k-token context window; char cap is conservative
    "elevenlabs": 10000,  # fallback when model-aware lookup can't resolve (multilingual_v2)
    "neutts": 2000,       # local model, quality falls off on long text
    "kittentts": 2000,    # local 25MB model
    "piper": 5000,        # local VITS model, phoneme-based; practical cap
}

# ElevenLabs caps vary by model_id. https://elevenlabs.io/docs/overview/models
ELEVENLABS_MODEL_MAX_TEXT_LENGTH: Dict[str, int] = {
    "eleven_v3": 5000, "eleven_ttv_v3": 5000,
    "eleven_multilingual_v2": 10000, "eleven_multilingual_v1": 10000,
    "eleven_english_sts_v2": 10000, "eleven_english_sts_v1": 10000,
    "eleven_flash_v2": 30000, "eleven_flash_v2_5": 40000}


def _positive_int(value: Any) -> Optional[int]:
    """*value* when it is a positive non-bool int, else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _resolve_max_text_length(provider: Optional[str], tts_config: Optional[Dict[str, Any]] = None) -> int:
    """Input-character cap for *provider*: ``tts.<provider>.max_text_length`` > ElevenLabs model
    table > ``PROVIDER_MAX_TEXT_LENGTH`` > command provider's own ``max_text_length`` (else
    ``DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH``) > ``FALLBACK_MAX_TEXT_LENGTH``. Non-positive /
    non-int overrides fall through so a broken config can't disable truncation."""
    if not provider:
        return FALLBACK_MAX_TEXT_LENGTH
    key = provider.lower().strip()
    cfg = tts_config or {}
    prov_cfg = _section(cfg, key)
    override = _positive_int(prov_cfg.get("max_text_length"))
    if override:
        return override
    if key == "elevenlabs":
        from tools.tts_tool_providers import DEFAULT_ELEVENLABS_MODEL_ID  # providers imports this module
        model_id = prov_cfg.get("model_id") or DEFAULT_ELEVENLABS_MODEL_ID
        mapped = ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get(str(model_id).strip())
        if mapped:
            return mapped
    if key in PROVIDER_MAX_TEXT_LENGTH:
        return PROVIDER_MAX_TEXT_LENGTH[key]
    if key not in BUILTIN_TTS_PROVIDERS:
        named = _get_named_provider_config(cfg, key)
        if _is_command_provider_config(named):
            return _positive_int(named.get("max_text_length")) or DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH
    return FALLBACK_MAX_TEXT_LENGTH


# PCM output specs for Gemini TTS (fixed by the API): 24kHz mono 16-bit (L16).
GEMINI_TTS_SAMPLE_RATE, GEMINI_TTS_CHANNELS, GEMINI_TTS_SAMPLE_WIDTH = 24000, 1, 2

# ffmpeg args producing the Ogg/Opus voice-bubble encoding Telegram & co expect.
_OPUS_VOICE_ARGS = [
    "-acodec", "libopus", "-ac", "1", "-b:a", "48k", "-vbr", "on",
    "-application", "voip", "-compression_level", "10"]


# --- Text chunking and delivery profiles ---
@dataclass(frozen=True)
class AudioDeliveryProfile:
    """Destination-platform constraints for generated TTS audio."""

    platform: str
    max_file_bytes: int
    safety_ratio: float = 0.85

    @property
    def target_file_bytes(self) -> int:
        """Conservative packing target below the platform hard limit."""
        return max(1, int(self.max_file_bytes * self.safety_ratio))


_PLATFORM_AUDIO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "discord": {"max_file_bytes": 10 * 1024 * 1024, "safety_ratio": 0.85},
    "telegram": {"max_file_bytes": 50 * 1024 * 1024, "safety_ratio": 0.85},
    "default": {"max_file_bytes": 10 * 1024 * 1024, "safety_ratio": 0.85}}


def _resolve_audio_delivery_profile(
    platform: Optional[str], tts_config: Optional[Dict[str, Any]] = None) -> AudioDeliveryProfile:
    """Resolve upload constraints, including optional ``tts.delivery_profiles`` overrides."""
    key = (platform or "default").lower().strip() or "default"
    defaults = dict(_PLATFORM_AUDIO_DEFAULTS.get(key) or _PLATFORM_AUDIO_DEFAULTS["default"])
    overrides = _section(_section(tts_config, "delivery_profiles"), key)
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    max_file_bytes = (_positive_int(defaults.get("max_file_bytes"))
                      or _PLATFORM_AUDIO_DEFAULTS["default"]["max_file_bytes"])
    safety_ratio = defaults.get("safety_ratio", 0.85)
    if (isinstance(safety_ratio, bool) or not isinstance(safety_ratio, (int, float))
            or not 0 < safety_ratio <= 1):
        safety_ratio = 0.85
    return AudioDeliveryProfile(platform=key, max_file_bytes=max_file_bytes, safety_ratio=float(safety_ratio))


def _pack_under_cap(pieces: List[str], max_chars: int, *, slice_oversized: bool = False) -> List[str]:
    """Greedily join *pieces* with single spaces, starting a new chunk past *max_chars*. With
    ``slice_oversized`` an over-long piece flushes the running chunk and emits its hard slices as
    their own chunks (the tail slice is not merged with following pieces)."""
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if slice_oversized and len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(piece[i:i + max_chars] for i in range(0, len(piece), max_chars))
            continue
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_sentence(sentence: str, max_chars: int) -> List[str]:
    """Split one over-limit sentence on word boundaries, then hard boundaries."""
    return _pack_under_cap(sentence.split(), max_chars, slice_oversized=True)


def _split_text_for_tts(text: str, max_chars: int) -> List[str]:
    """Split text under a provider cap without dropping normalized content."""
    if max_chars <= 0:
        max_chars = FALLBACK_MAX_TEXT_LENGTH
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]
    expanded: List[str] = []
    for sentence in filter(None, (s.strip() for s in re.split(r"(?<=[.!?;:,])\s+", normalized))):
        if len(sentence) <= max_chars:
            expanded.append(sentence)
        else:
            expanded.extend(_split_oversized_sentence(sentence, max_chars))
    return _pack_under_cap(expanded, max_chars)


def _pack_audio_files_for_delivery(audio_paths: List[str], profile: AudioDeliveryProfile) -> List[List[str]]:
    """Group final-encoded chunks under the size target; never mixes suffixes (can't concat-copy)."""
    groups: List[List[str]] = []
    current: List[str] = []
    current_size, current_suffix = 0, ""
    for path in audio_paths:
        size, suffix = Path(path).stat().st_size, Path(path).suffix.lower()
        if current and (current_size + size > profile.target_file_bytes or suffix != current_suffix):
            groups.append(current)
            current, current_size = [], 0
        current.append(path)
        current_size, current_suffix = current_size + size, suffix
    return groups + [current] if current else groups


# --- ffmpeg encoding helpers ---
def _ffmpeg_run(
    ffmpeg: str, args: List[str], *, timeout: int = 30, check: bool = False, capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``ffmpeg <args>`` headless (no stdin, hidden window on Windows)."""
    return subprocess.run([ffmpeg, *args], capture_output=capture, check=check, timeout=timeout,
                          stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())


def _remove_quietly(path: Optional[str]) -> None:
    """Best-effort unlink (missing/locked files are ignored); None is a no-op."""
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def _wav_sidecar_path(output_path: str) -> str:
    """Path a WAV-native engine writes to before conversion to *output_path*'s format."""
    return output_path if output_path.endswith(".wav") else output_path.rsplit(".", 1)[0] + ".wav"


def _finalize_wav_output(wav_path: str, output_path: str) -> str:
    """Move a WAV-native engine's output into the requested container: ffmpeg-convert when
    available, else rename so the tool stays usable (misleading extension, audio plays)."""
    if wav_path == output_path:
        return output_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        os.rename(wav_path, output_path)
        return output_path
    _ffmpeg_run(ffmpeg, ["-i", wav_path, "-y", "-loglevel", "error", output_path],
                check=True, capture=False)
    _remove_quietly(wav_path)
    return output_path


def _wrap_pcm_as_wav(
    pcm_bytes: bytes, sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
    channels: int = GEMINI_TTS_CHANNELS, sample_width: int = GEMINI_TTS_SAMPLE_WIDTH) -> bytes:
    """Wrap raw signed-little-endian PCM (e.g. Gemini's L16) with a minimal WAV RIFF header."""
    block_align = channels * sample_width
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, channels, sample_rate,
                            sample_rate * block_align, block_align, sample_width * 8)
    data_chunk_header = struct.pack("<4sI", b"data", len(pcm_bytes))
    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + len(pcm_bytes)
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    return riff_header + fmt_chunk + data_chunk_header + pcm_bytes


def _write_wav_bytes_as(wav_bytes: bytes, output_path: str) -> str:
    """Write in-memory WAV to *output_path*, ffmpeg-converting to its container. ``.ogg`` is forced
    to Opus (ffmpeg's .ogg default is Vorbis, which voice bubbles reject). A failed conversion
    raises RuntimeError; without ffmpeg the raw WAV is written under the requested name
    (misleading extension, but the audio still plays)."""
    if output_path.lower().endswith(".wav"):
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return output_path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name
    try:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            opus = _OPUS_VOICE_ARGS if output_path.lower().endswith(".ogg") else []
            result = _ffmpeg_run(
                ffmpeg, ["-i", wav_path, *opus, "-y", "-loglevel", "error", output_path])
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        else:
            logger.warning("ffmpeg not found; writing raw WAV to %s (extension may be misleading)", output_path)
            shutil.copyfile(wav_path, output_path)
    finally:
        _remove_quietly(wav_path)
    return output_path


def _convert_to_opus(mp3_path: str) -> Optional[str]:
    """Convert any ffmpeg-readable audio file to OGG Opus next to it; None on failure."""
    return _ffmpeg_transcode_to_opus(mp3_path, mp3_path.rsplit(".", 1)[0] + ".ogg")


def _ffmpeg_transcode_to_opus(input_path: str, ogg_path: str) -> Optional[str]:
    """Transcode *input_path* to real Ogg/Opus at *ogg_path* (in-place safe via temp file); None on failure."""
    if shutil.which("ffmpeg") is None:
        return None
    in_place = os.path.abspath(input_path) == os.path.abspath(ogg_path)
    work_path = ogg_path + ".tmp.ogg" if in_place else ogg_path
    try:
        result = _ffmpeg_run("ffmpeg", ["-i", input_path, *_OPUS_VOICE_ARGS, "-f", "ogg", work_path, "-y"])
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s",
                           result.returncode, result.stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if os.path.exists(work_path) and os.path.getsize(work_path) > 0:
            if in_place:
                os.replace(work_path, ogg_path)
            return ogg_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg OGG conversion timed out after 30s")
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
    finally:
        if in_place and os.path.exists(work_path):
            _remove_quietly(work_path)
    return None


# --- Container sniffing / repair ---
# Several backends ignore the requested opus format (Edge/xAI emit MP3, Piper WAV, some
# OpenAI-compatible servers ignore response_format="opus"), which breaks native voice bubbles:
# sniff the magic bytes once after synthesis and repair when they don't match the extension.

def _sniff_audio_container(path: str) -> str:
    """Return a container id ('ogg', 'wav', 'mp3', 'flac', ...) or 'unknown'."""
    from tools.audio_container import sniff_container
    try:
        with open(path, "rb") as fh:
            return sniff_container(fh.read(12)) or "unknown"
    except OSError:
        return "unknown"


def _repair_ogg_container(file_str: str) -> str:
    """Ensure a ``.ogg`` path really holds Ogg: transcode in place, else rename to the sniffed
    real extension so platforms get an honest file instead of a 0-second voice bubble."""
    container = _sniff_audio_container(file_str) if file_str.endswith(".ogg") else "ogg"
    if container in ("ogg", "unknown"):
        return file_str
    logger.info("TTS wrote %s bytes into a .ogg path (%s) — transcoding to real Ogg/Opus", container, file_str)
    repaired = _ffmpeg_transcode_to_opus(file_str, file_str)
    if repaired:
        return repaired
    honest = f"{file_str[:-4]}.{container}"
    try:
        os.replace(file_str, honest)
    except OSError:
        return file_str
    logger.warning("Could not transcode %s to Ogg/Opus — renamed to %s so the "
                   "file is delivered with its real format", file_str, honest)
    return honest


# --- Long-form audio combination and delivery packing ---
def _concat_audio_files(audio_paths: List[str], output_path: str, *, voice_compatible: bool = False) -> Optional[str]:
    """Combine independently encoded chunks with ffmpeg (never byte-joined). OGG/Opus is always
    re-encoded (even without voice opt-in); matching MP3 chunks keep their frames (``-c:a copy``).
    None when ffmpeg is missing/fails so callers keep the valid parts."""
    if not audio_paths:
        raise ValueError("No audio chunks to combine")
    if len(audio_paths) == 1:
        if os.path.abspath(audio_paths[0]) != os.path.abspath(output_path):
            shutil.copyfile(audio_paths[0], output_path)
        return output_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    concat_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.concat.txt")
    temp_output = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.combining{destination.suffix}")
    try:
        entries = "".join(f"file {shlex.quote(os.path.abspath(p))}\n" for p in audio_paths)
        concat_path.write_text(entries, encoding="utf-8")
        args = ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_path), "-vn"]
        suffix = destination.suffix.lower()
        if voice_compatible or suffix in {".ogg", ".opus"}:
            args += ["-c:a", "libopus", "-ac", "1", "-b:a", "64k", "-vbr", "off"]
        elif suffix == ".mp3" and all(Path(path).suffix.lower() == ".mp3" for path in audio_paths):
            args += ["-c:a", "copy"]
        result = _ffmpeg_run(ffmpeg, [*args, str(temp_output)], timeout=120)
        if result.returncode == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
            os.replace(temp_output, destination)
            return str(destination)
        logger.warning("ffmpeg audio combine failed: %s", result.stderr.decode("utf-8", errors="ignore")[:500])
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg audio combine failed: %s", exc)
    finally:
        for path in (concat_path, temp_output):
            _remove_quietly(path)
    return None


def _build_audio_delivery_files(
    audio_paths: List[str], output_path: str, profile: AudioDeliveryProfile, *, voice_compatible: bool = False,
) -> Tuple[List[str], bool]:
    """Pack final-encoded chunks under the hard upload limit -> ``(final_paths, combined_any)``.

    Groups are packed against the conservative target, then each combined artifact is checked
    at its real size; an over-limit group is split in half and retried. A failed combine
    returns the constituent files separately. A single chunk above the hard limit fails closed."""
    if not audio_paths:
        raise ValueError("No final-encoded TTS audio chunks")
    for path in audio_paths:
        size = Path(path).stat().st_size
        if size > profile.max_file_bytes:
            raise ValueError(
                f"Final-encoded TTS chunk exceeds {profile.platform} delivery "
                f"limit ({size} > {profile.max_file_bytes} bytes): {path}")
    base = Path(output_path)
    scratch_outputs: List[str] = []
    combined_any, combine_index = False, 0

    def emit(group: List[str]) -> List[str]:
        nonlocal combined_any, combine_index
        if len(group) == 1:
            return list(group)
        combine_index += 1
        scratch = base.with_name(f".{base.stem}.delivery{combine_index:03d}.{uuid.uuid4().hex}{base.suffix}")
        combined = _concat_audio_files(group, str(scratch), voice_compatible=voice_compatible)
        if not combined:
            return list(group)
        scratch_outputs.append(combined)
        if Path(combined).stat().st_size <= profile.max_file_bytes:
            combined_any = True
            return [combined]
        _remove_quietly(combined)
        midpoint = max(1, len(group) // 2)
        return emit(group[:midpoint]) + emit(group[midpoint:])
    groups = _pack_audio_files_for_delivery(audio_paths, profile)
    packed = [path for group in groups for path in emit(group)]
    final_paths: List[str] = []
    for index, source in enumerate(packed, start=1):
        destination = base
        if len(packed) > 1:
            destination = base.with_name(f"{base.stem}.part{index:02d}{Path(source).suffix or base.suffix}")
        if os.path.abspath(source) != os.path.abspath(destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        if destination.stat().st_size > profile.max_file_bytes:
            raise ValueError(f"Final TTS deliverable exceeds {profile.platform} delivery limit: {destination}")
        final_paths.append(str(destination))
    try:
        return final_paths, combined_any
    finally:
        for scratch in scratch_outputs:
            if scratch not in final_paths:
                _remove_quietly(scratch)
