"""
Chatterbox TTS provider for Hermes Agent.

Supports two deployment modes:
- **Local mode**: Imports the chatterbox-tts Python library directly for
  on-device inference. Requires PyTorch and a GPU (CUDA) or Apple Silicon
  (MPS). No API key needed.
- **Server mode**: Calls an OpenAI-compatible HTTP API served by a separate
  Chatterbox server instance (e.g. travisvn/chatterbox-tts-api Docker image).
  Decouples inference from the agent machine.

Both modes support voice cloning via a reference audio file — a unique
capability among Hermes TTS providers.

Configuration (in ~/.hermes/config.yaml)::

    tts:
      provider: chatterbox
      chatterbox:
        mode: local                  # "local" or "server"
        model: original              # "original", "turbo", or "multilingual"
        ref_audio: /path/to/ref.wav  # Voice cloning reference (optional)
        exaggeration: 0.5            # 0.0-2.0, emotion intensity
        cfg_weight: 0.5              # 0.0-1.0, classifier-free guidance
        temperature: 0.8             # Sampling temperature
        language_id: en               # Language (multilingual model only, e.g. en, fr, de)
        # Server mode only:
        url: http://localhost:7860   # Chatterbox API server URL
        predefined_voice_id: Abigail.wav  # Predefined voice filename (server mode)

Chatterbox has a practical generation limit of ~250 characters per call.
Longer text is automatically split at sentence boundaries, generated in
chunks, and concatenated into a single output file.
"""

import io
import logging
import os
import re
import shutil
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ===========================================================================
# Defaults
# ===========================================================================

DEFAULT_CHATTERBOX_MODE = "local"
DEFAULT_CHATTERBOX_MODEL = "original"
DEFAULT_CHATTERBOX_EXAGGERATION = 0.5
DEFAULT_CHATTERBOX_CFG_WEIGHT = 0.5
DEFAULT_CHATTERBOX_TEMPERATURE = 0.8
DEFAULT_CHATTERBOX_URL = "http://localhost:7860"
CHATTERBOX_CHUNK_LIMIT = 240  # chars per generation chunk (safe under ~250 limit)
CHATTERBOX_SAMPLE_RATE = 24000
DEFAULT_CHATTERBOX_LANGUAGE_ID = "en"
_MIN_WAV_HEADER_SIZE = 44  # Minimum valid WAV file: 44-byte header + data


# ===========================================================================
# Text chunking
# ===========================================================================

def _split_text_for_chatterbox(
    text: str,
    limit: int = CHATTERBOX_CHUNK_LIMIT,
) -> List[str]:
    """Split text into chunks at sentence boundaries, respecting the char limit.

    Chatterbox produces degraded output for inputs over ~250 characters.
    This splitter prioritises sentence boundaries (``.`` ``!`` ``?`` ``;``
    ``—``) and falls back to word boundaries when a single sentence
    exceeds the limit.

    Args:
        text: The input text to split.
        limit: Maximum character count per chunk.

    Returns:
        List of non-empty text chunks, each at most *limit* characters long.
        Returns ``[""]`` only if *text* is empty or whitespace-only.
    """
    stripped = text.strip()
    if not stripped:
        return [""]

    # Split on sentence-ending punctuation while keeping the delimiter
    sentences = re.split(r"(?<=[.!?;—])\s+", stripped)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        # If adding this sentence stays under limit, accumulate
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            # Flush current chunk if non-empty
            if current:
                chunks.append(current)
            # If the sentence itself exceeds the limit, split on words
            if len(sentence) > limit:
                words = sentence.split()
                current = ""
                for word in words:
                    test = f"{current} {word}".strip() if current else word
                    if len(test) <= limit:
                        current = test
                    else:
                        if current:
                            chunks.append(current)
                        # If a single word exceeds the limit (e.g. a URL),
                        # split it at the limit boundary
                        if len(word) > limit:
                            for start in range(0, len(word), limit):
                                chunks.append(word[start : start + limit])
                            current = ""
                        else:
                            current = word
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks or [stripped[:limit]]


# ===========================================================================
# WAV concatenation
# ===========================================================================

def _concatenate_wav_bytes(wav_segments: List[bytes]) -> bytes:
    """Concatenate multiple single-channel WAV byte strings into one.

    Assumes all segments share the same sample rate and bit depth.
    Parses each segment's RIFF structure to extract raw PCM data, then
    rebuilds a single valid WAV file.

    Args:
        wav_segments: List of WAV file contents as byte strings.

    Returns:
        A single concatenated WAV file as bytes.
    """
    if not wav_segments:
        return b""
    if len(wav_segments) == 1:
        return wav_segments[0]

    all_pcm = bytearray()
    fmt_chunk = None

    for seg in wav_segments:
        if len(seg) < _MIN_WAV_HEADER_SIZE:
            continue
        # Walk RIFF chunks to locate fmt and data
        pos = 12  # skip "RIFF" + size + "WAVE"
        while pos < len(seg) - 8:
            chunk_id = seg[pos : pos + 4]
            chunk_size = struct.unpack_from("<I", seg, pos + 4)[0]
            if chunk_id == b"fmt ":
                if fmt_chunk is None:
                    fmt_chunk = seg[pos : pos + 8 + chunk_size]
            elif chunk_id == b"data":
                all_pcm.extend(seg[pos + 8 : pos + 8 + chunk_size])
                break
            pos += 8 + chunk_size

    if fmt_chunk is None or not all_pcm:
        # Parsing failed — return the first segment unmodified
        return wav_segments[0]

    # Rebuild a valid WAV file
    data_size = len(all_pcm)
    riff_size = 4 + len(fmt_chunk) + 8 + data_size
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    data_header = b"data" + struct.pack("<I", data_size)
    return bytes(header + fmt_chunk + data_header + all_pcm)


# ===========================================================================
# Local mode
# ===========================================================================

def _check_chatterbox_available() -> bool:
    """Check if the chatterbox-tts package is importable."""
    try:
        import importlib.util

        return importlib.util.find_spec("chatterbox") is not None
    except Exception:
        return False


def _resolve_device() -> str:
    """Pick the best available PyTorch device for local inference.

    Returns:
        ``"cuda"`` if an NVIDIA GPU is available, ``"mps"`` on Apple
        Silicon, or ``"cpu"`` as a fallback.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_local_model(model_variant: str, device: str):
    """Load a Chatterbox model variant onto *device*.

    Handles the ``perth`` watermarker compatibility issue on platforms where
    the native watermarker C extension is unavailable (macOS ARM, some Linux
    containers).  Falls back to the bundled ``DummyWatermarker`` transparently.

    Args:
        model_variant: One of ``"original"``, ``"turbo"``, ``"multilingual"``.
        device: PyTorch device string (``"cuda"``, ``"mps"``, ``"cpu"``).

    Returns:
        A loaded Chatterbox model instance.
    """
    # Patch watermarker for environments where the native library is absent.
    try:
        import perth

        if perth.PerthImplicitWatermarker is None:
            logger.debug(
                "perth native watermarker unavailable, using DummyWatermarker"
            )
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
    except (ImportError, AttributeError):
        pass

    if model_variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        return ChatterboxTurboTTS.from_pretrained(device=device)
    elif model_variant == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=device)


# Module-level model cache to avoid reloading on every call.
_model_lock = threading.Lock()
_cached_model = None
_cached_model_key: Optional[tuple] = None


def _get_or_load_model(model_variant: str, device: str):
    """Return a cached model or load a new one (thread-safe).

    Args:
        model_variant: One of ``"original"``, ``"turbo"``, ``"multilingual"``.
        device: PyTorch device string.

    Returns:
        A loaded Chatterbox model instance.
    """
    global _cached_model, _cached_model_key
    key = (model_variant, device)
    with _model_lock:
        if _cached_model is not None and _cached_model_key == key:
            return _cached_model
        logger.info("Loading Chatterbox %s model on %s...", model_variant, device)
        _cached_model = _load_local_model(model_variant, device)
        _cached_model_key = key
        logger.info("Chatterbox model loaded.")
        return _cached_model


def _generate_local(
    text: str,
    output_path: str,
    cb_config: Dict[str, Any],
) -> str:
    """Generate audio using the local chatterbox-tts library.

    Splits long text into chunks, generates each with the configured
    parameters, and concatenates into a single WAV file.

    Args:
        text: The text to synthesise.
        output_path: Destination file path (WAV format).
        cb_config: The ``tts.chatterbox`` config dict.

    Returns:
        Path to the saved audio file.
    """
    import torch
    import torchaudio

    model_variant = cb_config.get("model", DEFAULT_CHATTERBOX_MODEL)
    device = _resolve_device()
    model = _get_or_load_model(model_variant, device)

    ref_audio = cb_config.get("ref_audio", "")
    exaggeration = float(
        cb_config.get("exaggeration", DEFAULT_CHATTERBOX_EXAGGERATION)
    )
    cfg_weight = float(cb_config.get("cfg_weight", DEFAULT_CHATTERBOX_CFG_WEIGHT))
    temperature = float(
        cb_config.get("temperature", DEFAULT_CHATTERBOX_TEMPERATURE)
    )

    # Validate ref_audio if provided
    if ref_audio and not os.path.isfile(ref_audio):
        logger.warning(
            "Chatterbox ref_audio not found: %s (proceeding without voice cloning)",
            ref_audio,
        )
        ref_audio = ""

    chunks = _split_text_for_chatterbox(text)
    wav_segments: List[bytes] = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        logger.debug(
            "Chatterbox generating chunk %d/%d (%d chars)",
            i + 1,
            len(chunks),
            len(chunk),
        )

        kwargs: Dict[str, Any] = {"text": chunk}
        if ref_audio:
            kwargs["audio_prompt_path"] = ref_audio
        # Pass variant-specific parameters
        if model_variant == "multilingual":
            kwargs["language_id"] = cb_config.get(
                "language_id", DEFAULT_CHATTERBOX_LANGUAGE_ID
            )
        if model_variant in ("original", "turbo"):
            kwargs["exaggeration"] = exaggeration
            kwargs["cfg_weight"] = cfg_weight
            kwargs["temperature"] = temperature

        wav_tensor = model.generate(**kwargs)

        # Normalise tensor shape to (1, samples) for torchaudio.save
        if wav_tensor.ndim == 3:
            wav_tensor = wav_tensor.squeeze(0)
        elif wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Convert to WAV bytes via an in-memory buffer
        buf = io.BytesIO()
        torchaudio.save(buf, wav_tensor.cpu(), CHATTERBOX_SAMPLE_RATE, format="wav")
        wav_segments.append(buf.getvalue())

    if not wav_segments:
        raise RuntimeError("Chatterbox produced no audio segments")

    # Concatenate all chunks and write to output
    combined = _concatenate_wav_bytes(wav_segments)
    with open(output_path, "wb") as f:
        f.write(combined)

    return output_path


# ===========================================================================
# Server mode
# ===========================================================================

def _generate_server(
    text: str,
    output_path: str,
    cb_config: Dict[str, Any],
) -> str:
    """Generate audio via a Chatterbox HTTP server.

    Tries an OpenAI-compatible ``/v1/audio/speech`` endpoint first, then
    falls back to a Gradio ``/api/predict`` endpoint.

    Args:
        text: The text to synthesise.
        output_path: Destination file path (WAV format).
        cb_config: The ``tts.chatterbox`` config dict.

    Returns:
        Path to the saved audio file.

    Raises:
        RuntimeError: If the server is unreachable or returns an error.
    """
    import requests

    url = cb_config.get("url", DEFAULT_CHATTERBOX_URL).rstrip("/")
    ref_audio = cb_config.get("ref_audio", "")
    model_variant = cb_config.get("model", DEFAULT_CHATTERBOX_MODEL)
    exaggeration = float(
        cb_config.get("exaggeration", DEFAULT_CHATTERBOX_EXAGGERATION)
    )
    cfg_weight = float(cb_config.get("cfg_weight", DEFAULT_CHATTERBOX_CFG_WEIGHT))
    temperature = float(
        cb_config.get("temperature", DEFAULT_CHATTERBOX_TEMPERATURE)
    )

    chunks = _split_text_for_chatterbox(text)
    wav_segments: List[bytes] = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        logger.debug("Chatterbox server chunk %d/%d", i + 1, len(chunks))

        # --- Try /tts endpoint ---
        voice_mode = "clone" if ref_audio else "predefined"
        payload: Dict[str, Any] = {
            "text": chunk,
            "voice_mode": voice_mode,
            "output_format": "wav",
            "split_text": False,
        }
        if voice_mode == "predefined":
            payload["predefined_voice_id"] = cb_config.get("predefined_voice_id", "default")
        else:
            payload["reference_audio_filename"] = os.path.basename(ref_audio)
        if exaggeration != DEFAULT_CHATTERBOX_EXAGGERATION:
            payload["exaggeration"] = exaggeration
        if cfg_weight != DEFAULT_CHATTERBOX_CFG_WEIGHT:
            payload["cfg_weight"] = cfg_weight
        if temperature != DEFAULT_CHATTERBOX_TEMPERATURE:
            payload["temperature"] = temperature
        if model_variant == "multilingual":
            payload["language"] = cb_config.get("language_id", DEFAULT_CHATTERBOX_LANGUAGE_ID)


        logger.info("Chatterbox server request: POST %s/tts payload=%s", url, payload)
        try:
            resp = requests.post(
                f"{url}/tts",
                json=payload,
                headers={"Connection": "close"},
                timeout=120,
            )
            resp.raise_for_status()
            wav_segments.append(resp.content)
            continue
        except requests.exceptions.RequestException as exc:
            response_text = getattr(getattr(exc, "response", None), "text", None)
            logger.info(
                "/tts endpoint failed: %s — trying Gradio. Server response: %s", exc, response_text
            )

        # --- Fallback: Gradio predict endpoint ---
        # gradio_payload = {
        #     "data": [
        #         chunk,
        #         ref_audio or None,
        #         exaggeration,
        #         cfg_weight,
        #         temperature,
        #     ],
        # }
        # try:
        #     resp = requests.post(
        #         f"{url}/api/predict",
        #         json=gradio_payload,
        #         timeout=120,
        #     )
        #     resp.raise_for_status()
        # except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Chatterbox server unreachable at {url}: {exc}"
            ) from exc

        result = resp.json()
        audio_data = result.get("data", [None])[0]

        if isinstance(audio_data, str) and audio_data.startswith("data:audio"):
            import base64

            b64_part = audio_data.split(",", 1)[1]
            wav_segments.append(base64.b64decode(b64_part))
        # Server returned a file path — only safe when the server is trusted
        elif isinstance(audio_data, str) and os.path.isfile(audio_data):
            with open(audio_data, "rb") as f:
                wav_segments.append(f.read())
        else:
            raise RuntimeError(
                f"Chatterbox server returned unexpected data format "
                f"(type={type(audio_data).__name__})"
            )

    if not wav_segments:
        raise RuntimeError("Chatterbox server produced no audio segments")

    combined = _concatenate_wav_bytes(wav_segments)
    with open(output_path, "wb") as f:
        f.write(combined)

    return output_path


# ===========================================================================
# Public entry point
# ===========================================================================

def generate_chatterbox_tts(
    text: str,
    output_path: str,
    tts_config: Dict[str, Any],
) -> str:
    """Generate speech using Chatterbox TTS.

    Dispatches to local or server mode based on configuration.  Text
    chunking and WAV concatenation are handled transparently.

    Args:
        text: The text to convert to speech.
        output_path: Where to save the audio file.  Chatterbox generates
            WAV natively; the gateway's Opus converter handles downstream
            format conversion when needed.
        tts_config: The full ``tts:`` config dict from config.yaml.

    Returns:
        Path to the saved audio file.

    Raises:
        ValueError: If local mode is selected but chatterbox-tts is not
            installed, or if *text* is empty.
        RuntimeError: On generation failure.
    """
    if not text or not text.strip():
        raise ValueError("Text is required for Chatterbox TTS generation")

    cb_config = tts_config.get("chatterbox", {})
    mode = cb_config.get("mode", DEFAULT_CHATTERBOX_MODE)

    # Chatterbox generates WAV natively.  If the caller requests a
    # different extension, generate WAV first — the gateway's Opus
    # converter handles MP3/WAV -> OGG conversion transparently.
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav" if "." in output_path else output_path + ".wav"

    if mode == "server":
        logger.info("Generating speech with Chatterbox (server mode)...")
        _generate_server(text, wav_path, cb_config)
    else:
        if not _check_chatterbox_available():
            raise ValueError(
                "Chatterbox TTS provider selected but 'chatterbox-tts' package is "
                "not installed. Install it with: pip install chatterbox-tts\n"
                "Or switch to server mode: set tts.chatterbox.mode to 'server' "
                "and run a Chatterbox server separately."
            )
        logger.info("Generating speech with Chatterbox (local mode)...")
        _generate_local(text, wav_path, cb_config)

    # Convert WAV to the requested format, or rename if no ffmpeg.
    # Matches the NeuTTS provider pattern in tts_tool.py.
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run(
                [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path],
                check=True,
                timeout=30,
            )
            os.remove(wav_path)
        else:
            # No ffmpeg — rename WAV to the requested extension; the
            # downstream Opus converter detects content format correctly.
            os.rename(wav_path, output_path)

    return output_path
