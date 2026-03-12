#!/usr/bin/env python3
"""
Transcription helpers for messaging platforms.

Speech-to-text is a gateway preprocessing step, not an agent tool call.
Hermes chooses the backend from STT configuration, with OpenAI preserving the
project's prior default behavior and local ``whisper.cpp`` available when
explicitly configured.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# Default STT provider/model.
DEFAULT_STT_PROVIDER = "openai"
DEFAULT_STT_MODEL = "whisper-1"
DEFAULT_WHISPERCPP_LANGUAGE = "auto"

# Supported input formats
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}

# Maximum file size (25MB - OpenAI limit, also a sensible local default)
MAX_FILE_SIZE = 25 * 1024 * 1024

WHISPERCPP_PROVIDER_ALIASES = {"whispercpp", "whisper.cpp", "local"}
OPENAI_PROVIDER_ALIASES = {"openai"}
WHISPERCPP_PROVIDER = "whispercpp"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into base defaults."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _default_stt_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "provider": DEFAULT_STT_PROVIDER,
        "model": DEFAULT_STT_MODEL,
        "whispercpp": {
            "binary_path": "",
            "model_path": "",
            "language": DEFAULT_WHISPERCPP_LANGUAGE,
            "ffmpeg_path": "ffmpeg",
        },
    }


def _normalize_provider(provider: str) -> str:
    normalized = str(provider or DEFAULT_STT_PROVIDER).strip().lower()
    if normalized in WHISPERCPP_PROVIDER_ALIASES:
        return WHISPERCPP_PROVIDER
    if normalized in OPENAI_PROVIDER_ALIASES:
        return "openai"
    return normalized


def resolve_stt_config() -> Dict[str, Any]:
    """
    Load the effective STT config from config.yaml plus optional env overrides.
    """
    config = _default_stt_config()

    try:
        from hermes_cli.config import load_config

        user_stt = load_config().get("stt", {})
        if isinstance(user_stt, dict):
            config = _deep_merge(config, user_stt)
    except Exception as exc:
        logger.debug("Failed to load Hermes STT config: %s", exc)

    whispercpp_cfg = config.setdefault("whispercpp", {})
    env_map = {
        "enabled": os.getenv("HERMES_STT_ENABLED"),
        "provider": os.getenv("HERMES_STT_PROVIDER"),
        "model": os.getenv("HERMES_STT_MODEL"),
    }
    whisper_env_map = {
        "binary_path": os.getenv("HERMES_STT_WHISPERCPP_BINARY_PATH"),
        "model_path": os.getenv("HERMES_STT_WHISPERCPP_MODEL_PATH"),
        "language": os.getenv("HERMES_STT_WHISPERCPP_LANGUAGE"),
        "ffmpeg_path": os.getenv("HERMES_STT_WHISPERCPP_FFMPEG_PATH"),
    }

    enabled_override = env_map["enabled"]
    if enabled_override is not None:
        config["enabled"] = enabled_override.strip().lower() in {"1", "true", "yes", "on"}

    for key in ("provider", "model"):
        value = env_map[key]
        if value:
            config[key] = value.strip()

    for key, value in whisper_env_map.items():
        if value:
            whispercpp_cfg[key] = value.strip()

    config["provider"] = _normalize_provider(config.get("provider", DEFAULT_STT_PROVIDER))
    whispercpp_cfg["language"] = str(
        whispercpp_cfg.get("language", DEFAULT_WHISPERCPP_LANGUAGE) or DEFAULT_WHISPERCPP_LANGUAGE
    ).strip() or DEFAULT_WHISPERCPP_LANGUAGE
    whispercpp_cfg["ffmpeg_path"] = str(whispercpp_cfg.get("ffmpeg_path", "ffmpeg") or "ffmpeg").strip()
    whispercpp_cfg["binary_path"] = str(whispercpp_cfg.get("binary_path", "") or "").strip()
    whispercpp_cfg["model_path"] = str(whispercpp_cfg.get("model_path", "") or "").strip()
    return config


def resolve_whispercpp_binary(config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the whisper.cpp executable from config or PATH."""
    stt_config = config or resolve_stt_config()
    whispercpp_cfg = stt_config.get("whispercpp", {})
    binary_path = str(whispercpp_cfg.get("binary_path", "") or "").strip()

    if binary_path:
        expanded = os.path.expanduser(binary_path)
        if os.path.isabs(expanded):
            return expanded if Path(expanded).is_file() else ""
        if Path(expanded).is_file():
            return expanded
        resolved = shutil.which(expanded)
        return resolved or ""

    for candidate in ("whisper-cli", "main"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def resolve_ffmpeg_binary(config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the ffmpeg executable from config or PATH."""
    stt_config = config or resolve_stt_config()
    ffmpeg_path = str(stt_config.get("whispercpp", {}).get("ffmpeg_path", "ffmpeg") or "ffmpeg").strip()
    if not ffmpeg_path:
        ffmpeg_path = "ffmpeg"
    expanded = os.path.expanduser(ffmpeg_path)
    if os.path.isabs(expanded):
        return expanded if Path(expanded).is_file() else ""
    if Path(expanded).is_file():
        return expanded
    return shutil.which(expanded) or ""


def _validate_audio_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Return an error dict if the input audio file is invalid."""
    audio_path = Path(file_path)

    if not audio_path.exists():
        return {
            "success": False,
            "transcript": "",
            "error": f"Audio file not found: {file_path}",
        }

    if not audio_path.is_file():
        return {
            "success": False,
            "transcript": "",
            "error": f"Path is not a file: {file_path}",
        }

    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": (
                f"Unsupported file format: {audio_path.suffix}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            ),
        }

    try:
        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "transcript": "",
                "error": (
                    f"File too large: {file_size / (1024 * 1024):.1f}MB "
                    f"(max {MAX_FILE_SIZE / (1024 * 1024)}MB)"
                ),
            }
    except OSError as exc:
        logger.error("Failed to get file size for %s: %s", file_path, exc, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Failed to access file: {exc}",
        }

    return None


def _format_failure(error: str) -> Dict[str, Any]:
    return {
        "success": False,
        "transcript": "",
        "error": error,
    }


def _convert_to_whispercpp_wav(input_path: str, output_path: str, ffmpeg_binary: str) -> None:
    """Convert any supported audio input into mono 16 kHz WAV for whisper.cpp."""
    result = subprocess.run(
        [
            ffmpeg_binary,
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg conversion failed: {detail}")


def _transcribe_with_whispercpp(file_path: str, stt_config: Dict[str, Any]) -> Dict[str, Any]:
    whispercpp_cfg = stt_config.get("whispercpp", {})
    binary_path = resolve_whispercpp_binary(stt_config)
    if not binary_path:
        return _format_failure(
            "whisper.cpp binary not found; set stt.whispercpp.binary_path or HERMES_STT_WHISPERCPP_BINARY_PATH"
        )

    model_path_raw = str(whispercpp_cfg.get("model_path", "") or "").strip()
    if not model_path_raw:
        return _format_failure(
            "whisper.cpp model path not configured; set stt.whispercpp.model_path or HERMES_STT_WHISPERCPP_MODEL_PATH"
        )

    model_path = Path(os.path.expanduser(model_path_raw))
    if not model_path.is_file():
        return _format_failure(f"whisper.cpp model not found: {model_path}")

    ffmpeg_binary = resolve_ffmpeg_binary(stt_config)
    if not ffmpeg_binary:
        return _format_failure(
            "ffmpeg not found; install ffmpeg or set stt.whispercpp.ffmpeg_path"
        )

    language = str(whispercpp_cfg.get("language", DEFAULT_WHISPERCPP_LANGUAGE) or DEFAULT_WHISPERCPP_LANGUAGE).strip()
    if not language:
        language = DEFAULT_WHISPERCPP_LANGUAGE

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-stt-") as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "input.wav"
            output_base = temp_path / "transcript"
            transcript_path = output_base.with_suffix(".txt")

            _convert_to_whispercpp_wav(file_path, str(wav_path), ffmpeg_binary)

            command = [
                binary_path,
                "-m",
                str(model_path),
                "-f",
                str(wav_path),
                "-of",
                str(output_base),
                "-otxt",
                "-l",
                language,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip() or "unknown whisper.cpp error"
                return _format_failure(f"whisper.cpp failed: {detail}")

            if not transcript_path.exists():
                return _format_failure(f"whisper.cpp did not create transcript: {transcript_path}")

            transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
            if not transcript_text:
                return _format_failure("whisper.cpp returned an empty transcript")

            logger.info("Transcribed %s locally with whisper.cpp (%d chars)", Path(file_path).name, len(transcript_text))
            return {
                "success": True,
                "transcript": transcript_text,
                "provider": WHISPERCPP_PROVIDER,
            }
    except PermissionError:
        logger.error("Permission denied accessing file: %s", file_path, exc_info=True)
        return _format_failure(f"Permission denied: {file_path}")
    except Exception as exc:
        logger.error("Unexpected whisper.cpp transcription error: %s", exc, exc_info=True)
        return _format_failure(f"Transcription failed: {exc}")


def _transcribe_with_openai(file_path: str, model: str) -> Dict[str, Any]:
    api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY")
    if not api_key:
        return _format_failure("VOICE_TOOLS_OPENAI_KEY not set")

    audio_path = Path(file_path)

    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )

        transcript_text = str(transcription).strip()

        logger.info("Transcribed %s with OpenAI (%d chars)", audio_path.name, len(transcript_text))

        return {
            "success": True,
            "transcript": transcript_text,
            "provider": "openai",
        }

    except PermissionError:
        logger.error("Permission denied accessing file: %s", file_path, exc_info=True)
        return _format_failure(f"Permission denied: {file_path}")
    except APIConnectionError as e:
        logger.error("API connection error during transcription: %s", e, exc_info=True)
        return _format_failure(f"Connection error: {e}")
    except APITimeoutError as e:
        logger.error("API timeout during transcription: %s", e, exc_info=True)
        return _format_failure(f"Request timeout: {e}")
    except APIError as e:
        logger.error("OpenAI API error during transcription: %s", e, exc_info=True)
        return _format_failure(f"API error: {e}")
    except Exception as e:
        logger.error("Unexpected error during transcription: %s", e, exc_info=True)
        return _format_failure(f"Transcription failed: {e}")


def transcribe_audio(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe a local audio file using the configured STT provider.

    Returns a dict with:
      - success (bool)
      - transcript (str)
      - error (str, optional)
      - provider (str, optional)
    """
    validation_error = _validate_audio_file(file_path)
    if validation_error:
        return validation_error

    stt_config = resolve_stt_config()
    if not stt_config.get("enabled", True):
        return _format_failure("speech-to-text is disabled")

    provider = _normalize_provider(stt_config.get("provider", DEFAULT_STT_PROVIDER))
    chosen_model = model or stt_config.get("model") or DEFAULT_STT_MODEL

    if provider == WHISPERCPP_PROVIDER:
        return _transcribe_with_whispercpp(file_path, stt_config)
    if provider == "openai":
        return _transcribe_with_openai(file_path, chosen_model)
    return _format_failure(f"Unsupported STT provider: {provider}")
