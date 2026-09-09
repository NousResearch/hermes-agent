"""Offline SenseVoiceSmall transcription through the FunASR GGUF runtime."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from tools.transcription_audio import _prepare_local_audio, _run_quiet
from tools.transcription_common import (
    _config_number, _error_result, _log_prompt_unsupported, _ok_result,
    _process_error_detail,
)


_DEFAULT_BINARY = "llama-funasr-sensevoice"
_SUPPORTED_BACKENDS = frozenset({"cpu", "cuda", "vulkan"})
logger = logging.getLogger("tools.transcription_tools")


def _configured_file(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value.strip()).expanduser()


def _sensevoice_binary(value: Any) -> Optional[str]:
    configured = value.strip() if isinstance(value, str) and value.strip() else _DEFAULT_BINARY
    path = Path(configured).expanduser()
    return str(path) if path.is_file() else shutil.which(configured)


def _transcribe_sensevoice(
    file_path: str, model_name: str, *, language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``llama-funasr-sensevoice`` with user-supplied GGUF weights."""
    from tools.transcription_tools import _load_stt_config

    if language:
        logger.debug("STT provider 'sensevoice' does not support language hints — using auto-detection")
    if prompt:
        _log_prompt_unsupported("STT provider 'sensevoice'")
    cfg = (_load_stt_config().get("sensevoice") or {})
    model = _configured_file(model_name)
    if model is None:
        return _error_result(
            "SenseVoice requires stt.sensevoice.model to point to a SenseVoiceSmall GGUF file")
    if not model.is_file():
        return _error_result(f"SenseVoice GGUF model not found: {model}")
    binary = _sensevoice_binary(cfg.get("binary"))
    if not binary:
        return _error_result(
            "SenseVoice runtime not found. Install llama-funasr-sensevoice or set "
            "stt.sensevoice.binary to its path")
    vad_model = _configured_file(cfg.get("vad_model"))
    if vad_model is not None and not vad_model.is_file():
        return _error_result(f"SenseVoice VAD GGUF model not found: {vad_model}")
    backend = str(cfg.get("backend") or "cpu").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        return _error_result(
            f"Unsupported SenseVoice backend {backend!r}; choose cpu, cuda, or vulkan")

    timeout = max(_config_number(cfg, "timeout_seconds", 300, int), 1)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-sensevoice-") as work_dir:
            prepared_input, prep_error = _prepare_local_audio(file_path, work_dir)
            if prep_error:
                return _error_result(prep_error)
            command = [binary, "-m", str(model), "-a", prepared_input]
            if vad_model is not None:
                command.extend(("--vad", str(vad_model)))
            command.extend(("--backend", backend))
            from tools.environments.local import hermes_subprocess_env
            result = _run_quiet(
                command, timeout=timeout,
                env=hermes_subprocess_env(inherit_credentials=False),
            )
        transcript = result.stdout.strip()
        if not transcript:
            return _error_result("SenseVoice completed but produced no transcript")
        return _ok_result(transcript, "sensevoice")
    except subprocess.TimeoutExpired:
        return _error_result(f"SenseVoice transcription timed out after {timeout} seconds")
    except subprocess.CalledProcessError as exc:
        return _error_result(f"SenseVoice transcription failed: {_process_error_detail(exc)}")
    except OSError as exc:
        return _error_result(f"SenseVoice runtime failed to start: {exc}")