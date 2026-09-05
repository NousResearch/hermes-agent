"""Local STT backends.

faster-whisper loading (CUDA->CPU fallback, Apple Silicon pinning), the
anti-hallucination transcribe kwargs and segment gate, and the local whisper CLI
(``local_command``) provider. The cached-model singleton and idle-unload watcher
stay in ``transcription_tools`` (module state) and are read from it lazily.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import tempfile
import importlib.util as _ilu
from pathlib import Path
from typing import Any, Dict, Optional

from tools.transcription_audio import _find_whisper_binary, _prepare_local_audio, _run_quiet
from tools.transcription_common import (
    DEFAULT_LOCAL_MODEL, DEFAULT_LOCAL_STT_LANGUAGE, GROQ_MODELS, LOCAL_STT_COMMAND_ENV,
    OPENAI_MODELS, _config_number, _error_result, _log_prompt_unsupported, _ok_result,
    _process_error_detail)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")


def _get_local_command_template() -> Optional[str]:
    configured = os.getenv(LOCAL_STT_COMMAND_ENV, "").strip()
    if configured:
        return configured
    whisper_binary = _find_whisper_binary()
    return (f"{shlex.quote(whisper_binary)} {{input_path}} --model {{model}} --output_format txt "
            "--output_dir {output_dir} --language {language}") if whisper_binary else None


def _has_local_command() -> bool:
    return _get_local_command_template() is not None


def _normalize_local_model(model_name: Optional[str]) -> str:
    """Return a valid faster-whisper size; cloud-only names (``whisper-1`` …) fall back to the default with a warning."""
    if not model_name:
        return DEFAULT_LOCAL_MODEL
    if model_name in OPENAI_MODELS | GROQ_MODELS:
        logger.warning(
            "STT model '%s' is a cloud-only name and cannot be used with the local "
            "provider. Falling back to '%s'. Set stt.local.model to a valid "
            "faster-whisper size (tiny, base, small, medium, large-v3).",
            model_name, DEFAULT_LOCAL_MODEL)
        return DEFAULT_LOCAL_MODEL
    return model_name


def _try_lazy_install_stt() -> bool:
    """Lazy-install faster-whisper and re-check dynamically so it's usable without a restart."""
    try:
        from tools.lazy_deps import ensure
        # prompt=False: a bare input() deadlocks under the interactive CLI where prompt_toolkit
        # owns stdin; the install is already gated by security.allow_lazy_installs.
        # prompt=False: never raise a blocking input() prompt mid-session. See #40490.
        ensure("stt.faster_whisper", prompt=False)
        if _ilu.find_spec("faster_whisper"):
            return True
        logger.warning("faster-whisper was installed but importlib still cannot find it (may require Python restart)")
    except Exception as exc:
        logger.warning(
            "Lazy install of faster-whisper failed: %s. "
            "This is often a permission issue: the Hermes process user cannot "
            "write to the virtual environment. Try running manually as the "
            "venv owner: `stat -c '%%u' '$(dirname $(dirname $(which python3)))'` "
            "then `su - <owner> -c 'VIRTUAL_ENV=/opt/hermes/.venv "
            "uv pip install faster-whisper==1.2.1'`",
            exc)
    return False


# Substrings identifying a missing/unloadable CUDA runtime library: the "auto" device
# picker has already committed to CUDA, so we fall back to CPU and reload. Deliberately
# narrow (library names + dlopen phrasing) so legitimate runtime failures like "CUDA
# out of memory" surface to the user instead of silently running on CPU.
_CUDA_LIB_ERROR_MARKERS = (
    "libcublas", "libcudnn", "libcudart", "cannot be loaded", "cannot open shared object",
    "no kernel image is available", "CUBLAS_STATUS_NOT_SUPPORTED", "no CUDA-capable device",
    "CUDA driver version is insufficient")


def _looks_like_cuda_lib_error(exc: BaseException) -> bool:
    """Heuristic: is this a missing/broken CUDA runtime library (not a legitimate runtime failure)?"""
    return any(marker in str(exc) for marker in _CUDA_LIB_ERROR_MARKERS)


def _sysctl_value(name: str) -> str:
    """Return a sysctl value, or an empty string when unavailable."""
    try:
        return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                       timeout=2).strip()
    except Exception:
        return ""


def _should_force_faster_whisper_cpu() -> bool:
    """Force CPU on Apple Silicon (incl. x86_64 under Rosetta), where ctranslate2's
    ``device="auto"`` can abort inside native code before Python can catch it."""
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return True
    # Under Rosetta platform.machine() reports x86_64; sysctl.proc_translated
    # flags translation and hw.optional.arm64 distinguishes Apple Silicon hosts.
    return _sysctl_value("sysctl.proc_translated") == "1" or _sysctl_value("hw.optional.arm64") == "1"


def _get_idle_unload_seconds(local_cfg: Dict[str, Any]) -> int:
    """Resolve the idle unload timeout from config; 0 = never (default), negatives clamp to 0."""
    return max(_config_number(local_cfg, "unload_after_idle_seconds", 0, int), 0)


def _load_local_whisper_model(model_name: str, device: str = "auto", compute_type: str = "auto"):
    """Load faster-whisper with graceful CUDA → CPU fallback. ``device="auto"`` picks CUDA
    whenever the ctranslate2 wheel ships CUDA libs, even on hosts without the NVIDIA runtime (WSL2,
    headless servers): try the requested config first; on a CUDA library load failure fall back to
    CPU + int8. Pass ``stt.local.device`` / ``compute_type`` to pin.

    ``device`` / ``compute_type`` default to ``"auto"`` so the historical behaviour is unchanged; pass
    explicit values from ``stt.local.device`` / ``stt.local.compute_type`` to pin a configuration (#9088).
    """
    force_cpu = _should_force_faster_whisper_cpu()
    if force_cpu:
        # Importing ctranslate2 can itself abort on Apple Silicon/Rosetta when
        # multiple Intel OpenMP runtimes are loaded — set before the import.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from faster_whisper import WhisperModel
    if force_cpu:
        logger.info("Apple Silicon/Rosetta detected — loading faster-whisper on CPU "
                    "(int8) to avoid native device autodetection crashes")
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        if not _looks_like_cuda_lib_error(exc):
            raise
        logger.warning("faster-whisper CUDA load failed (%s) — falling back to CPU (int8). "
                       "Install the NVIDIA CUDA runtime (libcublas/libcudnn) to use GPU.", exc)
        return WhisperModel(model_name, device="cpu", compute_type="int8")


# Silence-hallucination hardening for local faster-whisper (whisper decodes junk like
# "You"/"Thank you." from pure silence). Three layers, all tunable under ``stt.local``:
# Silero VAD so silence never reaches the model (``vad: false`` restores raw behaviour
# for music/ambient audio); condition_on_previous_text=False so one hallucinated token
# can't seed a run; and the segment confidence gate in _is_hallucinated_segment.
_VAD_MIN_SILENCE_MS_DEFAULT = 500
_NO_SPEECH_PROB_THRESHOLD_DEFAULT = 0.6
_LOGPROB_THRESHOLD_DEFAULT = -1.0
# Auto-detect confidence gate: faster-whisper's language_probability below this on an
# auto-detected clip triggers a second pass with ``stt.local.fallback_language``.
_LANGUAGE_CONFIDENCE_THRESHOLD_DEFAULT = 0.6


def build_local_transcribe_kwargs(stt_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Kwargs for EVERY local faster-whisper ``model.transcribe`` call — single owner of the anti-hallucination hardening."""
    from tools.transcription_tools import _load_stt_config, _resolve_stt_language
    stt_config = stt_config if isinstance(stt_config, dict) else _load_stt_config()
    local_cfg = stt_config.get("local") or {}
    # ``vad: null`` in YAML means "default on".
    vad_enabled = local_cfg.get("vad", True)
    kwargs: Dict[str, Any] = {
        "beam_size": 5,
        "condition_on_previous_text": False,
        "vad_filter": vad_enabled is None or bool(vad_enabled)}
    if kwargs["vad_filter"]:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": _config_number(local_cfg, "vad_min_silence_ms", _VAD_MIN_SILENCE_MS_DEFAULT, int)
        }
    # Push the confidence gate into faster-whisper itself: its internal defaults drop
    # low-confidence segments BEFORE our post-filter sees them, so the ``stt.local``
    # threshold knobs were dead for that first gate (non-English speech decodes at
    # lower avg_logprob and was silently discarded). Same values feed both gates.
    kwargs["no_speech_threshold"], kwargs["log_prob_threshold"] = _confidence_thresholds(local_cfg)
    forced_lang = _resolve_stt_language("local", stt_config)
    if forced_lang:
        kwargs["language"] = forced_lang
    initial_prompt = local_cfg.get("initial_prompt")
    if isinstance(initial_prompt, str) and initial_prompt.strip():
        kwargs["initial_prompt"] = initial_prompt
    return kwargs


def low_confidence_fallback_language(info: Any, local_cfg: Dict[str, Any]) -> Optional[str]:
    """The language to re-transcribe with when faster-whisper AUTO-detected one at low
    confidence, else None.

    Whisper's language detection misreads short or accented clips, which users see as a
    voice note transcribed in the wrong language. Forcing one language for everyone
    (the pre-2026-09 default ``stt.language: "en"``) fixed that for English speakers and
    turned every non-English voice note into nonsense. This gate keeps auto-detect as
    the default and reserves the forced language for the low-confidence case only:
    ``stt.local.fallback_language`` (``""`` = never) and
    ``stt.local.language_confidence_threshold`` (faster-whisper's ``language_probability``).
    A forced language (config, env or a pre_transcription hook) never reaches here.
    """
    fallback = str(local_cfg.get("fallback_language") or "").strip().lower()
    if not fallback:
        return None
    try:
        probability = float(getattr(info, "language_probability"))
    except (AttributeError, TypeError, ValueError):
        return None
    threshold = _config_number(
        local_cfg, "language_confidence_threshold", _LANGUAGE_CONFIDENCE_THRESHOLD_DEFAULT)
    detected = str(getattr(info, "language", "") or "").strip().lower()
    if detected == fallback or probability >= threshold:
        return None
    return fallback


def _confidence_thresholds(local_cfg: Dict[str, Any]) -> tuple[float, float]:
    """Resolve (no_speech_prob, avg_logprob) gate thresholds from config."""
    return (_config_number(local_cfg, "no_speech_prob_threshold", _NO_SPEECH_PROB_THRESHOLD_DEFAULT),
            _config_number(local_cfg, "logprob_threshold", _LOGPROB_THRESHOLD_DEFAULT))


def _is_hallucinated_segment(segment: Any, no_speech_threshold: float, logprob_threshold: float) -> bool:
    """True when a segment is very likely a silence hallucination. Conservative AND gate
    (openai-whisper's own heuristic): non-speech AND low decode confidence, so quiet-but-real speech
    survives. Unknown segment shapes are never dropped."""
    try:
        return (float(segment.no_speech_prob) > no_speech_threshold
                and float(segment.avg_logprob) < logprob_threshold)
    except (AttributeError, TypeError, ValueError):
        return False


def _join_confident_segments(segments: Any, local_cfg: Dict[str, Any]) -> str:
    """Join segment texts, dropping probable silence hallucinations."""
    no_speech_threshold, logprob_threshold = _confidence_thresholds(local_cfg)
    kept: list[str] = []
    for segment in segments:
        if _is_hallucinated_segment(segment, no_speech_threshold, logprob_threshold):
            logger.debug("Dropping probable hallucinated segment %r (no_speech_prob=%.3f, avg_logprob=%.3f)",
                         getattr(segment, "text", ""), getattr(segment, "no_speech_prob", float("nan")),
                         getattr(segment, "avg_logprob", float("nan")))
            continue
        kept.append(segment.text.strip())
    return " ".join(kept).strip()


def _transcribe_local_command(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Run the configured local STT command template and read back a .txt transcript."""
    from tools.transcription_tools import _resolve_stt_language
    if prompt:
        _log_prompt_unsupported("STT provider 'local_command'")
    command_template = _get_local_command_template()
    if not command_template:
        return _error_result(f"{LOCAL_STT_COMMAND_ENV} not configured and no local whisper binary was found")
    # Language: hook override > stt.local.language > stt.language > env > "en".
    language = language or _resolve_stt_language("local") or DEFAULT_LOCAL_STT_LANGUAGE
    normalized_model = _normalize_local_model(model_name)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-local-stt-") as output_dir:
            prepared_input, prep_error = _prepare_local_audio(file_path, output_dir)
            if prep_error:
                return _error_result(prep_error)
            command = command_template.format(
                input_path=shlex.quote(prepared_input), output_dir=shlex.quote(output_dir),
                language=shlex.quote(language), model=shlex.quote(normalized_model))
            # Scrub Hermes secrets from the child env (same policy as _run_command_stt).
            # Scrub Hermes secrets from the child env (sibling path to #56332 / _run_command_stt — this
            # local-whisper path previously inherited the full process environment).
            from tools.environments.local import hermes_subprocess_env
            _run_quiet(shlex.split(command), timeout=300, env=hermes_subprocess_env(inherit_credentials=False))
            txt_files = sorted(Path(output_dir).glob("*.txt"))
            if not txt_files:
                return _error_result("Local STT command completed but did not produce a .txt transcript")
            transcript_text = txt_files[0].read_text(encoding="utf-8").strip()
            logger.info("Transcribed %s via local STT command (%s, %d chars)",
                        Path(file_path).name, normalized_model, len(transcript_text))
            return _ok_result(transcript_text, "local_command")
    except KeyError as e:
        return _error_result(f"Invalid {LOCAL_STT_COMMAND_ENV} template, missing placeholder: {e}")
    except subprocess.CalledProcessError as e:
        details = _process_error_detail(e)
        logger.error("Local STT command failed for %s: %s", file_path, details)
        return _error_result(f"Local STT failed: {details}")
    except Exception as e:
        logger.error("Unexpected error during local command transcription: %s", e, exc_info=True)
        return _error_result(f"Local transcription failed: {e}")
