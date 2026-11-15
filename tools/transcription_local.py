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
    DEFAULT_LOCAL_MODEL, DEFAULT_LOCAL_STT_LANGUAGE, GROQ_MODELS, LOCAL_STT_COMMAND_ENV, RETIRED_GROQ_MODELS,
    OPENAI_MODELS, STT_MODEL_CATALOG, _config_number, _error_result, _log_prompt_unsupported, _ok_result,
    _process_error_detail)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")


_LOCAL_LANGUAGE_ALIASES = {
    "繁體中文": "zh",
    "繁体中文": "zh",
    "简体中文": "zh",
    "簡體中文": "zh",
}
_THREE_LETTER_WHISPER_CODES = frozenset({"haw", "yue"})


def _normalize_local_stt_language(
    language: Optional[str], supported_languages: object = None
) -> Optional[str]:
    """Return a Whisper language code, or None so the caller can fall back safely."""
    if not isinstance(language, str) or not language.strip():
        return None
    raw = language.strip()
    folded = raw.casefold().replace("_", "-")
    candidate = _LOCAL_LANGUAGE_ALIASES.get(raw, folded.split("-", 1)[0])
    code_shape_is_valid = len(candidate) == 2 or candidate in _THREE_LETTER_WHISPER_CODES
    if not (candidate.isascii() and candidate.isalpha() and code_shape_is_valid):
        logger.warning("Local STT language %r is not a language code; using fallback", raw)
        return None

    if isinstance(supported_languages, (list, tuple, set, frozenset)):
        supported_codes = {str(code).casefold() for code in supported_languages}
        if candidate not in supported_codes:
            logger.warning("Local STT language %r is unsupported; using fallback", raw)
            return None
    return candidate


def _get_local_command_template() -> Optional[str]:
    configured = os.getenv(LOCAL_STT_COMMAND_ENV, "").strip()
    if configured:
        return configured
    from tools.transcription_whisper_cpp import whisper_cpp_command
    managed_command = whisper_cpp_command()
    if managed_command:
        return managed_command
    whisper_binary = _find_whisper_binary()
    return (f"{shlex.quote(whisper_binary)} {{input_path}} --model {{model}} --output_format txt "
            "--output_dir {output_dir} --language {language}") if whisper_binary else None


def _has_local_command() -> bool:
    return _get_local_command_template() is not None


def _normalize_local_model(model_name: Optional[str]) -> str:
    """Return a valid faster-whisper size; cloud-only names (``whisper-1`` …) fall back to the default with a warning."""
    if not model_name:
        return DEFAULT_LOCAL_MODEL
    if model_name in OPENAI_MODELS | GROQ_MODELS | RETIRED_GROQ_MODELS:
        logger.warning(
            "STT model '%s' is a cloud-only name and cannot be used with the local "
            "provider. Falling back to '%s'. Set stt.local.model to a valid "
            "faster-whisper size (%s).",
            model_name, DEFAULT_LOCAL_MODEL, ", ".join(STT_MODEL_CATALOG["local"]))
        return DEFAULT_LOCAL_MODEL
    return model_name


def _try_lazy_install_stt() -> bool:
    """Install faster-whisper and re-check dynamically so it's usable without a restart.

    ACTION paths only (``_transcribe_local``). Nothing that merely *resolves* or *reports* a
    provider may call this: the install takes the per-install lock for as long as a full extra-set
    rebuild, and a status probe must never start one."""
    try:
        # pm installs are gated by security.allow_lazy_installs; never a blocking
        # prompt mid-session. See #40490.
        import pm
        pm.ensure_import("stt-whisper")
        if _ilu.find_spec("faster_whisper"):
            return True
        logger.warning("faster-whisper was installed but importlib still cannot find it (may require Python restart)")
    except Exception as exc:
        logger.warning(
            "Lazy install of faster-whisper failed: %s. "
            "When the message names a restart, this process selected its dependency generation at "
            "boot and a new one cannot take effect in-flight; otherwise the Hermes process user "
            "may not be able to write to the dependency environment. Run `hermes tools` as the "
            "Hermes installation owner and select Local Whisper under Speech-to-Text.",
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


def _get_idle_unload_seconds(local_cfg: dict[str, Any]) -> int:
    """Resolve the idle unload timeout from config; 0 = never (default), negatives clamp to 0."""
    return max(_config_number(local_cfg, "unload_after_idle_seconds", 0, int), 0)


def _hub_cache_miss_error() -> type:
    """Exception faster-whisper raises for a model missing from the local Hub cache.

    ``huggingface_hub`` is an optional dependency (it arrives with faster-whisper); when it is
    absent, its ``LocalEntryNotFoundError`` base class ``OSError`` is the closest match.
    """
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return OSError
    return LocalEntryNotFoundError


def _create_whisper_model(model_name: str, *, device: str, compute_type: str):
    """Use a cached model without contacting the Hub, downloading only on a cache miss."""
    from faster_whisper import WhisperModel

    kwargs = {"device": device, "compute_type": compute_type}
    try:
        return WhisperModel(model_name, local_files_only=True, **kwargs)
    except (_hub_cache_miss_error(), RuntimeError) as exc:
        # An interrupted first download leaves a snapshot folder without the weights;
        # snapshot_download still returns it and ctranslate2 raises "Unable to open file".
        if isinstance(exc, RuntimeError) and "Unable to open file" not in str(exc):
            raise
        logger.info("faster-whisper model '%s' is not cached; downloading it from the Hugging Face Hub", model_name)

    # huggingface_hub surfaces every Hub/network failure as an OSError subclass
    # (LocalEntryNotFoundError wrapping the ConnectTimeout, HfHubHTTPError). Anything else
    # (CUDA runtime, invalid model size) is not a download problem and propagates untouched.
    try:
        return WhisperModel(model_name, local_files_only=False, **kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to download faster-whisper model '{model_name}': {exc}. "
            "If huggingface.co is unreachable, set HF_ENDPOINT to an accessible mirror; "
            "when using a mirror with hf-xet installed, also set HF_HUB_DISABLE_XET=1."
        ) from exc


def _create_mlx_whisper_model(model_name: str):
    """Persistent Apple Silicon GPU mlx-whisper worker adapter compatible with faster-whisper's .transcribe() API."""
    import json
    import threading
    from types import SimpleNamespace

    mlx_python = Path.home() / ".hermes" / "tools" / "asr-sv" / ".venv" / "bin" / "python"
    if not mlx_python.is_file():
        raise RuntimeError(f"mlx-whisper venv python not found at {mlx_python}")

    repo_map = {
        "mlx-large-v3": "mlx-community/whisper-large-v3-mlx",
        "mlx-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    repo = repo_map.get(model_name, model_name)

    worker_code = r'''
import json, os, sys
real_stdout = sys.stdout
sys.stdout = sys.stderr
import numpy as np
import mlx_whisper
from mlx_whisper.transcribe import ModelHolder
from mlx_whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim, N_FRAMES, N_SAMPLES, SAMPLE_RATE
import mlx.core as mx

DROP_HALLUCINATIONS = {
    "Thank you.", "Thank you for watching!", "Thank you for watching.",
    "Tack för att du tittade!", "Textning.nu", "Undertextning.nu",
    "Teksting av Nicolai Winther", "! Teksting av Nicolai Winther",
    "Välkommen tillbaka!", "Tack.",
}
DROP_SUBSTRINGS = (
    "teksting av nicolai winther",
    "textning.nu",
    "undertextning.nu",
    "tack för att du tittade",
    "thank you for watching",
)

def strip_long_silence(audio_arr, sr: int = SAMPLE_RATE):
    audio = np.asarray(audio_arr, dtype=np.float32)
    if audio.size < sr:
        return audio, False
    win = sr // 20  # 50ms frames
    n_frames = audio.size // win
    if n_frames < 4:
        return audio, False
    frames = audio[:n_frames * win].reshape(n_frames, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    p95 = float(np.percentile(rms, 95))
    noise_floor = float(np.percentile(rms, 20))
    if p95 < 5e-4:
        return audio[:sr], True
    thresh = max(noise_floor * 2.5, min(0.008, p95 * 0.12))
    active = rms >= thresh
    if not np.any(active):
        return audio[:sr], True
    pad = 7  # 350ms context padding on each side of active speech
    kernel = np.ones(2 * pad + 1, dtype=np.int32)
    keep = np.convolve(active.astype(np.int32), kernel, mode="same") > 0
    return frames[keep].reshape(-1), False

def detect_sv_or_en(model, speech_audio):
    mel = log_mel_spectrogram(speech_audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    seg = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
    _, probs = model.detect_language(seg)
    sv_family = probs.get("sv", 0.0) + 0.5 * (
        probs.get("no", 0.0) + probs.get("nn", 0.0) + probs.get("da", 0.0)
    )
    en_prob = probs.get("en", 0.0)
    return "en" if en_prob > sv_family else "sv"

def clean_hallucinations(tx):
    s = str(tx or "").strip()
    if not s or s in DROP_HALLUCINATIONS:
        return ""
    low = s.lower()
    if low.startswith("! teksting av nicolai winther"):
        s = s[len("! Teksting av Nicolai Winther"):].strip(" ,.-!")
        low = s.lower()
    elif low.startswith("teksting av nicolai winther"):
        s = s[len("Teksting av Nicolai Winther"):].strip(" ,.-!")
        low = s.lower()
    if any(sub == low for sub in DROP_SUBSTRINGS):
        return ""
    return s

def emit(payload):
    real_stdout.write("__MLX_JSON__" + json.dumps(payload) + "\n")
    real_stdout.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    req_id = None
    try:
        req = json.loads(raw)
        req_id = req.get("id")
        cmd = req.get("cmd")
        repo = req.get("repo") or "mlx-community/whisper-large-v3-mlx"
        model = ModelHolder.get_model(repo, mx.float16)
        if cmd == "warmup":
            emit({"id": req_id, "ok": True})
            continue
        raw_audio = load_audio(req["audio"])
        speech_audio, is_silent = strip_long_silence(raw_audio)
        if is_silent:
            emit({"id": req_id, "ok": True, "language": "sv", "duration": 0.0, "segments": []})
            continue
        lang = req.get("language")
        if not lang or lang not in ("sv", "en"):
            lang = detect_sv_or_en(model, speech_audio)
        kwargs = {
            "path_or_hf_repo": repo,
            "language": lang,
            "verbose": None,
            "condition_on_previous_text": False,
            "word_timestamps": True,
            "hallucination_silence_threshold": 2.0,
            "no_speech_threshold": float(req.get("no_speech_threshold", 0.6)),
            "logprob_threshold": float(req.get("logprob_threshold", -1.0)),
        }
        if req.get("initial_prompt"):
            kwargs["initial_prompt"] = req["initial_prompt"].rstrip(".")
        res = mlx_whisper.transcribe(speech_audio, **kwargs)
        segs = []
        last_norm = None
        for s in (res.get("segments") or []):
            tx = clean_hallucinations(s.get("text"))
            if not tx:
                continue
            norm = tx.lower()
            if norm == last_norm:
                continue
            last_norm = norm
            segs.append({
                "text": tx,
                "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
                "avg_logprob": float(s.get("avg_logprob", -0.2)),
                "end": float(s.get("end", 0.0)),
            })
        if not segs and (res.get("text") or "").strip():
            tx = clean_hallucinations(res["text"])
            if tx:
                segs.append({"text": tx, "no_speech_prob": 0.0, "avg_logprob": -0.2, "end": 0.0})
        duration = segs[-1]["end"] if segs else 0.0
        emit({
            "id": req_id,
            "ok": True,
            "language": res.get("language") or lang or "sv",
            "duration": duration,
            "segments": segs,
        })
    except Exception as exc:
        emit({"id": req_id, "ok": False, "error": str(exc)})
'''

    class _MlxWhisperModel:
        def __init__(self, target_repo: str):
            self.repo = target_repo
            self._lock = threading.Lock()
            self._proc: Optional[subprocess.Popen] = None
            self._seq = 0
            self._ensure_worker()

        def _rpc(self, proc: subprocess.Popen, payload: dict) -> dict:
            self._seq += 1
            req_id = self._seq
            payload["id"] = req_id
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise EOFError("mlx-whisper worker closed stdout")
                line = line.strip()
                if not line.startswith("__MLX_JSON__"):
                    continue
                resp = json.loads(line[len("__MLX_JSON__"):])
                if resp.get("id") == req_id:
                    return resp

        def _ensure_worker(self) -> subprocess.Popen:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            env["HF_HUB_OFFLINE"] = "1"
            hermes_bin = str(Path.home() / ".hermes" / "bin")
            env["PATH"] = f"{hermes_bin}:{env.get('PATH', '/usr/bin:/bin')}"
            self._proc = subprocess.Popen(
                [str(mlx_python), "-E", "-u", "-c", worker_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env=env,
            )
            resp = self._rpc(self._proc, {"cmd": "warmup", "repo": self.repo})
            if not resp.get("ok"):
                raise RuntimeError(f"mlx-whisper warmup failed: {resp.get('error')}")
            return self._proc

        def transcribe(self, file_path: str, **kwargs):
            with self._lock:
                proc = self._ensure_worker()
                req = {
                    "cmd": "transcribe",
                    "audio": str(file_path),
                    "repo": self.repo,
                    "language": kwargs.get("language"),
                    "initial_prompt": kwargs.get("initial_prompt"),
                    "no_speech_threshold": kwargs.get("no_speech_threshold", 0.6),
                    "logprob_threshold": kwargs.get("log_prob_threshold", -1.0),
                }
                try:
                    resp = self._rpc(proc, req)
                except EOFError:
                    self._proc = None
                    proc = self._ensure_worker()
                    resp = self._rpc(proc, req)
                if not resp.get("ok"):
                    raise RuntimeError(f"mlx-whisper transcription failed: {resp.get('error')}")
                segments = [
                    SimpleNamespace(
                        text=s["text"],
                        no_speech_prob=s.get("no_speech_prob", 0.0),
                        avg_logprob=s.get("avg_logprob", -0.2),
                    )
                    for s in (resp.get("segments") or [])
                ]
                info = SimpleNamespace(
                    language=resp.get("language") or "sv",
                    duration=float(resp.get("duration") or 0.0),
                )
                return segments, info

        def __del__(self):
            proc = getattr(self, "_proc", None)
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    return _MlxWhisperModel(repo)


def _load_local_whisper_model(model_name: str, device: str = "auto", compute_type: str = "auto"):
    """Load faster-whisper with graceful CUDA → CPU fallback. ``device="auto"`` picks CUDA
    whenever the ctranslate2 wheel ships CUDA libs, even on hosts without the NVIDIA runtime (WSL2,
    headless servers): try the requested config first; on a CUDA library load failure fall back to
    CPU + int8. Pass ``stt.local.device`` / ``compute_type`` to pin.

    ``device`` / ``compute_type`` default to ``"auto"`` so the historical behaviour is unchanged; pass
    explicit values from ``stt.local.device`` / ``stt.local.compute_type`` to pin a configuration (#9088).
    """
    if isinstance(model_name, str) and (model_name.startswith("mlx-community/") or model_name.startswith("mlx-")):
        logger.info("Loading Apple Silicon GPU mlx-whisper model '%s'", model_name)
        return _create_mlx_whisper_model(model_name)
    force_cpu = _should_force_faster_whisper_cpu()
    if force_cpu:
        # Importing ctranslate2 can itself abort on Apple Silicon/Rosetta when
        # multiple Intel OpenMP runtimes are loaded — set before the import.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if force_cpu:
        logger.info("Apple Silicon/Rosetta detected — loading faster-whisper on CPU "
                    "(int8) to avoid native device autodetection crashes")
        return _create_whisper_model(model_name, device="cpu", compute_type="int8")
    try:
        return _create_whisper_model(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        if not _looks_like_cuda_lib_error(exc):
            raise
        logger.warning("faster-whisper CUDA load failed (%s) — falling back to CPU (int8). "
                       "Install the NVIDIA CUDA runtime (libcublas/libcudnn) to use GPU.", exc)
        return _create_whisper_model(model_name, device="cpu", compute_type="int8")


# Silence-hallucination hardening for local faster-whisper (whisper decodes junk like
# "You"/"Thank you." from pure silence). Three layers, all tunable under ``stt.local``:
# Silero VAD so silence never reaches the model (``vad: false`` restores raw behaviour
# for music/ambient audio); condition_on_previous_text=False so one hallucinated token
# can't seed a run; and the segment confidence gate in _is_hallucinated_segment.
_VAD_MIN_SILENCE_MS_DEFAULT = 500
_NO_SPEECH_PROB_THRESHOLD_DEFAULT = 0.6
_LOGPROB_THRESHOLD_DEFAULT = -1.0


def build_local_transcribe_kwargs(stt_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Kwargs for EVERY local faster-whisper ``model.transcribe`` call — single owner of the anti-hallucination hardening."""
    from tools.transcription_tools import _load_stt_config, _resolve_stt_language
    stt_config = stt_config if isinstance(stt_config, dict) else _load_stt_config()
    local_cfg = stt_config.get("local") or {}
    # ``vad: null`` in YAML means "default on".
    vad_enabled = local_cfg.get("vad", True)
    kwargs: dict[str, Any] = {
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


def _confidence_thresholds(local_cfg: dict[str, Any]) -> tuple[float, float]:
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


def _join_confident_segments(segments: Any, local_cfg: dict[str, Any]) -> str:
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
) -> dict[str, Any]:
    """Run the configured local STT command template and read back a .txt transcript."""
    from tools.transcription_tools import _resolve_stt_language
    if prompt:
        _log_prompt_unsupported("STT provider 'local_command'")
    command_template = _get_local_command_template()
    if not command_template:
        return _error_result(f"{LOCAL_STT_COMMAND_ENV} not configured and no local whisper binary was found")
    # Language: hook override > stt.local.language > stt.language > env > "en".
    configured_language = language or _resolve_stt_language("local")
    language = _normalize_local_stt_language(configured_language) or DEFAULT_LOCAL_STT_LANGUAGE
    normalized_model = _normalize_local_model(model_name)
    try:
        if not os.getenv(LOCAL_STT_COMMAND_ENV, "").strip():
            from tools.transcription_whisper_cpp import ensure_whisper_cpp_models, whisper_cpp_command
            if command_template == whisper_cpp_command():
                ensure_whisper_cpp_models(normalized_model)
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
            transcript_text = txt_files[0].read_text(encoding="utf-8-sig").strip()
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
