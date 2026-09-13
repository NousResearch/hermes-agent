#!/usr/bin/env python3
"""Speech-to-text transcription used by the gateway for voice messages.

Built-in providers: local (faster-whisper, default/free), local_command, groq, openai
(also serves the managed ``nous`` selection), mistral, xai, elevenlabs, deepinfra; plus
user-declared command providers and plugin providers. ``transcribe_audio(path)`` returns
``{"success", "transcript", "error"?, "provider"?}``. This module owns provider resolution,
the dispatcher and the cached local model + idle-unload state; backends live in
``transcription_{common,audio,local,cloud,command}``.
"""

import logging
import os
import shutil
import threading
import time
import importlib.util as _ilu
from pathlib import Path
from typing import Optional, Dict, Any

from utils import is_truthy_value
from tools.transcription_common import (
    BUILTIN_STT_PROVIDERS, CLOUD_STT_PROVIDERS, DEFAULT_ELEVENLABS_STT_MODEL,
    DEFAULT_GROQ_STT_MODEL, DEFAULT_LOCAL_MODEL, DEFAULT_MISTRAL_STT_MODEL, DEFAULT_PROVIDER,
    DEFAULT_STT_MODEL, LOCAL_STT_COMMAND_ENV, LOCAL_STT_LANGUAGE_ENV, _error_result,
    _get_stt_section, _ok_result)
from tools.transcription_audio import (
    _convert_caf_to_wav, _prepare_audio_for_transcription, _trim_silence_for_cloud_stt,
    _validate_audio_file, _validate_audio_file_size, _validate_audio_source_file)
from tools.transcription_local import (
    _get_idle_unload_seconds, _has_local_command, _join_confident_segments,
    _load_local_whisper_model, _looks_like_cuda_lib_error, _normalize_local_model,
    _transcribe_local_command, _try_lazy_install_stt, build_local_transcribe_kwargs)
# The ``_transcribe_<provider>`` handlers are looked up in this module's globals by _dispatch_stt_provider.
from tools.transcription_cloud import (  # noqa: F401  (handlers dispatched via globals())
    _has_xai_stt_credentials, _resolve_openai_audio_client_config, _transcribe_deepinfra,
    _transcribe_elevenlabs, _transcribe_groq, _transcribe_mistral, _transcribe_openai,
    _transcribe_xai)
from tools.transcription_command import (
    _apply_pre_transcription_hook, _dispatch_to_plugin_provider, _enforce_prompt_length_limit,
    _resolve_command_stt_provider_config, _transcribe_command_stt, _unregistered_stt_provider_error)

logger = logging.getLogger(__name__)


def get_env_value(name, default=None):
    """Read env values through the live config module (resolved per call: tests monkeypatch it around import)."""
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """STT API key via the shared voice-key resolver (config > env/.env > credential pool); resolved per call."""
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helpers are in-repo
        return str(get_env_value(env_var) or "").strip()
    return resolve_provider_secret(env_var, provider_id, env_getter=get_env_value)


def _safe_find_spec(module_name: str) -> bool:
    try:
        return _ilu.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in globals() or module_name in os.sys.modules


_HAS_FASTER_WHISPER, _HAS_OPENAI, _HAS_MISTRAL, _HAS_PILK = map(
    _safe_find_spec, ("faster_whisper", "openai", "mistralai", "pilk"))

# Local model singleton; the lock guards check-then-load against concurrent voice messages.
_local_model: Optional[object] = None
_local_model_name: Optional[str] = None
# See #24767.
_local_model_lock = threading.Lock()

# Idle unload: one daemon thread releases the model (hundreds of MB of RAM/VRAM) after a
# configurable idle period, then exits; the next voice message reloads and restarts it.
# _idle_unload_mgmt_lock serializes the start check so no duplicate watchers spawn.
_last_transcription_time: float = 0.0
_idle_unload_thread: Optional[threading.Thread] = None
_idle_unload_stop = threading.Event()
_idle_unload_mgmt_lock = threading.Lock()
_IDLE_UNLOAD_CHECK_INTERVAL = 30  # seconds between idle checks


# ---- Config helpers -----------------------------------------------------
def _load_stt_config() -> dict:
    """Load the ``stt`` section from user config, falling back to defaults."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("stt") or {}
    except Exception:
        return {}


def is_stt_enabled(stt_config: Optional[dict] = None) -> bool:
    cfg = _load_stt_config() if stt_config is None else stt_config
    return is_truthy_value(cfg.get("enabled", True), default=True)


def _resolve_stt_language(
    provider_key: str, stt_config: Optional[Dict[str, Any]] = None, *, extra_keys: tuple = ()
) -> Optional[str]:
    """Language hint for an STT provider, first non-empty wins (never ""): ``stt.<provider>.language``
    (plus *extra_keys* aliases, e.g. ``language_code``) > ``stt.language`` > ``HERMES_LOCAL_STT_LANGUAGE``
    env > None (provider auto-detects)."""
    if stt_config is None:
        stt_config = _load_stt_config()
    provider_cfg = _get_stt_section(stt_config, provider_key)
    candidates = [provider_cfg.get(key) for key in ("language", *extra_keys)]
    if isinstance(stt_config, dict):
        candidates.append(stt_config.get("language"))
    candidates.append(os.getenv(LOCAL_STT_LANGUAGE_ENV))
    return next((c.strip() for c in candidates if isinstance(c, str) and c.strip()), None)


def _openai_audio_unavailable_reason() -> Optional[str]:
    """None when OpenAI audio has usable credentials (config, env, or managed gateway); else the reason."""
    try:
        # Resolve directly instead of via the boolean probe: the probe flattens
        # _resolve_openai_audio_client_config's selection-specific ValueError into False, so a managed
        # openai-audio gateway outage would be logged as a generic "no API key" hint (#93045).
        _resolve_openai_audio_client_config()
        return None
    except ValueError as exc:
        return str(exc)


def _has_openai_audio_backend() -> bool:
    return _openai_audio_unavailable_reason() is None


def _is_local_stt_provider(provider: str, stt_config: Dict[str, Any]) -> bool:
    """Whether *provider* is exempt from Hermes's remote upload cap."""
    return (provider or "").lower().strip() in {"local", "local_command"}


# ---- Provider resolution ------------------------------------------------
def _has_key(env_var: str, provider: str, *, needs_openai: bool = False, needs_mistral: bool = False):
    """Availability probe factory: optional SDK flag AND a resolvable API key."""
    def probe() -> bool:
        sdk_ok = (not needs_openai or _HAS_OPENAI) and (not needs_mistral or _HAS_MISTRAL)
        return sdk_ok and bool(_resolve_provider_key(env_var, provider))
    return probe


def _has_xai_stt_credentials_quietly() -> bool:
    try:
        return _has_xai_stt_credentials()
    except Exception:
        return False


def _resolve_explicit_openai() -> str:
    if not _HAS_OPENAI:
        logger.warning("STT provider 'openai' configured but no API key available")
        return "none"
    # Resolved directly so a managed openai-audio gateway outage is logged with its real reason.
    reason = _openai_audio_unavailable_reason()
    if reason is None:
        return "openai"
    logger.warning("STT provider 'openai' configured but unavailable: %s", reason)
    return "none"


def _detect_local_backend() -> Optional[str]:
    """faster-whisper > local whisper CLI > lazy-installed faster-whisper; None when nothing local works."""
    if _HAS_FASTER_WHISPER:
        return "local"
    return "local_command" if _has_local_command() else ("local" if _try_lazy_install_stt() else None)


def _resolve_explicit_local() -> str:
    backend = _detect_local_backend()
    if not backend:
        logger.warning("STT provider 'local' configured but unavailable "
                       "(install faster-whisper or set HERMES_LOCAL_STT_COMMAND)")
    return backend or "none"


def _resolve_explicit_local_command() -> str:
    if _has_local_command():
        return "local_command"
    if _HAS_FASTER_WHISPER:
        logger.info("Local STT command unavailable, using local faster-whisper")
        return "local"
    logger.warning("STT provider 'local_command' configured but unavailable")
    return "none"


_has_groq_key = _has_key("GROQ_API_KEY", "groq", needs_openai=True)
_has_mistral_key = _has_key("MISTRAL_API_KEY", "mistral", needs_mistral=True)
_has_elevenlabs_key = _has_key("ELEVENLABS_API_KEY", "elevenlabs")
_has_deepinfra_key = _has_key("DEEPINFRA_API_KEY", "deepinfra", needs_openai=True)

# Cloud providers in AUTO-DETECT priority order:
#   name -> (explicit-selection probe, auto-detect probe, explicit warning, auto-detect log)
# The probes differ only for openai (explicit has its own resolver in _EXPLICIT_RESOLVERS;
# auto-detect also requires the SDK) and xai (auto-detect must never raise). DeepInfra is
# LAST so a DEEPINFRA_API_KEY set for chat never displaces an xAI/ElevenLabs auto-selection.
# Mistral only auto-selects when the SDK is present — no lazy-install during passive
# auto-detection (explicit ``provider: mistral`` installs on first use).
_CLOUD_PROVIDER_SPECS = {
    "groq": (_has_groq_key, _has_groq_key,
             "STT provider 'groq' configured but GROQ_API_KEY not set",
             "No local STT available, using Groq Whisper API"),
    "openai": (None, lambda: _HAS_OPENAI and _has_openai_audio_backend(),
               None,
               "No local STT available, using OpenAI Whisper API"),
    "mistral": (_has_mistral_key, _has_mistral_key,
                "STT provider 'mistral' configured but mistralai package not installed or MISTRAL_API_KEY not set",
                "No local STT available, using Mistral Voxtral Transcribe API"),
    "xai": (_has_xai_stt_credentials, _has_xai_stt_credentials_quietly,
            "STT provider 'xai' configured but no xAI credentials are available",
            "No local STT available, using xAI Grok STT API"),
    "elevenlabs": (_has_elevenlabs_key, _has_elevenlabs_key,
                   "STT provider 'elevenlabs' configured but ELEVENLABS_API_KEY not set",
                   "No local STT available, using ElevenLabs Scribe STT API"),
    "deepinfra": (_has_deepinfra_key, _has_deepinfra_key,
                  "STT provider 'deepinfra' configured but DEEPINFRA_API_KEY not set (or openai package missing)",
                  "No local STT available, using DeepInfra Whisper API")}

# Explicit selections whose resolution is more than a probe + warning.
_EXPLICIT_RESOLVERS = {
    "local": _resolve_explicit_local,
    "local_command": _resolve_explicit_local_command,
    "openai": _resolve_explicit_openai}


def _resolve_explicit_provider(provider: str) -> str:
    """Explicit ``stt.provider`` -> usable name or ``"none"``; unknown names pass through untouched
    so the dispatcher fails with the provider-not-registered message."""
    resolver = _EXPLICIT_RESOLVERS.get(provider)
    if resolver is not None:
        return resolver()
    spec = _CLOUD_PROVIDER_SPECS.get(provider)
    if spec is None or spec[0]():
        return provider
    logger.warning(spec[2])
    return "none"


def _get_provider(stt_config: dict) -> str:
    """Which STT provider to use: an explicit ``stt.provider`` is honoured (no silent cloud
    fallback); otherwise auto-detect local > groq > openai > mistral > xai > elevenlabs > deepinfra."""
    if not is_stt_enabled(stt_config):
        return "none"
    explicit = "provider" in stt_config
    provider = stt_config.get("provider", DEFAULT_PROVIDER)
    # The managed "Nous Subscription" selection is the OpenAI backend routed via the managed gateway.
    if isinstance(provider, str) and provider.strip().lower() == "nous":
        provider = "openai"
    if explicit and provider == "local":
        # Legacy DEFAULT_CONFIG seeded ``stt.provider: local`` on every install, so only a
        # raw config.yaml selection counts as explicit; otherwise autodetect (local-first anyway).
        try:
            from tools.tool_backend_helpers import read_selection
            if read_selection("stt") is None:
                explicit = False
        except Exception:  # pragma: no cover — helpers are in-repo
            pass
    if explicit:
        return _resolve_explicit_provider(provider)
    backend = _detect_local_backend()
    if backend:
        return backend
    for name, (_probe, available, _warning, message) in _CLOUD_PROVIDER_SPECS.items():
        if available():
            logger.info(message)
            return name
    return "none"


# ---- Provider: local (faster-whisper) -----------------------------------
def _unload_local_model() -> None:
    """Release the cached local whisper model. Thread-safe via the model lock."""
    global _local_model, _local_model_name
    with _local_model_lock:
        if _local_model is not None:
            logger.info("Unloading local whisper model '%s' after idle timeout", _local_model_name or "unknown")
            _local_model = None
            _local_model_name = None


def _start_idle_unload_watcher(timeout_seconds: int) -> None:
    """Ensure the single idle-unload watcher thread is running. The loop re-reads
    ``stt.local.unload_after_idle_seconds`` every cycle so config edits apply within one interval;
    ``timeout_seconds`` seeds the first cycle so a just-written config is honored even if a
    concurrent read races. Exits after unloading, when the timeout becomes 0, or when the model is gone."""
    global _idle_unload_thread
    with _idle_unload_mgmt_lock:
        if _idle_unload_thread is not None and _idle_unload_thread.is_alive():
            return

        def _watch(initial_timeout=timeout_seconds):
            while not _idle_unload_stop.wait(_IDLE_UNLOAD_CHECK_INTERVAL) and _local_model is not None:
                try:
                    timeout = _get_idle_unload_seconds(_load_stt_config().get("local") or {})
                except Exception:  # noqa: BLE001 - keep the seed value
                    timeout = initial_timeout
                if timeout <= 0:
                    break  # unload disabled mid-flight — stand down
                if time.monotonic() - _last_transcription_time >= timeout:
                    _unload_local_model()
                    break
        _idle_unload_stop.clear()
        _idle_unload_thread = threading.Thread(target=_watch, name="hermes-stt-idle-unload", daemon=True)
        _idle_unload_thread.start()


def _touch_transcription_time() -> None:
    """Record transcription activity (resets the idle timer)."""
    global _last_transcription_time
    _last_transcription_time = time.monotonic()


def _get_or_load_local_model(model_name: str, local_cfg: Dict[str, Any]):
    """Cached faster-whisper model, (re)loaded under a double-checked lock when needed. The returned
    strong reference stays valid even if the idle watcher nulls the global mid-transcription."""
    global _local_model, _local_model_name
    model = _local_model
    # Lazy-load the model (downloads on first use, ~150 MB for 'base'). Double-checked lock: concurrent
    # voice messages must not both download/load the model (#24767). ``model`` is a strong local reference
    # bound under the lock: the idle watcher may null the module global at any time, but this transcription
    # keeps using the instance it grabbed.
    if model is None or _local_model_name != model_name:
        with _local_model_lock:
            if _local_model is None or _local_model_name != model_name:
                logger.info("Loading faster-whisper model '%s' (first load downloads the model)...", model_name)
                # stt.local.device / compute_type pin a configuration where ``auto`` mis-detects.
                _local_model = _load_local_whisper_model(model_name, device=local_cfg.get("device", "auto"),
                                                         compute_type=local_cfg.get("compute_type", "auto"))
                _local_model_name = model_name
            model = _local_model
    return model


def _replace_cached_model_on_cpu(model_name: str):
    """Load *model_name* on CPU/int8 and make it the cached singleton."""
    global _local_model, _local_model_name
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    with _local_model_lock:
        _local_model, _local_model_name = model, model_name
    return model


def _transcribe_local(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe using faster-whisper (local, free)."""
    if not _HAS_FASTER_WHISPER and not _try_lazy_install_stt():
        return _error_result("faster-whisper not installed")
    try:
        stt_config = _load_stt_config()
        local_cfg = stt_config.get("local") or {}
        # Reset the idle timer BEFORE loading so a long in-flight transcription isn't counted as idle.
        _touch_transcription_time()
        model = _get_or_load_local_model(model_name, local_cfg)
        if model is None:  # defensive: load failed without raising
            return _error_result("Local whisper model failed to load")
        # pre_transcription hook overrides win over config-resolved values.
        transcribe_kwargs = build_local_transcribe_kwargs(stt_config)
        transcribe_kwargs.update({k: v for k, v in (("language", language), ("initial_prompt", prompt))
                                  if v})
        try:
            segments, info = model.transcribe(file_path, **transcribe_kwargs)
            # faster-whisper's transcribe() is lazy: the decode (and with it the
            # dlopen-on-first-use of the CUDA runtime on Windows) happens while
            # ITERATING segments, after this call has already returned (#103793).
            # Consume inside the guard so a first-use cuBLAS/cuDNN load failure
            # retries on CPU exactly like a load-time failure does.
            segments = list(segments)
        except Exception as exc:
            # CUDA libs can fail at dlopen-on-first-use, AFTER loading: evict the poisoned
            # cached model, reload on CPU and retry once, else every later message fails.
            if not _looks_like_cuda_lib_error(exc):
                raise
            logger.warning("faster-whisper CUDA runtime failed mid-transcribe (%s) — "
                           "evicting cached model and retrying on CPU (int8).", exc)
            model = _replace_cached_model_on_cpu(model_name)
            segments, info = model.transcribe(file_path, **transcribe_kwargs)
            segments = list(segments)
        transcript = _join_confident_segments(segments, local_cfg)
        logger.info("Transcribed %s via local whisper (%s, lang=%s, %.1fs audio)",
                    Path(file_path).name, model_name, info.language, info.duration)
        _touch_transcription_time()
        idle_timeout = _get_idle_unload_seconds(local_cfg)
        if idle_timeout > 0:
            _start_idle_unload_watcher(idle_timeout)
        return _ok_result(transcript, "local")
    except Exception as e:
        logger.error("Local transcription failed: %s", e, exc_info=True)
        return _error_result(f"Local transcription failed: {e}")


# ---- Public API ---------------------------------------------------------
def _read_block_error(file_path: str) -> Optional[Dict[str, Any]]:
    """Refuse to ship a credential store (auth.json, .env, OAuth tokens) to an STT provider.
    Mirrors the image-gen / video-gen read guards."""
    from agent.file_safety import get_read_block_error
    blocked = get_read_block_error(file_path)
    return _error_result(blocked) if blocked else None


def _transcribe_prepared_audio(
    file_path: str, model: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a validated audio file with the configured STT provider. ``model`` overrides the
    config default; ``source`` is a caller-surface label (``"gateway"``, ``"voice_mode"``) forwarded
    to the ``pre_transcription`` hook only."""
    # Validate before provider resolution so invalid files can't trigger provider setup
    # or lazy installation; the remote-upload size cap applies to non-local only.
    error = _read_block_error(file_path) or _validate_audio_file(file_path, enforce_size_limit=False)
    if error:
        return error
    stt_config = _load_stt_config()
    if not is_stt_enabled(stt_config):
        return _error_result("STT is disabled in config.yaml (stt.enabled: false).")
    provider = _get_provider(stt_config)
    if not _is_local_stt_provider(provider, stt_config):
        error = _validate_audio_file_size(Path(file_path))
        if error:
            return error
        # Convert CAF (iMessage voice notes) to WAV for cloud STT providers.
        if Path(file_path).suffix.lower() == ".caf":
            file_path = _convert_caf_to_wav(file_path)
            if not file_path:
                return _error_result("CAF audio could not be converted to WAV.")
    # Best-effort pre-upload silence trim for built-in cloud providers.
    trim_cleanup_dir: Optional[str] = None
    if provider in CLOUD_STT_PROVIDERS:
        trimmed = _trim_silence_for_cloud_stt(file_path, stt_config)
        if trimmed:
            file_path = trimmed
            trim_cleanup_dir = os.path.dirname(trimmed)
    try:
        return _dispatch_stt_provider(file_path, provider, stt_config, model, source)
    finally:
        if trim_cleanup_dir:
            shutil.rmtree(trim_cleanup_dir, ignore_errors=True)


# Built-in provider -> (stt section, config key, default, treat-empty-as-missing). "local_command"
# shares ``stt.local``; xAI takes no model (logging-only); deepinfra uses the live catalog when empty.
_BUILTIN_MODEL_KEYS = {
    "local": ("local", "model", DEFAULT_LOCAL_MODEL, False),
    "local_command": ("local", "model", DEFAULT_LOCAL_MODEL, False),
    "groq": ("groq", "model", DEFAULT_GROQ_STT_MODEL, True),
    "openai": ("openai", "model", DEFAULT_STT_MODEL, False),
    "mistral": ("mistral", "model", DEFAULT_MISTRAL_STT_MODEL, False),
    "elevenlabs": ("elevenlabs", "model_id", DEFAULT_ELEVENLABS_STT_MODEL, False),
    "deepinfra": ("deepinfra", "model", "", True)}


def _builtin_model_name(provider: str, stt_config: Dict[str, Any], model: Optional[str]) -> str:
    """Resolve the model for a built-in provider: caller override > ``stt.<provider>`` config > default."""
    if model:
        return model
    if provider == "xai":
        return "grok-stt"
    section, key, default, empty_is_missing = _BUILTIN_MODEL_KEYS[provider]
    cfg = _get_stt_section(stt_config, section)
    return (cfg.get(key) or default) if empty_is_missing else cfg.get(key, default)


def _dispatch_stt_provider(
    file_path: str, provider: str, stt_config: Dict[str, Any], model: Optional[str] = None,
    source: Optional[str] = None) -> Dict[str, Any]:
    """Route *file_path* to the handler for *provider* (built-in > command > plugin)."""
    # Static ``stt.prompt`` is the base; hook results mutate on top (last hook to set a field wins).
    prompt = stt_config.get("prompt")
    prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
    # Fires after provider resolution and BEFORE any backend; ``language`` stays None unless a hook sets it.
    model, language, prompt = _apply_pre_transcription_hook(
        file_path=file_path, provider=provider, model=model,
        language=_get_stt_section(stt_config, provider).get("language"), prompt=prompt, source=source,
    )
    prompt = _enforce_prompt_length_limit(prompt, provider)
    if provider in BUILTIN_STT_PROVIDERS:
        # Looked up in this module at call time so tests may patch ``_transcribe_*``.
        handler = globals()[f"_transcribe_{provider}"]
        model_name = _builtin_model_name(provider, stt_config, model)
        if provider in ("local", "local_command"):
            model_name = _normalize_local_model(model_name)
        return handler(file_path, model_name, language=language, prompt=prompt)
    # Command providers: after built-ins (``stt.providers.openai.command`` can't override the
    # real handler) and BEFORE plugins, since config is more local than a plugin install.
    # User-declared command-type provider (``stt.providers.<name>: type: command``). See #17843.
    command_provider_config = _resolve_command_stt_provider_config(provider, stt_config)
    if command_provider_config is not None:
        return _transcribe_command_stt(file_path, provider, command_provider_config, stt_config,
                                       model_override=model, language_override=language, prompt=prompt)
    # Plugin backend: reads ``stt.<provider>`` like built-ins; the ``model`` argument overrides it.
    plugin_result = _dispatch_to_plugin_provider(
        file_path, provider, stt_config, model=model or _get_stt_section(stt_config, provider).get("model"),
        language=language or _resolve_stt_language(provider, stt_config), prompt=prompt)
    return plugin_result if plugin_result is not None else _no_provider_error(provider, stt_config)


def _no_provider_error(provider: str, stt_config: Dict[str, Any]) -> Dict[str, Any]:
    """Error envelope when nothing claimed *provider*: unregistered name > openai selection reason > generic hint."""
    provider_key = str(provider or "").strip().lower()
    if "provider" in stt_config and provider_key and provider_key not in BUILTIN_STT_PROVIDERS and provider_key != "none":
        return _unregistered_stt_provider_error(provider_key)
    # An explicit openai selection flattened to "none" has a specific reason (e.g. managed gateway down).
    # Surface it — with its `hermes tools` remediation — instead of the all-provider setup hint (#93045).
    if provider_key == "none" and str(stt_config.get("provider") or "") == "openai" and _HAS_OPENAI:
        reason = _openai_audio_unavailable_reason()
        if reason is not None:
            return _error_result(reason)
    return _error_result(
        "No STT provider available. Install faster-whisper for free local "
        f"transcription, configure {LOCAL_STT_COMMAND_ENV} or install a local whisper CLI, "
        "set GROQ_API_KEY for free Groq Whisper, set MISTRAL_API_KEY for Mistral "
        "Voxtral Transcribe, configure xAI OAuth or set XAI_API_KEY for xAI Grok STT, "
        "set ELEVENLABS_API_KEY for ElevenLabs Scribe, or set VOICE_TOOLS_OPENAI_KEY "
        "or OPENAI_API_KEY for the OpenAI Whisper API.")


def transcribe_audio(
    file_path: str, model: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Any]:
    """Validate, preprocess supported inputs, and dispatch transcription. ``source`` is a caller-surface
    label (``"gateway"``, ``"voice_mode"``) forwarded to the ``pre_transcription`` hook only."""
    # Secret-store refusal runs before ANY validation so the error names the real reason.
    blocked = _read_block_error(file_path)
    if blocked:
        return blocked
    # Cap .silk sources before the decoder runs; for other inputs the upload cap is
    # provider-scoped in _transcribe_prepared_audio so local whisper can take big files.
    is_silk = Path(file_path).suffix.lower() == ".silk"
    source_error = _validate_audio_source_file(file_path, enforce_size_limit=is_silk)
    if source_error:
        return source_error
    prepared_path, cleanup_dir, prep_error = _prepare_audio_for_transcription(file_path)
    if prep_error or prepared_path is None:
        return prep_error or _error_result("Audio preprocessing did not produce a file for transcription.")
    try:
        return (_validate_audio_file(prepared_path, enforce_size_limit=False)
                or _transcribe_prepared_audio(prepared_path, model, source))
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def transcribe_audio_local_fallback(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Try an already-installed local STT backend without changing config: passive inbound-media
    recovery after the configured provider failed — never lazy-installs or falls through to cloud."""
    error = _validate_audio_file(file_path)
    if error:
        return error
    local_model = model or (_load_stt_config().get("local") or {}).get("model", DEFAULT_LOCAL_MODEL)
    if _HAS_FASTER_WHISPER:
        return _transcribe_local(file_path, _normalize_local_model(local_model))
    if _has_local_command():
        return _transcribe_local_command(file_path, _normalize_local_model(local_model))
    return _error_result("No installed local STT backend is available.", provider="local")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import platform  # noqa: F401,E402
import queue  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import tempfile  # noqa: F401,E402
from urllib.parse import urljoin  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'COMMAND_STT_OUTPUT_FORMATS': ('tools.transcription_command', 'COMMAND_STT_OUTPUT_FORMATS'),
    'COMMON_LOCAL_BIN_DIRS': ('tools.transcription_common', 'COMMON_LOCAL_BIN_DIRS'),
    'DEFAULT_COMMAND_STT_LANGUAGE': ('tools.transcription_command', 'DEFAULT_COMMAND_STT_LANGUAGE'),
    'DEFAULT_COMMAND_STT_OUTPUT_FORMAT': ('tools.transcription_command', 'DEFAULT_COMMAND_STT_OUTPUT_FORMAT'),
    'DEFAULT_COMMAND_STT_TIMEOUT_SECONDS': ('tools.transcription_command', 'DEFAULT_COMMAND_STT_TIMEOUT_SECONDS'),
    'DEFAULT_LOCAL_STT_LANGUAGE': ('tools.transcription_common', 'DEFAULT_LOCAL_STT_LANGUAGE'),
    'ELEVENLABS_STT_BASE_URL': ('tools.transcription_common', 'ELEVENLABS_STT_BASE_URL'),
    'GROQ_BASE_URL': ('tools.transcription_common', 'GROQ_BASE_URL'),
    'GROQ_MODELS': ('tools.transcription_common', 'GROQ_MODELS'),
    'LOCAL_NATIVE_AUDIO_FORMATS': ('tools.transcription_common', 'LOCAL_NATIVE_AUDIO_FORMATS'),
    'MAX_FILE_SIZE': ('tools.transcription_common', 'MAX_FILE_SIZE'),
    'OPENAI_BASE_URL': ('tools.transcription_common', 'OPENAI_BASE_URL'),
    'OPENAI_MODELS': ('tools.transcription_common', 'OPENAI_MODELS'),
    'SUPPORTED_FORMATS': ('tools.transcription_common', 'SUPPORTED_FORMATS'),
    'XAI_STT_BASE_URL': ('tools.transcription_common', 'XAI_STT_BASE_URL'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'resolve_managed_tool_gateway': ('tools.managed_tool_gateway', 'resolve_managed_tool_gateway'),
    'resolve_openai_audio_api_key': ('tools.tool_backend_helpers', 'resolve_openai_audio_api_key'),
    'windows_hide_flags': ('hermes_cli._subprocess_compat', 'windows_hide_flags'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
