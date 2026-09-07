#!/usr/bin/env python3
"""Text-to-speech tool: config resolution, built-in provider dispatch, output policy, registration.

Built-ins: Edge (free default), ElevenLabs, OpenAI, DeepInfra, MiniMax, Mistral, Gemini, xAI,
local NeuTTS / KittenTTS / Piper; plus ``type: command`` providers under ``tts.providers.<name>``
and plugin-registered ones. Output is Opus (.ogg) on voice-bubble platforms, MP3 elsewhere.
Sibling ``tts_tool_*`` modules hold backends/delivery/lifecycle; they read the seams defined
here (config, provider resolution, lazy SDK importers) through ``_origin()`` at call time.
"""

import asyncio
import contextlib
import datetime
import importlib.util
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)


def get_env_value(name, default=None):
    """Read env values through the live config module (resolved per call so test patches apply)."""
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve a TTS provider API key via the shared voice-key resolver (config > env/.env > pool)."""
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helpers are in-repo
        return str(get_env_value(env_var) or "").strip()
    return resolve_provider_secret(env_var, provider_id, env_getter=get_env_value)


from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS, _configured_command_tts_output_path, _generate_command_tts,
    _get_command_tts_output_format, _is_command_tts_voice_compatible, _resolve_command_provider_config)
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
from tools.tts_tool_delivery import (
    _resolve_max_text_length, _build_audio_delivery_files, _convert_to_opus, _remove_quietly,
    _repair_ogg_container, _resolve_audio_delivery_profile, _split_text_for_tts)
from tools.tts_tool_providers import (
    _generate_edge_tts, _generate_elevenlabs, _generate_gemini_tts, _generate_minimax_tts,
    _generate_mistral_tts, _generate_xai_tts, _resolve_minimax_tts_runtime)
from tools.tts_tool_local import _generate_kittentts, _generate_neutts, _generate_piper_tts
from tools.tts_tool_plugins import (
    _dispatch_to_plugin_provider, _plugin_provider_is_available,
    _plugin_provider_is_voice_compatible)
from tools.tts_tool_openai import _generate_deepinfra_tts, _generate_openai_tts, _has_openai_audio_backend


# --- Lazy SDK importers -- providers import only when used (headless boxes lack PortAudio etc.) ---
def _sdk_importer(module: str, attr: Optional[str] = None, feature: Optional[str] = None) -> Callable[[], Any]:
    """Lazy SDK importer: returns ``module`` (or ``module.attr``), raising ImportError when absent.

    ``feature`` names a ``tools.lazy_deps`` feature to best-effort install first (users who enabled
    a provider in config.yaml never ran the post-setup hook); any failure there falls through so
    the raw import still raises cleanly. sounddevice also raises OSError without PortAudio."""
    def _import():
        if feature:
            with contextlib.suppress(Exception):
                from tools.lazy_deps import ensure
                ensure(feature, prompt=False)
        mod = importlib.import_module(module)
        return getattr(mod, attr) if attr else mod
    _import.__name__ = f"_import_{module.split('.')[0]}"
    return _import


_import_edge_tts = _sdk_importer("edge_tts", feature="tts.edge")
_import_elevenlabs = _sdk_importer("elevenlabs.client", "ElevenLabs", feature="tts.elevenlabs")
_import_openai_client = _sdk_importer("openai", "OpenAI")
_import_mistral_client = _sdk_importer("mistralai.client", "Mistral", feature="tts.mistral")
_import_sounddevice = _sdk_importer("sounddevice")
_import_kittentts = _sdk_importer("kittentts", "KittenTTS")
_import_piper = _sdk_importer("piper", "PiperVoice")  # piper-tts wheels embed espeak-ng


def _importable(importer: Callable[[], Any]) -> bool:
    try:
        importer()
        return True
    except ImportError:
        return False


def _package_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _check_neutts_available() -> bool: return _package_installed("neutts")
def _check_kittentts_available() -> bool: return _package_installed("kittentts")
def _check_piper_available() -> bool: return _package_installed("piper")


# --- Defaults / config ---
DEFAULT_PROVIDER = "edge"


def _get_default_output_dir() -> str:
    from hermes_constants import get_hermes_dir
    return str(get_hermes_dir("cache/audio", "audio_cache"))


DEFAULT_OUTPUT_DIR = _DEFAULT_OUTPUT_DIR_AT_IMPORT = _get_default_output_dir()


def _default_output_dir() -> str:
    """The active profile's audio output dir at call time (long-lived runtimes switch profiles
    after import); a monkeypatched ``DEFAULT_OUTPUT_DIR`` wins.

    Same bug class as skills_tool (f8723c478) and skills_sync (#65828): long-lived multi-profile runtimes
    (dashboard console, TUI/Desktop backend, cron, kanban workers) import this module once under the launch
    HERMES_HOME and later scope requests to a different profile via
    ``hermes_constants.set_hermes_home_override()`` — a frozen module constant keeps writing synthesized
    audio into the launch profile's cache instead of the active profile's (#98749). Keep the legacy
    ``DEFAULT_OUTPUT_DIR`` module attribute for tests and external patchers; when it has not been patched,
    re-resolve from the live profile-scoped HERMES_HOME on every call.
    """
    if DEFAULT_OUTPUT_DIR != _DEFAULT_OUTPUT_DIR_AT_IMPORT:
        return DEFAULT_OUTPUT_DIR
    return _get_default_output_dir()


def _load_tts_config() -> Dict[str, Any]:
    """Return the ``tts`` config section ({} when unavailable)."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("tts") or {}
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
    return {}


def _get_provider(tts_config: Dict[str, Any]) -> str:
    """Configured provider or the free default (inference credentials never imply consent to paid
    speech); ``nous`` is serviced by the OpenAI path through the managed openai-audio gateway."""
    provider = (tts_config.get("provider") or DEFAULT_PROVIDER).lower().strip()
    return "openai" if provider == NOUS_MANAGED_PROVIDER else provider


# Platforms whose native voice-bubble delivery requires Ogg/Opus (MP3 renders broken there).
OPUS_VOICE_PLATFORMS = frozenset({"telegram", "matrix", "feishu", "whatsapp", "signal"})
# Built-ins that emit Opus natively when asked for .ogg; the rest need ffmpeg for voice bubbles.
_NATIVE_OPUS_PROVIDERS = frozenset({"openai", "elevenlabs", "mistral", "gemini"})
_FFMPEG_OPUS_PROVIDERS = frozenset({"edge", "neutts", "minimax", "xai", "kittentts", "piper"})


# --- Built-in provider dispatch ---
# provider -> (availability predicate or None, log label, generator name, "package missing" error).
# Predicates/generator names resolve module globals at call time so test monkeypatches apply.
_BUILTIN_DISPATCH: Dict[str, tuple] = {
    "elevenlabs": (lambda: _importable(_import_elevenlabs), "ElevenLabs", "_generate_elevenlabs",
                   "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs"),
    "openai": (lambda: _importable(_import_openai_client), "OpenAI TTS", "_generate_openai_tts",
               "OpenAI provider selected but 'openai' package not installed."),
    "deepinfra": (lambda: _importable(_import_openai_client), "DeepInfra TTS", "_generate_deepinfra_tts",
                  "DeepInfra TTS uses the 'openai' SDK but it isn't installed."),
    "minimax": (None, "MiniMax TTS", "_generate_minimax_tts", None),
    "xai": (None, "xAI TTS", "_generate_xai_tts", None),
    "mistral": (lambda: _importable(_import_mistral_client), "Mistral Voxtral TTS", "_generate_mistral_tts",
                "Mistral provider selected but 'mistralai' package not installed. "
                "Run `hermes setup` to install Mistral support."),
    "gemini": (None, "Google Gemini TTS", "_generate_gemini_tts", None),
    "neutts": (lambda: _check_neutts_available(), "NeuTTS (local)", "_generate_neutts",
               "NeuTTS provider selected but neutts is not installed. "
               "Run hermes setup and choose NeuTTS, or install espeak-ng and run python -m pip install -U neutts[all]."),
    "kittentts": (lambda: _importable(_import_kittentts), "KittenTTS (local, ~25MB)", "_generate_kittentts",
                  "KittenTTS provider selected but 'kittentts' package not installed. "
                  "Run 'hermes setup tts' and choose KittenTTS, or install manually: "
                  "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"),
    "piper": (lambda: _importable(_import_piper), "Piper (local)", "_generate_piper_tts",
              "Piper provider selected but 'piper-tts' package not installed. "
              "Run 'hermes tools' and select Piper under TTS, or install manually: "
              "pip install piper-tts")}


def _error_json(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _run_edge_tts(text: str, file_str: str, tts_config: Dict[str, Any]) -> None:
    """Run the async Edge generator from sync code (worker thread; direct run if that fails)."""
    run = lambda: asyncio.run(_generate_edge_tts(text, file_str, tts_config))  # noqa: E731
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=60)
    except RuntimeError:
        run()


def _select_builtin_engine(provider: str) -> tuple:
    """SDK check -> ``(engine, None)`` or ``(provider, error_json)``. Unknown names take the Edge
    default; without edge-tts NeuTTS is the fallback (engine != provider)."""
    entry = _BUILTIN_DISPATCH.get(provider)
    if entry is not None:
        available, _label, _generator, missing_error = entry
        return provider, (_error_json(missing_error) if available is not None and not available() else None)
    if _importable(_import_edge_tts):
        return provider, None  # Edge default; the reported provider stays as configured
    if _check_neutts_available():
        logger.info("Edge TTS not available, falling back to NeuTTS (local)...")
        return "neutts", None
    return provider, _error_json(
        "No TTS provider available. Install edge-tts (pip install edge-tts) "
        "or set up NeuTTS for local synthesis.")


def _synthesize_builtin(engine: str, text: str, file_str: str, tts_config: Dict[str, Any], instructions: Optional[str]) -> None:
    """Run the already-selected built-in *engine*."""
    entry = _BUILTIN_DISPATCH.get(engine)
    logger.info("Generating speech with %s...", entry[1] if entry else "Edge TTS")
    if entry is None:
        _run_edge_tts(text, file_str, tts_config)
    elif engine == "openai":
        _generate_openai_tts(text, file_str, tts_config, instructions=instructions)
    else:
        globals()[entry[2]](text, file_str, tts_config)


def _finalize_voice_delivery(
    file_str: str, provider: str, command_provider_config: Optional[Dict[str, Any]], want_opus: bool,
) -> tuple:
    """Voice-bubble eligibility (Opus-converting when needed) -> ``(path, voice_compatible)``.

    Command/plugin providers are documents unless they opt in via ``voice_compatible``; native-Opus
    built-ins qualify when the platform wants Opus and they wrote .ogg; MP3/WAV built-ins are
    ffmpeg-converted only when the platform needs Opus."""
    if command_provider_config is not None:
        opted_in = _is_command_tts_voice_compatible(command_provider_config)
    elif provider not in BUILTIN_TTS_PROVIDERS:
        opted_in = _plugin_provider_is_voice_compatible(provider)
    elif want_opus and provider in _FFMPEG_OPUS_PROVIDERS and not file_str.endswith(".ogg"):
        opus_path = _convert_to_opus(file_str)
        return (opus_path, True) if opus_path else (file_str, False)
    else:
        native = provider in _NATIVE_OPUS_PROVIDERS
        return file_str, native and want_opus and file_str.endswith(".ogg")
    if not opted_in:
        return file_str, False
    # Plugin-registered provider (issue #30398). Voice-bubble delivery opts in via
    # ``TTSProvider.voice_compatible`` (mirrors the command-provider opt-in). Plugins that already write
    # Opus skip the ffmpeg conversion.
    if not file_str.endswith(".ogg"):
        file_str = _convert_to_opus(file_str) or file_str
    return file_str, file_str.endswith(".ogg")


# --- Main tool function ---
def _apply_call_overrides(tts_config: Dict[str, Any], speed: Optional[float], provider: Optional[str]):
    """Apply per-call ``speed`` (clamped, on a shallow copy so the cached config isn't mutated) and
    resolve the provider name."""
    if speed is not None:
        tts_config = {**tts_config, "speed": max(0.25, min(4.0, float(speed)))}
    return tts_config, provider.lower().strip() if provider else _get_provider(tts_config)


def _session_platform() -> tuple:
    """``(platform, wants_opus)`` — platforms delivering voice bubbles only as Ogg/Opus want Opus."""
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    return platform, platform in OPUS_VOICE_PLATFORMS


def _resolve_output_base(
    output_path: Optional[str], provider: str, command_provider_config: Optional[Dict[str, Any]], want_opus: bool,
) -> tuple:
    """Pick the output file -> ``(Path, None)`` or ``(None, error_json)``.

    A caller path is rejected on ``..`` traversal (bug or prompt-injection; absolute is fine) and
    on protected credential/system locations. Default ``<audio cache>/tts_<timestamp>.<ext>``: the
    command format, ``.ogg`` for native-Opus providers on Opus platforms, else ``.mp3``."""
    if output_path:
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return None, _error_json(
                f"output_path contains '..' traversal component: {output_path}. "
                "Use an absolute path or one relative to the current directory without '..'.")
        file_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            file_path = _configured_command_tts_output_path(file_path, command_provider_config)
        from agent.file_safety import is_write_approval_required, is_write_denied
        if is_write_denied(str(file_path)) or is_write_approval_required(str(file_path)):
            return None, _error_json(
                f"output_path targets a protected credential or system path: "
                f"{file_path}. Choose a normal audio output location.")
    else:
        if command_provider_config is not None:
            ext = _get_command_tts_output_format(command_provider_config)
        else:
            ext = "ogg" if want_opus and provider in _NATIVE_OPUS_PROVIDERS else "mp3"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = Path(_default_output_dir()) / f"tts_{timestamp}.{ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path, None


def _media_tag(paths: List[str], voice_compatible: bool) -> str:
    """``MEDIA:<path>`` lines; the ``[[audio_as_voice]]`` marker asks the platform for a voice bubble."""
    media_tag = "\n".join(f"MEDIA:{path}" for path in paths)
    return f"[[audio_as_voice]]\n{media_tag}" if voice_compatible else media_tag


def _tool_failure(prefix: str, provider: str, exc: BaseException) -> str:
    """Log and wrap a synthesis failure as the standard error envelope (traceback except for config errors)."""
    error_msg = f"{prefix} ({provider}): {exc}"
    logger.error("%s", error_msg, exc_info=not isinstance(exc, ValueError))
    return tool_error(error_msg, success=False)


def _text_to_speech_single(
    text: str, file_str: str, *, provider: str, tts_config: Dict[str, Any],
    command_provider_config: Optional[Dict[str, Any]], want_opus: bool, instructions: Optional[str],
) -> str:
    """Synthesize one provider-safe chunk into *file_str*; returns the result envelope.

    Command providers resolve BEFORE built-in dispatch, but built-in names short-circuit so
    ``tts.providers.openai.command`` can't shadow OpenAI. Plugins fire only for names that are
    neither; a None return falls through to built-in dispatch (unknown -> Edge default)."""
    try:
        if command_provider_config is not None:
            logger.info("Generating speech with command TTS provider '%s'...", provider)
            file_str = _generate_command_tts(
                text, file_str, provider, command_provider_config, tts_config)
        # Plugin-registered TTS backend (issue #30398). Fires when the configured provider is neither a
        # built-in nor a command-type entry, AND a plugin is registered under that name. The walrus binds
        # `_plugin_path` only when the dispatcher returns a path (i.e. a plugin was actually found); a None
        # return falls through to the built-in elif chain so unknown names hit the Edge TTS default at the
        # bottom. The dispatcher itself enforces built-ins-always-win + command-wins-over-plugin
        # defensively.
        elif provider not in BUILTIN_TTS_PROVIDERS and (
            _plugin_path := _dispatch_to_plugin_provider(text, file_str, provider, tts_config)
        ) is not None:
            file_str = _plugin_path
        else:
            provider, error = _select_builtin_engine(provider)
            if error:
                return error
            _synthesize_builtin(provider, text, file_str, tts_config, instructions)
        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return _error_json(f"TTS generation produced no output (provider: {provider})")

        # Sniff once for every provider: MP3/WAV bytes in a .ogg path render as 0-second bubbles.
        file_str = _repair_ogg_container(file_str)
        file_str, voice_compatible = _finalize_voice_delivery(
            file_str, provider, command_provider_config, want_opus)
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{os.path.getsize(file_str):,}", provider)
        return json.dumps({
            "success": True, "file_path": file_str, "media_tag": _media_tag([file_str], voice_compatible),
            "provider": provider, "voice_compatible": voice_compatible,
        }, ensure_ascii=False)
    except ValueError as e:
        return _tool_failure("TTS configuration error", provider, e)
    except FileNotFoundError as e:
        return _tool_failure("TTS dependency missing", provider, e)
    except Exception as e:
        return _tool_failure("TTS generation failed", provider, e)


class _ChunkFailed(Exception):
    """One chunk's synthesis returned an error envelope; message is the final tool error text."""


def _synthesize_chunks(chunks: List[str], base_path: Path, generated_artifacts: set, **single_kwargs) -> tuple:
    """Synthesize chunks into ``<base>.chunkNNN<ext>`` (or ``base`` alone) -> ``(encoded_paths, results)``.

    Every touched path lands in *generated_artifacts* for the caller's sweep. Raises
    :class:`_ChunkFailed` on a reported failure, ``RuntimeError`` on garbage or missing audio."""
    provider = single_kwargs["provider"]
    encoded_paths: List[str] = []
    chunk_results: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = base_path
        if len(chunks) > 1:
            chunk_path = base_path.with_name(f"{base_path.stem}.chunk{index:03d}{base_path.suffix}")
        generated_artifacts.add(str(chunk_path))
        raw_result = _text_to_speech_single(chunk, str(chunk_path), **single_kwargs)
        try:
            chunk_result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError(f"TTS chunk {index} returned invalid JSON: {str(raw_result)[:200]}")
        if not chunk_result.get("success"):
            error_msg = chunk_result.get("error", "unknown error")
            raise _ChunkFailed(f"TTS chunk {index} failed ({provider}): {error_msg}")
        actual_path = str(chunk_result.get("file_path") or chunk_path)
        if not os.path.isfile(actual_path) or os.path.getsize(actual_path) <= 0:
            raise RuntimeError(f"TTS chunk {index} produced no final audio: {actual_path}")
        generated_artifacts.add(actual_path)
        encoded_paths.append(actual_path)
        chunk_results.append(chunk_result)
    return encoded_paths, chunk_results


def text_to_speech_tool(
    text: str, output_path: Optional[str] = None, speed: Optional[float] = None,
    instructions: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Convert text to speech with long-form chunking; returns the JSON result envelope.

    Text is normalized, split into provider-safe chunks (never silently truncated), synthesized
    sequentially, then packed against the platform's upload limit: a failed combine keeps the
    separate valid files and no over-limit artifact is ever returned."""
    if not text or not text.strip():
        return tool_error("Text is required", success=False)
    try:  # shared cleaner: markdown, emoji, think blocks, verifier footer, units, newlines
        from tools.tts_text_normalize import prepare_spoken_text
        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)
    tts_config, provider = _apply_call_overrides(_load_tts_config(), speed, provider)
    command_provider_config = _resolve_command_provider_config(provider, tts_config)
    max_len = _resolve_max_text_length(provider, tts_config)
    chunks = _split_text_for_tts(text, max_len)
    if not chunks:
        return tool_error("Text is required", success=False)
    if len(chunks) > 1:
        logger.info("TTS text for provider %s split into %d chunks (input=%d chars, cap=%d)",
                    provider, len(chunks), len(text), max_len)
    platform, want_opus = _session_platform()
    delivery_profile = _resolve_audio_delivery_profile(platform, tts_config)
    base_path, error = _resolve_output_base(
        output_path, provider, command_provider_config, want_opus)
    if error:
        return error
    generated_artifacts: set[str] = set()
    final_paths: List[str] = []
    try:
        encoded_paths, chunk_results = _synthesize_chunks(
            chunks, base_path, generated_artifacts, provider=provider, tts_config=tts_config,
            command_provider_config=command_provider_config, want_opus=want_opus,
            instructions=instructions)
        voice_compatible = bool(chunk_results) and all(bool(r.get("voice_compatible")) for r in chunk_results)
        delivery_base = base_path.with_suffix(Path(encoded_paths[0]).suffix)
        final_paths, combined_chunks = _build_audio_delivery_files(
            encoded_paths, str(delivery_base), delivery_profile, voice_compatible=voice_compatible)
        for path in final_paths:
            logger.info("TTS audio saved: %s (%s bytes, provider: %s)", path, f"{os.path.getsize(path):,}", provider)
        return json.dumps({
            "success": True, "file_path": final_paths[0], "file_paths": final_paths,
            "media_tag": _media_tag(final_paths, voice_compatible),
            "provider": chunk_results[0].get("provider", provider), "voice_compatible": voice_compatible,
            "chunk_count": len(chunks), "delivery_file_count": len(final_paths),
            "combined_chunks": bool(combined_chunks),
            "delivery_profile": {
                "platform": delivery_profile.platform, "max_file_bytes": delivery_profile.max_file_bytes,
                "target_file_bytes": delivery_profile.target_file_bytes},
        }, ensure_ascii=False)
    except _ChunkFailed as exc:
        return tool_error(str(exc), success=False)
    except ValueError as exc:
        return _tool_failure("TTS delivery error", provider, exc)
    except Exception as exc:
        return _tool_failure("TTS long-form generation failed", provider, exc)
    finally:
        final_absolute = {os.path.abspath(path) for path in final_paths}
        for artifact in generated_artifacts:
            if os.path.abspath(artifact) not in final_absolute:
                _remove_quietly(artifact)


# --- check_fn ---
def _minimax_requirements() -> bool:
    try:
        _resolve_minimax_tts_runtime(_load_tts_config())
        return True
    except ValueError:
        return False


def _xai_requirements() -> bool:
    try:
        from tools.xai_http import resolve_xai_http_credentials
        return bool(resolve_xai_http_credentials().get("api_key"))
    except Exception:
        return False


# Must mirror text_to_speech_tool dispatch: unrelated cloud credentials never make the Edge
# default usable, and an explicit provider is checked on its own.
_BUILTIN_REQUIREMENTS: Dict[str, Callable[[], bool]] = {
    "edge": lambda: _importable(_import_edge_tts) or _check_neutts_available(),
    "elevenlabs": lambda: _importable(_import_elevenlabs) and bool(_resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")),
    "openai": lambda: _package_installed("openai") and _has_openai_audio_backend(),
    "deepinfra": lambda: _package_installed("openai") and bool(_resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")),
    "minimax": _minimax_requirements,
    "xai": _xai_requirements,
    "gemini": lambda: bool(_resolve_provider_key("GEMINI_API_KEY", "gemini") or _resolve_provider_key("GOOGLE_API_KEY", "gemini")),
    "mistral": lambda: _importable(_import_mistral_client) and bool(_resolve_provider_key("MISTRAL_API_KEY", "mistral")),
    "neutts": lambda: _check_neutts_available(),
    "kittentts": lambda: _check_kittentts_available(),
    "piper": lambda: _check_piper_available()}


def check_tts_requirements() -> bool:
    """Return whether the explicitly resolved TTS provider can run."""
    tts_config = _load_tts_config()
    provider = _get_provider(tts_config)
    if _resolve_command_provider_config(provider, tts_config) is not None:
        return True
    check = _BUILTIN_REQUIREMENTS.get(provider)
    return check() if check is not None else _plugin_provider_is_available(provider)


# --- Registry ---
from tools.registry import registry, tool_error

TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured (built-in providers like edge/openai or custom command providers under tts.providers.<name>), not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. Provider-specific per-request character caps apply automatically (OpenAI 4096, xAI 15000, MiniMax 10000, ElevenLabs 5k-40k depending on model); longer input is split into ordered chunks without silent truncation."
            },
            "output_path": {
                "type": "string",
                "description": f"Optional custom file path to save the audio. Defaults to {display_hermes_home()}/audio_cache/<timestamp>.mp3"
            },
            "speed": {
                "type": "number",
                "description": "Playback speed multiplier. 1.0 = normal, 0.5 = very slow (language learning), 2.0 = fast. Range: 0.25-4.0. Overrides the speed configured in config.yaml."
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional voice-design guidance: tone, emotion, pacing, accent, "
                    "whispering, impressions (e.g. 'Speak in a cheerful, excited whisper'). "
                    "Forwarded to the OpenAI backend (gpt-4o-mini-tts and OpenAI-compatible "
                    "voice-design servers). Silently ignored by backends that don't support it."
                )
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional TTS provider override. Accepts built-in names "
                    "(edge, openai, elevenlabs, minimax, xai, mistral, gemini, "
                    "neutts, kittentts, piper), user-declared command provider "
                    "names from tts.providers.<name>, or plugin-registered names. "
                    "When omitted, the configured tts.provider from config.yaml is used."
                )
            }
        },
        "required": ["text"]
    }
}

registry.register(
    name="text_to_speech",
    toolset="tts",
    schema=TTS_SCHEMA,
    handler=lambda args, **kw: text_to_speech_tool(
        text=args.get("text", ""),
        **{k: args.get(k) for k in ("output_path", "speed", "instructions", "provider")}),
    check_fn=check_tts_requirements,
    emoji="🔊")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from concurrent.futures import Future  # noqa: F401,E402
from typing import Iterator  # noqa: F401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
import base64  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import platform  # noqa: F401,E402
import queue  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import shutil  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import threading  # noqa: F401,E402
import time  # noqa: F401,E402
from urllib.parse import urljoin  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402
import uuid  # noqa: F401,E402

GEMINI_TTS_CHANNELS = 1

GEMINI_TTS_SAMPLE_RATE = 24000

GEMINI_TTS_SAMPLE_WIDTH = 2  # 16-bit PCM (L16)

FALLBACK_MAX_TEXT_LENGTH = 4000

MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


_PLUGIN_COMPAT_LAZY = {
    'AudioDeliveryProfile': ('tools.tts_tool_delivery', 'AudioDeliveryProfile'),
    'COMMAND_TTS_OUTPUT_FORMATS': ('tools.tts_command_provider', 'COMMAND_TTS_OUTPUT_FORMATS'),
    'DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH': ('tools.tts_command_provider', 'DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH'),
    'DEFAULT_COMMAND_TTS_OUTPUT_FORMAT': ('tools.tts_command_provider', 'DEFAULT_COMMAND_TTS_OUTPUT_FORMAT'),
    'DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS': ('tools.tts_command_provider', 'DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS'),
    'DEFAULT_DEEPINFRA_TTS_VOICE': ('tools.tts_tool_openai', 'DEFAULT_DEEPINFRA_TTS_VOICE'),
    'DEFAULT_EDGE_VOICE': ('tools.tts_tool_providers', 'DEFAULT_EDGE_VOICE'),
    'DEFAULT_ELEVENLABS_MODEL_ID': ('tools.tts_tool_providers', 'DEFAULT_ELEVENLABS_MODEL_ID'),
    'DEFAULT_ELEVENLABS_STREAMING_MODEL_ID': ('tools.tts_tool_providers', 'DEFAULT_ELEVENLABS_STREAMING_MODEL_ID'),
    'DEFAULT_ELEVENLABS_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_ELEVENLABS_VOICE_ID'),
    'DEFAULT_GEMINI_AUDIO_TAGS': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_AUDIO_TAGS'),
    'DEFAULT_GEMINI_TTS_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_TTS_BASE_URL'),
    'DEFAULT_GEMINI_TTS_MODEL': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_TTS_MODEL'),
    'DEFAULT_GEMINI_TTS_VOICE': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_TTS_VOICE'),
    'DEFAULT_KITTENTTS_MODEL': ('tools.tts_tool_local', 'DEFAULT_KITTENTTS_MODEL'),
    'DEFAULT_KITTENTTS_VOICE': ('tools.tts_tool_local', 'DEFAULT_KITTENTTS_VOICE'),
    'DEFAULT_MINIMAX_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_BASE_URL'),
    'DEFAULT_MINIMAX_CN_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_CN_BASE_URL'),
    'DEFAULT_MINIMAX_MODEL': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_MODEL'),
    'DEFAULT_MINIMAX_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_VOICE_ID'),
    'DEFAULT_MISTRAL_TTS_MODEL': ('tools.tts_tool_providers', 'DEFAULT_MISTRAL_TTS_MODEL'),
    'DEFAULT_MISTRAL_TTS_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_MISTRAL_TTS_VOICE_ID'),
    'DEFAULT_OPENAI_BASE_URL': ('tools.tts_tool_openai', 'DEFAULT_OPENAI_BASE_URL'),
    'DEFAULT_OPENAI_MODEL': ('tools.tts_tool_openai', 'DEFAULT_OPENAI_MODEL'),
    'DEFAULT_OPENAI_VOICE': ('tools.tts_tool_openai', 'DEFAULT_OPENAI_VOICE'),
    'DEFAULT_PIPER_VOICE': ('tools.tts_tool_local', 'DEFAULT_PIPER_VOICE'),
    'DEFAULT_XAI_AUTO_SPEECH_TAGS': ('tools.tts_tool_providers', 'DEFAULT_XAI_AUTO_SPEECH_TAGS'),
    'DEFAULT_XAI_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_XAI_BASE_URL'),
    'DEFAULT_XAI_BIT_RATE': ('tools.tts_tool_providers', 'DEFAULT_XAI_BIT_RATE'),
    'DEFAULT_XAI_LANGUAGE': ('tools.tts_tool_providers', 'DEFAULT_XAI_LANGUAGE'),
    'DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT': ('tools.tts_tool_providers', 'DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT'),
    'DEFAULT_XAI_SAMPLE_RATE': ('tools.tts_tool_providers', 'DEFAULT_XAI_SAMPLE_RATE'),
    'DEFAULT_XAI_SPEED_DEFAULT': ('tools.tts_tool_providers', 'DEFAULT_XAI_SPEED_DEFAULT'),
    'DEFAULT_XAI_SPEED_MAX': ('tools.tts_tool_providers', 'DEFAULT_XAI_SPEED_MAX'),
    'DEFAULT_XAI_SPEED_MIN': ('tools.tts_tool_providers', 'DEFAULT_XAI_SPEED_MIN'),
    'DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT': ('tools.tts_tool_providers', 'DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT'),
    'DEFAULT_XAI_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_XAI_VOICE_ID'),
    'ELEVENLABS_MODEL_MAX_TEXT_LENGTH': ('tools.tts_tool_delivery', 'ELEVENLABS_MODEL_MAX_TEXT_LENGTH'),
    'FALLBACK_MAX_TEXT_LENGTH': ('tools.tts_tool_delivery', 'FALLBACK_MAX_TEXT_LENGTH'),
    'GEMINI_AUDIO_TAG_REWRITE_TASK': ('tools.tts_tool_providers', 'GEMINI_AUDIO_TAG_REWRITE_TASK'),
    'MANAGED_OPENAI_TTS_MODELS': ('tools.tts_tool_openai', 'MANAGED_OPENAI_TTS_MODELS'),
    'PROVIDER_MAX_TEXT_LENGTH': ('tools.tts_tool_delivery', 'PROVIDER_MAX_TEXT_LENGTH'),
    'TTS_RESPONSE_BODY_CHUNK_BYTES': ('tools.tts_tool_providers', 'TTS_RESPONSE_BODY_CHUNK_BYTES'),
    'TTS_RESPONSE_BODY_LIMIT_BYTES': ('tools.tts_tool_providers', 'TTS_RESPONSE_BODY_LIMIT_BYTES'),
    'acquire_tts_lease': ('tools.tts_tool_lifecycle', 'acquire_tts_lease'),
    'hermes_xai_user_agent': ('tools.xai_http', 'hermes_xai_user_agent'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'read_selection': ('tools.tool_backend_helpers', 'read_selection'),
    'release_tts_lease': ('tools.tts_tool_lifecycle', 'release_tts_lease'),
    'release_tts_provider': ('tools.tts_tool_lifecycle', 'release_tts_provider'),
    'resolve_managed_tool_gateway': ('tools.managed_tool_gateway', 'resolve_managed_tool_gateway'),
    'resolve_openai_audio_api_key': ('tools.tool_backend_helpers', 'resolve_openai_audio_api_key'),
    'selection_error': ('tools.tool_backend_helpers', 'selection_error'),
    'stream_tts_to_speaker': ('tools.tts_tool_speaker', 'stream_tts_to_speaker'),
    'tts_lease_holders': ('tools.tts_tool_lifecycle', 'tts_lease_holders'),
    'warm_tts_provider': ('tools.tts_tool_lifecycle', 'warm_tts_provider'),
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
