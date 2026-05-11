#!/usr/bin/env python3
"""
Text-to-Speech Tool Module

Built-in TTS providers:
- Edge TTS (default, free, no API key): Microsoft Edge neural voices
- ElevenLabs (premium): High-quality voices, needs ELEVENLABS_API_KEY
- OpenAI TTS: Good quality, needs OPENAI_API_KEY
- MiniMax TTS: High-quality with voice cloning, needs MINIMAX_API_KEY
- Mistral (Voxtral TTS): Multilingual, native Opus, needs MISTRAL_API_KEY
- Google Gemini TTS: Controllable, 30 prebuilt voices, needs GEMINI_API_KEY
- xAI TTS: Grok voices, needs XAI_API_KEY
- NeuTTS (local, free, no API key): On-device TTS via neutts
- KittenTTS (local, free, no API key): On-device 25MB model
- Piper (local, free, no API key): OHF-Voice/piper1-gpl neural VITS, 44 languages

Custom command providers:
- Users can declare any number of named providers with ``type: command``
  under ``tts.providers.<name>`` in ``~/.hermes/config.yaml``. Hermes
  writes the input text to a temp file and runs the configured shell
  command, which must produce the audio file at the expected path.
  See the Local Command section of ``website/docs/user-guide/features/tts.md``.

Output formats:
- Opus (.ogg) for Telegram voice bubbles (requires ffmpeg for Edge TTS)
- MP3 (.mp3) for everything else (CLI, Discord, WhatsApp)

Configuration is loaded from ~/.hermes/config.yaml under the 'tts:' key.
The user chooses the provider and voice; the model just sends text.

Usage:
    from tools.tts_tool import text_to_speech_tool, check_tts_requirements

    result = text_to_speech_tool(text="Hello world")
"""

import asyncio
import base64
import datetime
import json
import logging
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from urllib.parse import urljoin

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)
def get_env_value(name, default=None):
    """Read env values through the live config module.

    Tests may monkeypatch and later restore ``hermes_cli.config.get_env_value``
    before this module is imported. Resolve the helper at call time so TTS does
    not keep a stale imported function for the rest of the test process.
    """
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import managed_nous_tools_enabled, prefers_gateway, resolve_openai_audio_api_key
from tools.xai_http import hermes_xai_user_agent

# ---------------------------------------------------------------------------
# Lazy imports -- providers are imported only when actually used to avoid
# crashing in headless environments (SSH, Docker, WSL, no PortAudio).
# ---------------------------------------------------------------------------

def _import_edge_tts():
    """Lazy import edge_tts. Returns the module or raises ImportError."""
    import edge_tts
    return edge_tts

def _import_elevenlabs():
    """Lazy import ElevenLabs client. Returns the class or raises ImportError."""
    from elevenlabs.client import ElevenLabs
    return ElevenLabs

def _import_openai_client():
    """Lazy import OpenAI client. Returns the class or raises ImportError."""
    from openai import OpenAI as OpenAIClient
    return OpenAIClient

def _import_mistral_client():
    """Lazy import Mistral client. Returns the class or raises ImportError."""
    from mistralai.client import Mistral
    return Mistral

def _import_sounddevice():
    """Lazy import sounddevice. Returns the module or raises ImportError/OSError."""
    import sounddevice as sd
    return sd


def _import_kittentts():
    """Lazy import KittenTTS. Returns the class or raises ImportError."""
    from kittentts import KittenTTS
    return KittenTTS


def _import_piper():
    """Lazy import Piper. Returns the PiperVoice class or raises ImportError.

    Piper is an optional, fully-local neural TTS engine (Home Assistant /
    Open Home Foundation). ``pip install piper-tts`` provides cross-platform
    wheels (Linux / macOS / Windows, x86_64 + ARM64) with embedded espeak-ng.
    Voice models (.onnx + .onnx.json) are downloaded on first use.
    """
    from piper import PiperVoice
    return PiperVoice


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "edge"
DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_STREAMING_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
DEFAULT_KITTENTTS_MODEL = "KittenML/kitten-tts-nano-0.8-int8"  # 25MB
DEFAULT_KITTENTTS_VOICE = "Jasper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # balanced size/quality
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MINIMAX_MODEL = "speech-01"
DEFAULT_MINIMAX_VOICE_ID = "female-shaonv"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.chat/v1/text_to_speech"
DEFAULT_MISTRAL_TTS_MODEL = "voxtral-mini-tts-2603"
DEFAULT_MISTRAL_TTS_VOICE_ID = "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral
DEFAULT_XAI_VOICE_ID = "eve"
DEFAULT_XAI_LANGUAGE = "en"
DEFAULT_XAI_SAMPLE_RATE = 24000
DEFAULT_XAI_BIT_RATE = 128000
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_GEMINI_TTS_VOICE = "Kore"
DEFAULT_GEMINI_TTS_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# PCM output specs for Gemini TTS (fixed by the API)
GEMINI_TTS_SAMPLE_RATE = 24000
GEMINI_TTS_CHANNELS = 1
GEMINI_TTS_SAMPLE_WIDTH = 2  # 16-bit PCM (L16)

def _get_default_output_dir() -> str:
    from hermes_constants import get_hermes_dir
    return str(get_hermes_dir("cache/audio", "audio_cache"))

DEFAULT_OUTPUT_DIR = _get_default_output_dir()

# ---------------------------------------------------------------------------
# Per-provider input-character limits (from official provider docs).
# A single global cap was wrong: OpenAI is 4096, xAI is 15k, MiniMax is 10k,
# ElevenLabs is model-dependent (5k / 10k / 30k / 40k), Gemini caps at ~8k
# input tokens.  Users can override any of these via
# ``tts.<provider>.max_text_length`` in config.yaml.
# ---------------------------------------------------------------------------
PROVIDER_MAX_TEXT_LENGTH: Dict[str, int] = {
    "edge": 5000,         # edge-tts practical sync limit
    "openai": 4096,       # https://platform.openai.com/docs/guides/text-to-speech
    "xai": 15000,         # https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
    "minimax": 10000,     # https://platform.minimax.io/docs/api-reference/speech-t2a-http (sync)
    "mistral": 4000,      # conservative; no published per-request cap
    "gemini": 5000,       # Gemini TTS caps at ~8k input tokens / ~655s audio
    "elevenlabs": 10000,  # fallback when model-aware lookup can't resolve (multilingual_v2)
    "neutts": 2000,       # local model, quality falls off on long text
    "kittentts": 2000,    # local 25MB model
    "piper": 5000,        # local VITS model, phoneme-based; practical cap
}

# ElevenLabs caps vary by model_id. https://elevenlabs.io/docs/overview/models
ELEVENLABS_MODEL_MAX_TEXT_LENGTH: Dict[str, int] = {
    "eleven_v3": 5000,
    "eleven_ttv_v3": 5000,
    "eleven_multilingual_v2": 10000,
    "eleven_multilingual_v1": 10000,
    "eleven_english_sts_v2": 10000,
    "eleven_english_sts_v1": 10000,
    "eleven_flash_v2": 30000,
    "eleven_flash_v2_5": 40000,
}

# Final fallback when provider isn't recognised at all.
FALLBACK_MAX_TEXT_LENGTH = 4000

# Back-compat alias. Prefer ``_resolve_max_text_length()`` for new code.
MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


def _resolve_max_text_length(
    provider: Optional[str],
    tts_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Return the input-character cap for *provider*.

    Resolution order:
      1. ``tts.<provider>.max_text_length`` (user override in config.yaml)
      2. ``tts.providers.<provider>.max_text_length`` for user-declared
         command providers
      3. ElevenLabs model-aware table (keyed on configured ``model_id``)
      4. ``PROVIDER_MAX_TEXT_LENGTH`` default
      5. ``DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH`` when the provider is a
         command-type user provider without an explicit cap
      6. ``FALLBACK_MAX_TEXT_LENGTH`` (4000)

    Non-positive or non-integer overrides fall through to the default so a
    broken config can't accidentally disable truncation entirely.
    """
    if not provider:
        return FALLBACK_MAX_TEXT_LENGTH
    key = provider.lower().strip()
    cfg = tts_config or {}

    # Built-in-style override at tts.<provider>.max_text_length wins first,
    # matching historical behavior.
    prov_cfg = cfg.get(key) if isinstance(cfg.get(key), dict) else {}
    override = prov_cfg.get("max_text_length") if prov_cfg else None
    if isinstance(override, bool):
        override = None
    if isinstance(override, int) and override > 0:
        return override

    if key == "elevenlabs":
        model_id = (prov_cfg or {}).get("model_id") or DEFAULT_ELEVENLABS_MODEL_ID
        mapped = ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get(str(model_id).strip())
        if mapped:
            return mapped

    if key in PROVIDER_MAX_TEXT_LENGTH:
        return PROVIDER_MAX_TEXT_LENGTH[key]

    # User-declared command provider (under tts.providers.<name>)
    if key not in BUILTIN_TTS_PROVIDERS:
        named = _get_named_provider_config(cfg, key)
        if _is_command_provider_config(named):
            named_override = named.get("max_text_length")
            if isinstance(named_override, bool):
                named_override = None
            if isinstance(named_override, int) and named_override > 0:
                return named_override
            return DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH

    return FALLBACK_MAX_TEXT_LENGTH


# ===========================================================================
# Config loader -- reads tts: section from ~/.hermes/config.yaml
# ===========================================================================
def _load_tts_config() -> Dict[str, Any]:
    """
    Load TTS configuration from ~/.hermes/config.yaml.

    Returns a dict with provider settings. Falls back to defaults
    for any missing fields.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("tts", {})
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
        return {}
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
        return {}


def _get_provider(tts_config: Dict[str, Any]) -> str:
    """Get the configured TTS provider name."""
    return (tts_config.get("provider") or DEFAULT_PROVIDER).lower().strip()


# ===========================================================================
# Custom command providers (type: command under tts.providers.<name>)
# ===========================================================================
#
# Users can declare any number of command-type providers alongside the
# built-ins so they can plug any local CLI (Piper, VoxCPM, Kokoro CLIs,
# custom voice-cloning scripts, etc.) into Hermes without any Python code
# changes. The config shape is::
#
#     tts:
#       provider: piper-en
#       providers:
#         piper-en:
#           type: command
#           command: "piper -m ~/model.onnx -f {output_path} < {input_path}"
#           output_format: wav
#
# Hermes writes the input text to a temp UTF-8 file, runs the command with
# placeholder substitution, and reads the audio file the command wrote to
# ``{output_path}``. Supported placeholders: ``{input_path}``,
# ``{text_path}`` (alias for input_path), ``{output_path}``, ``{format}``,
# ``{voice}``, ``{model}``, ``{speed}``. Use ``{{`` / ``}}`` for literal braces.
#
# Built-in provider names always win over an entry with the same name under
# ``tts.providers``, so user config can't silently shadow ``edge`` etc.
#
# Placeholder values are shell-quoted for their surrounding context
# (bare / single / double quote), so paths with spaces work transparently.

# Built-in provider names. Any ``tts.provider`` value NOT in this set is
# interpreted as a reference to ``tts.providers.<name>``.
BUILTIN_TTS_PROVIDERS = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "neutts",
    "kittentts",
    "piper",
})

DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS = 120
DEFAULT_COMMAND_TTS_OUTPUT_FORMAT = "mp3"
COMMAND_TTS_OUTPUT_FORMATS = frozenset({"mp3", "wav", "ogg", "flac"})
DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH = 5000


def _get_provider_section(tts_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a provider config block if it's a dict, else an empty dict."""
    if not isinstance(tts_config, dict):
        return {}
    section = tts_config.get(name)
    return section if isinstance(section, dict) else {}


def _get_named_provider_config(
    tts_config: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    """Return the config dict for a user-declared provider.

    Looks up ``tts.providers.<name>`` first (the canonical location), and
    falls back to ``tts.<name>`` so users who followed the built-in layout
    still work. Returns an empty dict when the provider is not declared.
    """
    providers = _get_provider_section(tts_config, "providers")
    section = providers.get(name) if isinstance(providers, dict) else None
    if isinstance(section, dict):
        return section
    # Back-compat: allow ``tts.<name>`` for user-declared providers too,
    # but only when the name is not a built-in (so a user's ``tts.openai``
    # block still means the OpenAI provider, not a custom command).
    if name.lower() not in BUILTIN_TTS_PROVIDERS:
        legacy = _get_provider_section(tts_config, name)
        if legacy:
            return legacy
    return {}


def _is_command_provider_config(config: Dict[str, Any]) -> bool:
    """Return True when *config* declares a command-type provider."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = config.get("command")
    return isinstance(command, str) and bool(command.strip())


def _resolve_command_provider_config(
    provider: str,
    tts_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the provider config if *provider* resolves to a command type.

    Built-in provider names are rejected (they have native handlers).
    Returns None when the name is a built-in, unknown, or not a command
    type.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return None
    config = _get_named_provider_config(tts_config, key)
    if _is_command_provider_config(config):
        return config
    return None


def _iter_command_providers(tts_config: Dict[str, Any]):
    """Yield (name, config) pairs for every declared command-type provider."""
    if not isinstance(tts_config, dict):
        return
    providers = _get_provider_section(tts_config, "providers")
    for name, cfg in (providers or {}).items():
        if isinstance(name, str) and name.lower() not in BUILTIN_TTS_PROVIDERS:
            if _is_command_provider_config(cfg):
                yield name, cfg


def _get_command_tts_timeout(config: Dict[str, Any]) -> float:
    """Return timeout in seconds, falling back when invalid."""
    raw = config.get("timeout", config.get("timeout_seconds", DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
    if value <= 0:
        return float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
    return value


def _get_command_tts_output_format(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Return the validated output format (mp3/wav/ogg/flac)."""
    if output_path:
        suffix = Path(output_path).suffix.lower().strip().lstrip(".")
        if suffix in COMMAND_TTS_OUTPUT_FORMATS:
            return suffix
    raw = (
        config.get("format")
        or config.get("output_format")
        or DEFAULT_COMMAND_TTS_OUTPUT_FORMAT
    )
    fmt = str(raw).lower().strip().lstrip(".")
    return fmt if fmt in COMMAND_TTS_OUTPUT_FORMATS else DEFAULT_COMMAND_TTS_OUTPUT_FORMAT


def _is_command_tts_voice_compatible(config: Dict[str, Any]) -> bool:
    """Return True only when the user explicitly opted in to voice delivery."""
    value = config.get("voice_compatible", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _shell_quote_context(command_template: str, position: int) -> Optional[str]:
    """Return the shell quote character active right before *position*.

    Returns ``"'"`` / ``'"'`` when inside a single- / double-quoted region
    of the template, ``None`` for bare context.
    """
    quote: Optional[str] = None
    escaped = False
    i = 0
    while i < position:
        char = command_template[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char == "'":
            quote = "'"
        elif char == '"':
            quote = '"'
        elif char == "\\":
            i += 1
        i += 1
    return quote


def _quote_command_tts_placeholder(value: str, quote_context: Optional[str]) -> str:
    """Quote a placeholder value for its position in a shell command template."""
    if quote_context == "'":
        return value.replace("'", r"'\''")
    if quote_context == '"':
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', r'\"')
            .replace("$", r"\$")
            .replace("`", r"\`")
        )
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _render_command_tts_template(
    command_template: str,
    placeholders: Dict[str, str],
) -> str:
    """Replace supported placeholders while preserving ``{{`` / ``}}``."""
    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(
        rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})"
    )
    replacements: list[tuple[str, str]] = []

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_TTS_PLACEHOLDER_{len(replacements)}__"
        replacements.append((
            token,
            _quote_command_tts_placeholder(
                placeholders[name],
                _shell_quote_context(command_template, match.start()),
            ),
        ))
        return token

    rendered = pattern.sub(replace_match, command_template)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


def _terminate_command_tts_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            proc.kill()
        return

    import psutil
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
    except psutil.NoSuchProcess:
        return
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        return
    except Exception:
        proc.kill()


def _run_command_tts(command: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a command-provider shell command with process-tree timeout cleanup."""
    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_command_tts_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except Exception:
            stdout = getattr(exc, "output", None)
            stderr = getattr(exc, "stderr", None)
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc

    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _configured_command_tts_output_path(path: Path, config: Dict[str, Any]) -> Path:
    """Return an output path whose extension matches the provider's output_format."""
    fmt = _get_command_tts_output_format(config)
    return path.with_suffix(f".{fmt}")


def _generate_command_tts(
    text: str,
    output_path: str,
    provider_name: str,
    config: Dict[str, Any],
    tts_config: Dict[str, Any],
) -> str:
    """Generate speech by running a user-configured shell command.

    Returns the absolute path of the audio file the command wrote.
    Raises ``ValueError`` when the provider config is invalid, and
    ``RuntimeError`` for timeouts / non-zero exits / empty output.
    """
    command_template = str(config.get("command") or "").strip()
    if not command_template:
        raise ValueError(
            f"tts.providers.{provider_name}.command is not configured"
        )

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    timeout = _get_command_tts_timeout(config)
    output_format = _get_command_tts_output_format(config, str(output))
    speed = config.get("speed", tts_config.get("speed", ""))

    with tempfile.TemporaryDirectory() as tmpdir:
        text_path = Path(tmpdir) / "input.txt"
        text_path.write_text(text, encoding="utf-8")

        placeholders = {
            "input_path": str(text_path),
            "text_path": str(text_path),
            "output_path": str(output),
            "format": output_format,
            "voice": str(config.get("voice", "")),
            "model": str(config.get("model", "")),
            "speed": str(speed),
        }
        command = _render_command_tts_template(command_template, placeholders)

        try:
            _run_command_tts(command, timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"TTS provider '{provider_name}' timed out after {timeout:g}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail_parts = []
            if exc.stderr:
                detail_parts.append(f"stderr: {exc.stderr.strip()}")
            if exc.stdout:
                detail_parts.append(f"stdout: {exc.stdout.strip()}")
            detail = "; ".join(detail_parts) or "no command output"
            raise RuntimeError(
                f"TTS provider '{provider_name}' exited with code "
                f"{exc.returncode}: {detail}"
            ) from exc

    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(
            f"TTS provider '{provider_name}' produced no output at {output}"
        )
    return str(output)


def _has_any_command_tts_provider(tts_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when any command-type TTS provider is configured."""
    if tts_config is None:
        tts_config = _load_tts_config()
    for _name, _cfg in _iter_command_providers(tts_config):
        return True
    return False


# ===========================================================================
# ffmpeg Opus conversion (Edge TTS MP3 -> OGG Opus for Telegram)
# ===========================================================================
def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _convert_to_opus(mp3_path: str) -> Optional[str]:
    """
    Convert an MP3 file to OGG Opus format for Telegram voice bubbles.

    Args:
        mp3_path: Path to the input MP3 file.

    Returns:
        Path to the .ogg file, or None if conversion fails.
    """
    if not _has_ffmpeg():
        return None

    ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-acodec", "libopus",
             "-ac", "1", "-b:a", "64k", "-vbr", "off", ogg_path, "-y"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s", 
                          result.returncode, result.stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0:
            return ogg_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg OGG conversion timed out after 30s")
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
    return None


# ===========================================================================
# Provider: Edge TTS (free)
# ===========================================================================
async def _generate_edge_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using Edge TTS.

    Args:
        text: Text to convert.
        output_path: Where to save the MP3 file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    _edge_tts = _import_edge_tts()
    edge_config = tts_config.get("edge", {})
    voice = edge_config.get("voice", DEFAULT_EDGE_VOICE)
    speed = float(edge_config.get("speed", tts_config.get("speed", 1.0)))

    kwargs = {"voice": voice}
    if speed != 1.0:
        pct = round((speed - 1.0) * 100)
        kwargs["rate"] = f"{pct:+d}%"

    communicate = _edge_tts.Communicate(text, **kwargs)
    await communicate.save(output_path)
    return output_path


# ===========================================================================
# Provider: ElevenLabs (premium)
# ===========================================================================
def _generate_elevenlabs(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using ElevenLabs.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    api_key = (get_env_value("ELEVENLABS_API_KEY") or "")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/")

    el_config = tts_config.get("elevenlabs", {})
    voice_id = el_config.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
    model_id = el_config.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID)

    # Determine output format based on file extension
    if output_path.endswith(".ogg"):
        output_format = "opus_48000_64"
    else:
        output_format = "mp3_44100_128"

    ElevenLabs = _import_elevenlabs()
    client = ElevenLabs(api_key=api_key)
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )

    # audio_generator yields chunks -- write them all
    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    return output_path


# ===========================================================================
# Provider: OpenAI TTS
# ===========================================================================
def _generate_openai_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using OpenAI TTS.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    api_key, base_url = _resolve_openai_audio_client_config()

    oai_config = tts_config.get("openai", {})
    model = oai_config.get("model", DEFAULT_OPENAI_MODEL)
    voice = oai_config.get("voice", DEFAULT_OPENAI_VOICE)
    base_url = oai_config.get("base_url", base_url)
    speed = float(oai_config.get("speed", tts_config.get("speed", 1.0)))

    # Determine response format from extension
    if output_path.endswith(".ogg"):
        response_format = "opus"
    else:
        response_format = "mp3"

    OpenAIClient = _import_openai_client()
    client = OpenAIClient(api_key=api_key, base_url=base_url)
    try:
        create_kwargs = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
            "extra_headers": {"x-idempotency-key": str(uuid.uuid4())},
        }
        if speed != 1.0:
            create_kwargs["speed"] = max(0.25, min(4.0, speed))
        response = client.audio.speech.create(**create_kwargs)

        response.stream_to_file(output_path)
        return output_path
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ===========================================================================
# Provider: xAI TTS
# ===========================================================================
def _generate_xai_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using xAI TTS.

    xAI exposes a dedicated /v1/tts endpoint instead of the OpenAI audio.speech
    API shape, so this is implemented as a separate backend.
    """
    import requests

    api_key = (get_env_value("XAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("XAI_API_KEY not set. Get one at https://console.x.ai/")

    xai_config = tts_config.get("xai", {})
    voice_id = str(xai_config.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
    language = str(xai_config.get("language", DEFAULT_XAI_LANGUAGE)).strip() or DEFAULT_XAI_LANGUAGE
    sample_rate = int(xai_config.get("sample_rate", DEFAULT_XAI_SAMPLE_RATE))
    bit_rate = int(xai_config.get("bit_rate", DEFAULT_XAI_BIT_RATE))
    base_url = str(
        xai_config.get("base_url")
        or get_env_value("XAI_BASE_URL")
        or DEFAULT_XAI_BASE_URL
    ).strip().rstrip("/")

    # Match the documented minimal POST /v1/tts shape by default. Only send
    # output_format when Hermes actually needs a non-default format/override.
    codec = "wav" if output_path.endswith(".wav") else "mp3"
    payload: Dict[str, Any] = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
    }
    if (
        codec != "mp3"
        or sample_rate != DEFAULT_XAI_SAMPLE_RATE
        or (codec == "mp3" and bit_rate != DEFAULT_XAI_BIT_RATE)
    ):
        output_format: Dict[str, Any] = {"codec": codec}
        if sample_rate:
            output_format["sample_rate"] = sample_rate
        if codec == "mp3" and bit_rate:
            output_format["bit_rate"] = bit_rate
        payload["output_format"] = output_format

    response = requests.post(
        f"{base_url}/tts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


# ===========================================================================
# Provider: MiniMax TTS
# ===========================================================================
def _generate_minimax_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using MiniMax TTS API (v1/text_to_speech).

    The current API (api.minimax.chat/v1/text_to_speech) uses a simple payload
    and returns raw audio bytes directly (Content-Type: audio/mpeg), unlike
    the deprecated v1/t2a_v2 endpoint which returned JSON with hex-encoded audio.

    Args:
        text: Text to convert (max 10,000 characters).
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    import requests

    api_key = (get_env_value("MINIMAX_API_KEY") or "")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY not set. Get one at https://platform.minimax.io/")

    mm_config = tts_config.get("minimax", {})
    model = mm_config.get("model", DEFAULT_MINIMAX_MODEL)
    voice_id = mm_config.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)
    base_url = mm_config.get("base_url", DEFAULT_MINIMAX_BASE_URL)

    payload = {
        "model": model,
        "text": text,
        "voice_id": voice_id,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(base_url, json=payload, headers=headers, timeout=60)

    content_type = response.headers.get("Content-Type", "")

    if "audio/" in content_type:
        # New API: returns raw audio directly
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    # Legacy / fallback: try parsing as JSON with hex-encoded audio
    try:
        result = response.json()
    except Exception:
        response.raise_for_status()
        raise RuntimeError(
            f"MiniMax TTS returned unexpected Content-Type '{content_type}' "
            f"({len(response.content)} bytes)"
        )

    base_resp = result.get("base_resp", {})
    status_code = base_resp.get("status_code", -1)

    if status_code != 0:
        status_msg = base_resp.get("status_msg", "unknown error")
        raise RuntimeError(f"MiniMax TTS API error (code {status_code}): {status_msg}")

    hex_audio = result.get("data", {}).get("audio", "")
    if not hex_audio:
        raise RuntimeError("MiniMax TTS returned empty audio data")

    # Legacy: hex-encoded audio
    audio_bytes = bytes.fromhex(hex_audio)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


# ===========================================================================
# Provider: Mistral (Voxtral TTS)
# ===========================================================================
def _generate_mistral_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate audio using Mistral Voxtral TTS API.

    The API returns base64-encoded audio; this function decodes it
    and writes the raw bytes to *output_path*.
    Supports native Opus output for Telegram voice bubbles.
    """
    api_key = (get_env_value("MISTRAL_API_KEY") or "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set. Get one at https://console.mistral.ai/")

    mi_config = tts_config.get("mistral", {})
    model = mi_config.get("model", DEFAULT_MISTRAL_TTS_MODEL)
    voice_id = mi_config.get("voice_id") or DEFAULT_MISTRAL_TTS_VOICE_ID

    if output_path.endswith(".ogg"):
        response_format = "opus"
    elif output_path.endswith(".wav"):
        response_format = "wav"
    elif output_path.endswith(".flac"):
        response_format = "flac"
    else:
        response_format = "mp3"

    Mistral = _import_mistral_client()
    try:
        with Mistral(api_key=api_key) as client:
            response = client.audio.speech.complete(
                model=model,
                input=text,
                voice_id=voice_id,
                response_format=response_format,
            )
            audio_bytes = base64.b64decode(response.audio_data)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Mistral TTS failed: %s", e, exc_info=True)
        raise RuntimeError(f"Mistral TTS failed: {type(e).__name__}") from e

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


# ===========================================================================
# Provider: Google Gemini TTS
# ===========================================================================
def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
    channels: int = GEMINI_TTS_CHANNELS,
    sample_width: int = GEMINI_TTS_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw signed-little-endian PCM with a standard WAV RIFF header.

    Gemini TTS returns audio/L16;codec=pcm;rate=24000 -- raw PCM samples with
    no container. We add a minimal WAV header so the file is playable and
    ffmpeg can re-encode it to MP3/Opus downstream.
    """
    import struct

    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,             # fmt chunk size (PCM)
        1,              # audio format (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
    )
    data_chunk_header = struct.pack("<4sI", b"data", data_size)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    return riff_header + fmt_chunk + data_chunk_header + pcm_bytes


def _generate_gemini_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate audio using Google Gemini TTS.

    Gemini's generateContent endpoint with responseModalities=["AUDIO"] returns
    raw 24kHz mono 16-bit PCM (L16) as base64. We wrap it with a WAV RIFF
    header to produce a playable file, then ffmpeg-convert to MP3 / Opus if
    the caller requested those formats (same pattern as NeuTTS).

    Args:
        text: Text to convert (prompt-style; supports inline direction like
              "Say cheerfully:" and audio tags like [whispers]).
        output_path: Where to save the audio file (.wav, .mp3, or .ogg).
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    import requests

    api_key = (get_env_value("GEMINI_API_KEY") or get_env_value("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey"
        )

    gemini_config = tts_config.get("gemini", {})
    model = str(gemini_config.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
    voice = str(gemini_config.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
    base_url = str(
        gemini_config.get("base_url")
        or get_env_value("GEMINI_BASE_URL")
        or DEFAULT_GEMINI_TTS_BASE_URL
    ).strip().rstrip("/")

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice},
                },
            },
        },
    }

    endpoint = f"{base_url}/models/{model}:generateContent"
    response = requests.post(
        endpoint,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        # Surface the API error message when present
        try:
            err = response.json().get("error", {})
            detail = err.get("message") or response.text[:300]
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(
            f"Gemini TTS API error (HTTP {response.status_code}): {detail}"
        )

    try:
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        audio_part = next((p for p in parts if "inlineData" in p or "inline_data" in p), None)
        if audio_part is None:
            raise RuntimeError("Gemini TTS response contained no audio data")
        inline = audio_part.get("inlineData") or audio_part.get("inline_data") or {}
        audio_b64 = inline.get("data", "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Gemini TTS response was malformed: {e}") from e

    if not audio_b64:
        raise RuntimeError("Gemini TTS returned empty audio data")

    pcm_bytes = base64.b64decode(audio_b64)
    wav_bytes = _wrap_pcm_as_wav(pcm_bytes)

    # Fast path: caller wants WAV directly, just write.
    if output_path.lower().endswith(".wav"):
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return output_path

    # Otherwise write WAV to a temp file and ffmpeg-convert to the target
    # format (.mp3 or .ogg). If ffmpeg is missing, fall back to renaming the
    # WAV -- this matches the NeuTTS behavior and keeps the tool usable on
    # systems without ffmpeg (audio still plays, just with a misleading
    # extension).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name

    try:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            # For .ogg output, force libopus encoding (Telegram voice bubbles
            # require Opus specifically; ffmpeg's default for .ogg is Vorbis).
            if output_path.lower().endswith(".ogg"):
                cmd = [
                    ffmpeg, "-i", wav_path,
                    "-acodec", "libopus", "-ac", "1",
                    "-b:a", "64k", "-vbr", "off",
                    "-y", "-loglevel", "error",
                    output_path,
                ]
            else:
                cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        else:
            logger.warning(
                "ffmpeg not found; writing raw WAV to %s (extension may be misleading)",
                output_path,
            )
            shutil.copyfile(wav_path, output_path)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return output_path


# ===========================================================================
# NeuTTS (local, on-device TTS via neutts_cli)
# ===========================================================================

def _check_neutts_available() -> bool:
    """Check if the neutts engine is importable (installed locally)."""
    try:
        import importlib.util
        return importlib.util.find_spec("neutts") is not None
    except Exception:
        return False


def _check_kittentts_available() -> bool:
    """Check if the kittentts engine is importable (installed locally)."""
    try:
        import importlib.util
        return importlib.util.find_spec("kittentts") is not None
    except Exception:
        return False


def _default_neutts_ref_audio() -> str:
    """Return path to the bundled default voice reference audio."""
    return str(Path(__file__).parent / "neutts_samples" / "jo.wav")


def _default_neutts_ref_text() -> str:
    """Return path to the bundled default voice reference transcript."""
    return str(Path(__file__).parent / "neutts_samples" / "jo.txt")


def _generate_neutts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate speech using the local NeuTTS engine.

    Runs synthesis in a subprocess via tools/neutts_synth.py to keep the
    ~500MB model in a separate process that exits after synthesis.
    Outputs WAV; the caller handles conversion for Telegram if needed.
    """
    import sys

    neutts_config = tts_config.get("neutts", {})
    ref_audio = neutts_config.get("ref_audio", "") or _default_neutts_ref_audio()
    ref_text = neutts_config.get("ref_text", "") or _default_neutts_ref_text()
    model = neutts_config.get("model", "neuphonic/neutts-air-q4-gguf")
    device = neutts_config.get("device", "cpu")

    # NeuTTS outputs WAV natively — use a .wav path for generation,
    # let the caller convert to the final format afterward.
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    synth_script = str(Path(__file__).parent / "neutts_synth.py")
    cmd = [
        sys.executable, synth_script,
        "--text", text,
        "--out", wav_path,
        "--ref-audio", ref_audio,
        "--ref-text", ref_text,
        "--model", model,
        "--device", device,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Filter out the "OK:" line from stderr
        error_lines = [l for l in stderr.splitlines() if not l.startswith("OK:")]
        raise RuntimeError(f"NeuTTS synthesis failed: {chr(10).join(error_lines) or 'unknown error'}")

    # If the caller wanted .mp3 or .ogg, convert from WAV
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30)
            os.remove(wav_path)
        else:
            # No ffmpeg — just rename the WAV to the expected path
            os.rename(wav_path, output_path)

    return output_path


# ===========================================================================
# Provider: Piper (local, neural VITS, 44 languages)
# ===========================================================================

# Module-level cache for Piper voice instances. Voices are keyed on their
# absolute .onnx model path so switching voices doesn't invalidate older
# cached voices.
_piper_voice_cache: Dict[str, Any] = {}


def _check_piper_available() -> bool:
    """Check whether the piper-tts package is importable."""
    try:
        import importlib.util
        return importlib.util.find_spec("piper") is not None
    except Exception:
        return False


def _get_piper_voices_dir() -> Path:
    """Return the directory where Hermes caches Piper voice models.

    Resolves to ``~/.hermes/cache/piper-voices/`` under the active
    HERMES_HOME so voice downloads follow profile boundaries.
    """
    from hermes_constants import get_hermes_dir
    root = Path(get_hermes_dir("cache/piper-voices", "piper_voices_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_piper_voice_path(voice: str, download_dir: Path) -> str:
    """Resolve *voice* (a model name or path) to a concrete .onnx file path.

    Accepts any of:
      - Absolute / expanded path to an .onnx file the user already has
      - A voice *name* like ``en_US-lessac-medium`` (downloads to
        ``download_dir`` on first use via ``python -m piper.download_voices``)

    Raises RuntimeError if the model can't be located or downloaded.
    """
    if not voice:
        voice = DEFAULT_PIPER_VOICE

    # Case 1: user gave a direct file path.
    candidate = Path(voice).expanduser()
    if candidate.suffix.lower() == ".onnx" and candidate.exists():
        return str(candidate)

    # Case 2: user gave a voice *name*. See if it's already downloaded.
    cached = download_dir / f"{voice}.onnx"
    if cached.exists() and (download_dir / f"{voice}.onnx.json").exists():
        return str(cached)

    # Case 3: download the voice. piper ships a download helper module.
    import sys as _sys
    logger.info("[Piper] Downloading voice '%s' to %s (first use)", voice, download_dir)
    try:
        result = subprocess.run(
            [_sys.executable, "-m", "piper.download_voices", voice,
             "--download-dir", str(download_dir)],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Piper voice download timed out after 300s for '{voice}'"
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "no stderr output"
        raise RuntimeError(
            f"Piper voice download failed for '{voice}': {stderr[:400]}"
        )

    if not cached.exists():
        raise RuntimeError(
            f"Piper voice download completed but {cached} is missing — "
            f"check voice name (see: https://github.com/OHF-Voice/piper1-gpl/"
            f"blob/main/docs/VOICES.md)"
        )
    return str(cached)


def _generate_piper_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate speech using the local Piper engine.

    Loads the voice model once per process (cached by absolute path) and
    writes a WAV file. Caller is responsible for converting to MP3/Opus
    via ffmpeg when a different output format is required.
    """
    PiperVoice = _import_piper()
    import wave

    piper_config = tts_config.get("piper", {}) if isinstance(tts_config, dict) else {}
    voice_name = piper_config.get("voice") or DEFAULT_PIPER_VOICE
    download_dir = Path(piper_config.get("voices_dir") or _get_piper_voices_dir()).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = bool(piper_config.get("use_cuda", False))

    model_path = _resolve_piper_voice_path(voice_name, download_dir)

    cache_key = f"{model_path}::cuda={use_cuda}"
    global _piper_voice_cache
    if cache_key not in _piper_voice_cache:
        logger.info("[Piper] Loading voice: %s", model_path)
        _piper_voice_cache[cache_key] = PiperVoice.load(model_path, use_cuda=use_cuda)
        logger.info("[Piper] Voice loaded")
    voice = _piper_voice_cache[cache_key]

    # Optional synthesis knobs — only pass a SynthesisConfig when at least
    # one advanced knob is configured, so we don't depend on a newer Piper
    # version than the user's installed one unless we need to.
    syn_config = None
    has_advanced = any(
        k in piper_config
        for k in ("length_scale", "noise_scale", "noise_w_scale", "volume", "normalize_audio")
    )
    if has_advanced:
        try:
            from piper import SynthesisConfig  # type: ignore
            syn_config = SynthesisConfig(
                length_scale=float(piper_config.get("length_scale", 1.0)),
                noise_scale=float(piper_config.get("noise_scale", 0.667)),
                noise_w_scale=float(piper_config.get("noise_w_scale", 0.8)),
                volume=float(piper_config.get("volume", 1.0)),
                normalize_audio=bool(piper_config.get("normalize_audio", True)),
            )
        except ImportError:
            logger.warning(
                "[Piper] SynthesisConfig not available in this piper-tts "
                "version — advanced knobs ignored"
            )

    # Piper outputs WAV. Caller handles downstream MP3/Opus conversion.
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    with wave.open(wav_path, "wb") as wav_file:
        if syn_config is not None:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        else:
            voice.synthesize_wav(text, wav_file)

    # Convert to desired format if caller requested mp3/ogg
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30)
            try:
                os.remove(wav_path)
            except OSError:
                pass
        else:
            # No ffmpeg — keep WAV and return that path
            os.rename(wav_path, output_path)

    return output_path


# ===========================================================================
# Provider: KittenTTS (local, lightweight)
# ===========================================================================

# Module-level cache for KittenTTS model instance
_kittentts_model_cache: Dict[str, Any] = {}


def _generate_kittentts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate speech using KittenTTS local ONNX model.

    KittenTTS is a lightweight TTS engine (25-80MB models) that runs
    entirely on CPU without requiring a GPU or API key.

    Args:
        text: Text to convert to speech.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    KittenTTS = _import_kittentts()
    kt_config = tts_config.get("kittentts", {})
    model_name = kt_config.get("model", DEFAULT_KITTENTTS_MODEL)
    voice = kt_config.get("voice", DEFAULT_KITTENTTS_VOICE)
    speed = kt_config.get("speed", 1.0)
    clean_text = kt_config.get("clean_text", True)

    # Use cached model instance if available
    global _kittentts_model_cache
    if model_name not in _kittentts_model_cache:
        logger.info("[KittenTTS] Loading model: %s", model_name)
        _kittentts_model_cache[model_name] = KittenTTS(model_name)
        logger.info("[KittenTTS] Model loaded successfully")

    model = _kittentts_model_cache[model_name]

    # Generate audio (returns numpy array at 24kHz)
    audio = model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

    # Save as WAV
    import soundfile as sf
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    sf.write(wav_path, audio, 24000)

    # Convert to desired format if needed
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30)
            os.remove(wav_path)
        else:
            # No ffmpeg — rename the WAV to the expected path
            os.rename(wav_path, output_path)

    return output_path


# ===========================================================================
# Main tool function
# ===========================================================================
def text_to_speech_tool(
    text: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert text to speech audio.

    Reads provider/voice config from ~/.hermes/config.yaml (tts: section).
    The model sends text; the user configures voice and provider.

    On messaging platforms, the returned MEDIA:<path> tag is intercepted
    by the send pipeline and delivered as a native voice message.
    In CLI mode, the file is saved to ~/voice-memos/.

    Args:
        text: The text to convert to speech.
        output_path: Optional custom save path. Defaults to ~/voice-memos/<timestamp>.mp3

    Returns:
        str: JSON result with success, file_path, and optionally MEDIA tag.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    tts_config = _load_tts_config()
    provider = _get_provider(tts_config)

    # User-declared command provider (type: command under tts.providers.<name>)
    # resolves BEFORE the built-in dispatch. Built-in names short-circuit here
    # so a user's ``tts.providers.openai.command`` can't override the real
    # OpenAI handler.
    command_provider_config = _resolve_command_provider_config(provider, tts_config)

    # Truncate very long text with a warning. The cap is per-provider
    # (OpenAI 4096, xAI 15k, MiniMax 10k, ElevenLabs model-aware, etc.).
    max_len = _resolve_max_text_length(provider, tts_config)
    if len(text) > max_len:
        logger.warning(
            "TTS text too long for provider %s (%d chars), truncating to %d",
            provider, len(text), max_len,
        )
        text = text[:max_len]

    # Detect platform from gateway env var to choose the best output format.
    # Telegram voice bubbles require Opus (.ogg); OpenAI and ElevenLabs can
    # produce Opus natively (no ffmpeg needed).  Edge TTS always outputs MP3
    # and needs ffmpeg for conversion.
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    want_opus = (platform == "telegram")

    # Determine output path
    if output_path:
        file_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            # Respect caller-supplied path but align the extension with the
            # provider's configured output_format so the command writes to a
            # path the caller actually expects.
            file_path = _configured_command_tts_output_path(
                file_path, command_provider_config
            )
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        if command_provider_config is not None:
            fmt = _get_command_tts_output_format(command_provider_config)
            file_path = out_dir / f"tts_{timestamp}.{fmt}"
        # Use .ogg for Telegram with providers that support native Opus output,
        # otherwise fall back to .mp3 (Edge TTS will attempt ffmpeg conversion later).
        elif want_opus and provider in {"openai", "elevenlabs", "mistral", "gemini"}:
            file_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            file_path = out_dir / f"tts_{timestamp}.mp3"

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_str = str(file_path)

    try:
        # Generate audio with the configured provider
        if command_provider_config is not None:
            logger.info(
                "Generating speech with command TTS provider '%s'...", provider,
            )
            file_str = _generate_command_tts(
                text, file_str, provider, command_provider_config, tts_config,
            )

        elif provider == "elevenlabs":
            try:
                _import_elevenlabs()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs"
                }, ensure_ascii=False)
            logger.info("Generating speech with ElevenLabs...")
            _generate_elevenlabs(text, file_str, tts_config)

        elif provider == "openai":
            try:
                _import_openai_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "OpenAI provider selected but 'openai' package not installed."
                }, ensure_ascii=False)
            logger.info("Generating speech with OpenAI TTS...")
            _generate_openai_tts(text, file_str, tts_config)

        elif provider == "minimax":
            logger.info("Generating speech with MiniMax TTS...")
            _generate_minimax_tts(text, file_str, tts_config)

        elif provider == "xai":
            logger.info("Generating speech with xAI TTS...")
            _generate_xai_tts(text, file_str, tts_config)

        elif provider == "mistral":
            try:
                _import_mistral_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "Mistral provider selected but 'mistralai' package not installed. "
                             "Run: pip install 'hermes-agent[mistral]'"
                }, ensure_ascii=False)
            logger.info("Generating speech with Mistral Voxtral TTS...")
            _generate_mistral_tts(text, file_str, tts_config)

        elif provider == "gemini":
            logger.info("Generating speech with Google Gemini TTS...")
            _generate_gemini_tts(text, file_str, tts_config)

        elif provider == "neutts":
            if not _check_neutts_available():
                return json.dumps({
                    "success": False,
                    "error": "NeuTTS provider selected but neutts is not installed. "
                             "Run hermes setup and choose NeuTTS, or install espeak-ng and run python -m pip install -U neutts[all]."
                }, ensure_ascii=False)
            logger.info("Generating speech with NeuTTS (local)...")
            _generate_neutts(text, file_str, tts_config)

        elif provider == "kittentts":
            try:
                _import_kittentts()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "KittenTTS provider selected but 'kittentts' package not installed. "
                             "Run 'hermes setup tts' and choose KittenTTS, or install manually: "
                             "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"
                }, ensure_ascii=False)
            logger.info("Generating speech with KittenTTS (local, ~25MB)...")
            _generate_kittentts(text, file_str, tts_config)

        elif provider == "piper":
            try:
                _import_piper()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "Piper provider selected but 'piper-tts' package not installed. "
                             "Run 'hermes tools' and select Piper under TTS, or install manually: "
                             "pip install piper-tts",
                }, ensure_ascii=False)
            logger.info("Generating speech with Piper (local)...")
            _generate_piper_tts(text, file_str, tts_config)

        else:
            # Default: Edge TTS (free), with NeuTTS as local fallback
            edge_available = True
            try:
                _import_edge_tts()
            except ImportError:
                edge_available = False

            if edge_available:
                logger.info("Generating speech with Edge TTS...")
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(
                            lambda: asyncio.run(_generate_edge_tts(text, file_str, tts_config))
                        ).result(timeout=60)
                except RuntimeError:
                    asyncio.run(_generate_edge_tts(text, file_str, tts_config))
            elif _check_neutts_available():
                logger.info("Edge TTS not available, falling back to NeuTTS (local)...")
                provider = "neutts"
                _generate_neutts(text, file_str, tts_config)
            else:
                return json.dumps({
                    "success": False,
                    "error": "No TTS provider available. Install edge-tts (pip install edge-tts) "
                             "or set up NeuTTS for local synthesis."
                }, ensure_ascii=False)

        # Check the file was actually created
        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return json.dumps({
                "success": False,
                "error": f"TTS generation produced no output (provider: {provider})"
            }, ensure_ascii=False)

        # Try Opus conversion for Telegram compatibility
        # Edge TTS outputs MP3, NeuTTS/KittenTTS output WAV — all need ffmpeg conversion
        voice_compatible = False
        if command_provider_config is not None:
            # Command providers are documents by default. Voice-bubble
            # delivery only kicks in when the user explicitly opts in
            # via ``voice_compatible: true`` in their provider config.
            if _is_command_tts_voice_compatible(command_provider_config):
                if not file_str.endswith(".ogg"):
                    opus_path = _convert_to_opus(file_str)
                    if opus_path:
                        file_str = opus_path
                voice_compatible = file_str.endswith(".ogg")
        elif provider in {"edge", "neutts", "minimax", "xai", "kittentts", "piper"} and not file_str.endswith(".ogg"):
            opus_path = _convert_to_opus(file_str)
            if opus_path:
                file_str = opus_path
                voice_compatible = True
        elif provider in {"elevenlabs", "openai", "mistral", "gemini"}:
            voice_compatible = file_str.endswith(".ogg")

        file_size = os.path.getsize(file_str)
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{file_size:,}", provider)

        # Build response with MEDIA tag for platform delivery
        media_tag = f"MEDIA:{file_str}"
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": file_str,
            "media_tag": media_tag,
            "provider": provider,
            "voice_compatible": voice_compatible,
        }, ensure_ascii=False)

    except ValueError as e:
        # Configuration errors (missing API keys, etc.)
        error_msg = f"TTS configuration error ({provider}): {e}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except FileNotFoundError as e:
        # Missing dependencies or files
        error_msg = f"TTS dependency missing ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    except Exception as e:
        # Unexpected errors
        error_msg = f"TTS generation failed ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)


# ===========================================================================
# Requirements check
# ===========================================================================
def check_tts_requirements() -> bool:
    """
    Check if at least one TTS provider is available.

    Edge TTS needs no API key and is the default, so if the package
    is installed, TTS is available. A user-declared command provider
    also satisfies the requirement.

    Returns:
        bool: True if at least one provider can work.
    """
    # Any configured command provider counts as available.
    if _has_any_command_tts_provider():
        return True
    try:
        _import_edge_tts()
        return True
    except ImportError:
        pass
    try:
        _import_elevenlabs()
        if get_env_value("ELEVENLABS_API_KEY"):
            return True
    except ImportError:
        pass
    try:
        _import_openai_client()
        if _has_openai_audio_backend():
            return True
    except ImportError:
        pass
    if get_env_value("MINIMAX_API_KEY"):
        return True
    if get_env_value("XAI_API_KEY"):
        return True
    if get_env_value("GEMINI_API_KEY") or get_env_value("GOOGLE_API_KEY"):
        return True
    try:
        _import_mistral_client()
        if get_env_value("MISTRAL_API_KEY"):
            return True
    except ImportError:
        pass
    if _check_neutts_available():
        return True
    if _check_kittentts_available():
        return True
    if _check_piper_available():
        return True
    return False


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """Return direct OpenAI audio config or a managed gateway fallback.

    When ``tts.use_gateway`` is set in config, the Tool Gateway is preferred
    even if direct OpenAI credentials are present.
    """
    direct_api_key = resolve_openai_audio_api_key()
    if direct_api_key and not prefers_gateway("tts"):
        return direct_api_key, DEFAULT_OPENAI_BASE_URL

    managed_gateway = resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        message = "Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set"
        if managed_nous_tools_enabled():
            message += ", and the managed OpenAI audio gateway is unavailable"
        raise ValueError(message)

    return managed_gateway.nous_user_token, urljoin(
        f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"
    )


def _has_openai_audio_backend() -> bool:
    """Return True when OpenAI audio can use direct credentials or the managed gateway."""
    return bool(resolve_openai_audio_api_key() or resolve_managed_tool_gateway("openai-audio"))


# ===========================================================================
# Streaming TTS: sentence-by-sentence pipeline for ElevenLabs
# ===========================================================================
# Sentence boundary pattern: punctuation followed by space or newline
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])(?:\s|\n)|(?:\n\n)')

# Markdown stripping patterns (same as cli.py _voice_speak_response)
_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_URL = re.compile(r'https?://\S+')
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'\*(.+?)\*')
_MD_INLINE_CODE = re.compile(r'`(.+?)`')
_MD_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
_MD_LIST_ITEM = re.compile(r'^\s*[-*]\s+', flags=re.MULTILINE)
_MD_HR = re.compile(r'---+')
_MD_EXCESS_NL = re.compile(r'\n{3,}')


def _strip_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting that shouldn't be spoken aloud."""
    text = _MD_CODE_BLOCK.sub(' ', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_URL.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_LIST_ITEM.sub('', text)
    text = _MD_HR.sub('', text)
    text = _MD_EXCESS_NL.sub('\n\n', text)
    return text.strip()


def stream_tts_to_speaker(
    text_queue: queue.Queue,
    stop_event: threading.Event,
    tts_done_event: threading.Event,
    display_callback: Optional[Callable[[str], None]] = None,
):
    """Consume text deltas from *text_queue*, buffer them into sentences,
    and stream each sentence through ElevenLabs TTS to the speaker in
    real-time.

    Protocol:
        * The producer puts ``str`` deltas onto *text_queue*.
        * A ``None`` sentinel signals end-of-text (flush remaining buffer).
        * *stop_event* can be set to abort early (e.g. user interrupt).
        * *tts_done_event* is **set** in the ``finally`` block so callers
          waiting on it (continuous voice mode) know playback is finished.
    """
    tts_done_event.clear()

    try:
        # --- TTS client setup (optional -- display_callback works without it) ---
        client = None
        output_stream = None
        voice_id = DEFAULT_ELEVENLABS_VOICE_ID
        model_id = DEFAULT_ELEVENLABS_STREAMING_MODEL_ID

        tts_config = _load_tts_config()
        el_config = tts_config.get("elevenlabs", {})
        voice_id = el_config.get("voice_id", voice_id)
        model_id = el_config.get("streaming_model_id",
                                 el_config.get("model_id", model_id))
        # Per-sentence cap for the streaming path. Look up the cap against
        # the *streaming* model_id (defaults to eleven_flash_v2_5 = 40k chars),
        # not the sync model_id. A user override
        # (tts.elevenlabs.max_text_length) still wins.
        stream_max_len = _resolve_max_text_length(
            "elevenlabs",
            {**tts_config, "elevenlabs": {**el_config, "model_id": model_id}},
        )

        api_key = (get_env_value("ELEVENLABS_API_KEY") or "")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set; streaming TTS audio disabled")
        else:
            try:
                ElevenLabs = _import_elevenlabs()
                client = ElevenLabs(api_key=api_key)
            except ImportError:
                logger.warning("elevenlabs package not installed; streaming TTS disabled")

            # Open a single sounddevice output stream for the lifetime of
            # this function.  ElevenLabs pcm_24000 produces signed 16-bit
            # little-endian mono PCM at 24 kHz.
            if client is not None:
                try:
                    sd = _import_sounddevice()
                    output_stream = sd.OutputStream(
                        samplerate=24000, channels=1, dtype="int16",
                    )
                    output_stream.start()
                except (ImportError, OSError) as exc:
                    logger.debug("sounddevice not available: %s", exc)
                    output_stream = None
                except Exception as exc:
                    logger.warning("sounddevice OutputStream failed: %s", exc)
                    output_stream = None

        sentence_buf = ""
        min_sentence_len = 20
        long_flush_len = 100
        queue_timeout = 0.5
        _spoken_sentences: list[str] = []  # track spoken sentences to skip duplicates
        # Regex to strip complete <think>...</think> blocks from buffer
        _think_block_re = re.compile(r'<think[\s>].*?</think>', flags=re.DOTALL)

        def _speak_sentence(sentence: str):
            """Display sentence and optionally generate + play audio."""
            if stop_event.is_set():
                return
            cleaned = _strip_markdown_for_tts(sentence).strip()
            if not cleaned:
                return
            # Skip duplicate/near-duplicate sentences (LLM repetition)
            cleaned_lower = cleaned.lower().rstrip(".!,")
            for prev in _spoken_sentences:
                if prev.lower().rstrip(".!,") == cleaned_lower:
                    return
            _spoken_sentences.append(cleaned)
            # Display raw sentence on screen before TTS processing
            if display_callback is not None:
                display_callback(sentence)
            # Skip audio generation if no TTS client available
            if client is None:
                return
            # Truncate very long sentences (ElevenLabs streaming path)
            if len(cleaned) > stream_max_len:
                cleaned = cleaned[:stream_max_len]
            try:
                audio_iter = client.text_to_speech.convert(
                    text=cleaned,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="pcm_24000",
                )
                if output_stream is not None:
                    for chunk in audio_iter:
                        if stop_event.is_set():
                            break
                        import numpy as _np
                        audio_array = _np.frombuffer(chunk, dtype=_np.int16)
                        output_stream.write(audio_array.reshape(-1, 1))
                else:
                    # Fallback: write chunks to temp file and play via system player
                    _play_via_tempfile(audio_iter, stop_event)
            except Exception as exc:
                logger.warning("Streaming TTS sentence failed: %s", exc)

        def _play_via_tempfile(audio_iter, stop_evt):
            """Write PCM chunks to a temp WAV file and play it."""
            tmp_path = None
            try:
                import wave
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(24000)
                    for chunk in audio_iter:
                        if stop_evt.is_set():
                            break
                        wf.writeframes(chunk)
                from tools.voice_mode import play_audio_file
                play_audio_file(tmp_path)
            except Exception as exc:
                logger.warning("Temp-file TTS fallback failed: %s", exc)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        while not stop_event.is_set():
            # Read next delta from queue
            try:
                delta = text_queue.get(timeout=queue_timeout)
            except queue.Empty:
                # Timeout: if we have accumulated a long buffer, flush it
                if len(sentence_buf) > long_flush_len:
                    _speak_sentence(sentence_buf)
                    sentence_buf = ""
                continue

            if delta is None:
                # End-of-text sentinel: strip any remaining think blocks, flush
                sentence_buf = _think_block_re.sub('', sentence_buf)
                if sentence_buf.strip():
                    _speak_sentence(sentence_buf)
                break

            sentence_buf += delta

            # --- Think block filtering ---
            # Strip complete <think>...</think> blocks from buffer.
            # Works correctly even when tags span multiple deltas.
            sentence_buf = _think_block_re.sub('', sentence_buf)

            # If an incomplete <think tag is at the end, wait for more data
            # before extracting sentences (the closing tag may arrive next).
            if '<think' in sentence_buf and '</think>' not in sentence_buf:
                continue

            # Check for sentence boundaries
            while True:
                m = _SENTENCE_BOUNDARY_RE.search(sentence_buf)
                if m is None:
                    break
                end_pos = m.end()
                sentence = sentence_buf[:end_pos]
                sentence_buf = sentence_buf[end_pos:]
                # Merge short fragments into the next sentence
                if len(sentence.strip()) < min_sentence_len:
                    sentence_buf = sentence + sentence_buf
                    break
                _speak_sentence(sentence)

        # Drain any remaining items from the queue
        while True:
            try:
                text_queue.get_nowait()
            except queue.Empty:
                break

        # output_stream is closed in the finally block below

    except Exception as exc:
        logger.warning("Streaming TTS pipeline error: %s", exc)
    finally:
        # Always close the audio output stream to avoid locking the device
        if output_stream is not None:
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass
        tts_done_event.set()


# ===========================================================================
# Main -- quick diagnostics
# ===========================================================================
if __name__ == "__main__":
    print("🔊 Text-to-Speech Tool Module")
    print("=" * 50)

    def _check(importer, label):
        try:
            importer()
            return True
        except ImportError:
            return False

    print("\nProvider availability:")
    print(f"  Edge TTS:   {'installed' if _check(_import_edge_tts, 'edge') else 'not installed (pip install edge-tts)'}")
    print(f"  ElevenLabs: {'installed' if _check(_import_elevenlabs, 'el') else 'not installed (pip install elevenlabs)'}")
    print(f"    API Key:  {'set' if get_env_value('ELEVENLABS_API_KEY') else 'not set'}")
    print(f"  OpenAI:     {'installed' if _check(_import_openai_client, 'oai') else 'not installed'}")
    print(
        "    API Key:  "
        f"{'set' if resolve_openai_audio_api_key() else 'not set (VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)'}"
    )
    print(f"  MiniMax:    {'API key set' if get_env_value('MINIMAX_API_KEY') else 'not set (MINIMAX_API_KEY)'}")
    print(f"  Piper:      {'installed' if _check_piper_available() else 'not installed (pip install piper-tts)'}")
    print(f"  ffmpeg:     {'✅ found' if _has_ffmpeg() else '❌ not found (needed for Telegram Opus)'}")
    print(f"\n  Output dir: {DEFAULT_OUTPUT_DIR}")

    config = _load_tts_config()
    provider = _get_provider(config)
    print(f"  Configured provider: {provider}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured (built-in providers like edge/openai or custom command providers under tts.providers.<name>), not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. Provider-specific character caps apply and are enforced automatically (OpenAI 4096, xAI 15000, MiniMax 10000, ElevenLabs 5k-40k depending on model); over-long input is truncated."
            },
            "output_path": {
                "type": "string",
                "description": f"Optional custom file path to save the audio. Defaults to {display_hermes_home()}/audio_cache/<timestamp>.mp3"
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
        output_path=args.get("output_path")),
    check_fn=check_tts_requirements,
    emoji="🔊",
)
