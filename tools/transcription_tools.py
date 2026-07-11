#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription with six providers:

  - **local** (default, free) — faster-whisper running locally, no API key needed.
    Auto-downloads the model (~150 MB for ``base``) on first use.
  - **groq** (free tier) — Groq Whisper API, requires ``GROQ_API_KEY``.
  - **openai** (paid) — OpenAI Whisper API, requires ``VOICE_TOOLS_OPENAI_KEY``.
  - **mistral** — Mistral Voxtral Transcribe API, requires ``MISTRAL_API_KEY``.
  - **xai** — xAI Grok STT API, requires ``XAI_API_KEY``. High accuracy,
    Inverse Text Normalization, diarization, 21 languages.
  - **elevenlabs** — ElevenLabs Scribe API, requires ``ELEVENLABS_API_KEY``.

Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, Slack, and Signal.

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, aac

Usage::

    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from hermes_cli._subprocess_compat import windows_hide_flags
from utils import is_truthy_value
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_openai_audio_api_key,
)

logger = logging.getLogger(__name__)

def get_env_value(name, default=None):
    """Read env values through the live config module.

    Tests may monkeypatch and later restore ``hermes_cli.config.get_env_value``
    before this module is imported. Resolve the helper at call time so STT does
    not keep a stale imported function for the rest of the test process.
    """
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _safe_find_spec(module_name: str) -> bool:
    try:
        return _ilu.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in globals() or module_name in os.sys.modules


_HAS_FASTER_WHISPER = _safe_find_spec("faster_whisper")
_HAS_OPENAI = _safe_find_spec("openai")
_HAS_MISTRAL = _safe_find_spec("mistralai")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "base"
DEFAULT_LOCAL_STT_LANGUAGE = "en"
DEFAULT_STT_MODEL = os.getenv("STT_OPENAI_MODEL", "whisper-1")
DEFAULT_GROQ_STT_MODEL = os.getenv("STT_GROQ_MODEL", "whisper-large-v3-turbo")
DEFAULT_MISTRAL_STT_MODEL = os.getenv("STT_MISTRAL_MODEL", "voxtral-mini-latest")
DEFAULT_ELEVENLABS_STT_MODEL = os.getenv("STT_ELEVENLABS_MODEL", "scribe_v2")
LOCAL_STT_COMMAND_ENV = "HERMES_LOCAL_STT_COMMAND"
LOCAL_STT_LANGUAGE_ENV = "HERMES_LOCAL_STT_LANGUAGE"
COMMON_LOCAL_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_BASE_URL = os.getenv("STT_OPENAI_BASE_URL", "https://api.openai.com/v1")
XAI_STT_BASE_URL = os.getenv("XAI_STT_BASE_URL", "https://api.x.ai/v1")
ELEVENLABS_STT_BASE_URL = os.getenv("ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1")

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".aac", ".flac"}
LOCAL_NATIVE_AUDIO_FORMATS = {".wav", ".aiff", ".aif"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Known model sets for auto-correction
OPENAI_MODELS = {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
GROQ_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"}

# Singleton for the local model — loaded once, reused across calls
_local_model: Optional[object] = None
_local_model_name: Optional[str] = None

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------



def _load_stt_config() -> dict:
    """Load the ``stt`` section from user config, falling back to defaults."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("stt") or {}
    except Exception:
        return {}


def is_stt_enabled(stt_config: Optional[dict] = None) -> bool:
    """Return whether STT is enabled in config."""
    if stt_config is None:
        stt_config = _load_stt_config()
    enabled = stt_config.get("enabled", True)
    return is_truthy_value(enabled, default=True)


def _has_openai_audio_backend() -> bool:
    """Return True when OpenAI audio can use config credentials, env credentials, or the managed gateway."""
    try:
        _resolve_openai_audio_client_config()
        return True
    except ValueError:
        return False


def _find_binary(binary_name: str) -> Optional[str]:
    """Find a local binary, checking common Homebrew/local prefixes as well as PATH."""
    for directory in COMMON_LOCAL_BIN_DIRS:
        candidate = Path(directory) / binary_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(binary_name)


def _find_ffmpeg_binary() -> Optional[str]:
    return _find_binary("ffmpeg")


def _find_whisper_binary() -> Optional[str]:
    return _find_binary("whisper")


def _get_local_command_template() -> Optional[str]:
    configured = os.getenv(LOCAL_STT_COMMAND_ENV, "").strip()
    if configured:
        return configured

    whisper_binary = _find_whisper_binary()
    if whisper_binary:
        quoted_binary = shlex.quote(whisper_binary)
        return (
            f"{quoted_binary} {{input_path}} --model {{model}} --output_format txt "
            "--output_dir {output_dir} --language {language}"
        )
    return None


def _has_local_command() -> bool:
    return _get_local_command_template() is not None


def _normalize_local_model(model_name: Optional[str]) -> str:
    """Return a valid faster-whisper model size, mapping cloud-only names to the default.

    Cloud providers like OpenAI use names such as ``whisper-1`` which are not
    valid for faster-whisper (which expects ``tiny``, ``base``, ``small``,
    ``medium``, or ``large-v*``).  When such a name is detected we fall back to
    the default local model and emit a warning so the user knows what happened.
    """
    if not model_name or model_name in OPENAI_MODELS or model_name in GROQ_MODELS:
        if model_name and (model_name in OPENAI_MODELS or model_name in GROQ_MODELS):
            logger.warning(
                "STT model '%s' is a cloud-only name and cannot be used with the local "
                "provider. Falling back to '%s'. Set stt.local.model to a valid "
                "faster-whisper size (tiny, base, small, medium, large-v3).",
                model_name,
                DEFAULT_LOCAL_MODEL,
            )
        return DEFAULT_LOCAL_MODEL
    return model_name


def _normalize_local_command_model(model_name: Optional[str]) -> str:
    return _normalize_local_model(model_name)


def _try_lazy_install_stt() -> bool:
    """Attempt to lazy-install faster-whisper and return True on success.

    The module-level ``_HAS_FASTER_WHISPER`` flag is set at import time and
    cached. If the package wasn't installed at startup, calling ``ensure()``
    installs it. This function re-checks dynamically after installation so
    the provider can use it immediately without a process restart.
    """
    try:
        from tools.lazy_deps import ensure
        # prompt=False: never raise a blocking input() prompt mid-session.
        # Under the interactive CLI prompt_toolkit owns stdin, so a bare
        # input() deadlocks the terminal (#40490). The install is already
        # gated by security.allow_lazy_installs, so reaching here is opt-in.
        ensure("stt.faster_whisper", prompt=False)
        # Re-check dynamically after install
        import importlib.util as _iu
        if _iu.find_spec("faster_whisper"):
            return True
    except Exception as exc:
        logger.debug("Lazy install of faster-whisper failed: %s", exc)
    return False


# Names of the 6 STT providers with native handlers in this module.
# Kept in sync with ``agent.transcription_registry._BUILTIN_NAMES`` —
# a regression test fails if they drift. The plugin hook from
# issue #30398-style follow-up rejects plugins registering under any
# of these names; the dispatcher in ``transcribe_audio`` short-circuits
# them defensively as well.
BUILTIN_STT_PROVIDERS = frozenset({
    "local",
    "local_command",
    "groq",
    "openai",
    "mistral",
    "xai",
})


# ---------------------------------------------------------------------------
# Command-provider registry (``stt.providers.<name>: type: command``)
# ---------------------------------------------------------------------------
#
# Mirrors the TTS command-provider registry shipped in PR #17843 — same
# placeholder grammar, same shell-quote-aware rendering, same process-tree
# termination on timeout. Lets any whisper CLI / ASR CLI / curl pipeline
# become an STT backend with zero Python.
#
# Resolution order:
#   1. Built-in (``local``, ``local_command``, ``groq``, ``openai``,
#      ``mistral``, ``xai``)              → native handler. **Always wins.**
#   2. ``stt.providers.<name>: type: command``  → command-provider runner.
#   3. Plugin-registered TranscriptionProvider  → plugin dispatch.
#   4. No match                                 → "No STT provider available".
#
# The single-env-var ``HERMES_LOCAL_STT_COMMAND`` escape hatch is preserved
# untouched via the built-in ``local_command`` path. Use the command-provider
# registry when you want MULTIPLE shell-driven STT engines, or you want a
# named provider you can pick via ``stt.provider`` in config.yaml.
DEFAULT_COMMAND_STT_TIMEOUT_SECONDS = 300
DEFAULT_COMMAND_STT_LANGUAGE = "en"
DEFAULT_COMMAND_STT_OUTPUT_FORMAT = "txt"
COMMAND_STT_OUTPUT_FORMATS = frozenset({"txt", "json", "srt", "vtt"})


def _get_stt_section(stt_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return an stt sub-section if it's a dict, else an empty dict."""
    if not isinstance(stt_config, dict):
        return {}
    section = stt_config.get(name)
    return section if isinstance(section, dict) else {}


def _get_named_stt_provider_config(
    stt_config: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    """Return the config dict for a user-declared STT command provider.

    Looks up ``stt.providers.<name>`` first (the canonical location), and
    falls back to ``stt.<name>`` so users who followed the built-in layout
    still work. Returns an empty dict when the provider is not declared.

    Built-in names are NOT special-cased here — the caller short-circuits
    them before this is consulted, AND ``_is_command_stt_provider_config``
    requires an explicit ``command:`` value, so a built-in section like
    ``stt.openai`` (which has ``model``/``language`` but no ``command``)
    can't accidentally be treated as a command provider.
    """
    providers = _get_stt_section(stt_config, "providers")
    section = providers.get(name) if isinstance(providers, dict) else None
    if isinstance(section, dict):
        return section
    # Back-compat: allow ``stt.<name>`` for user-declared providers too,
    # but only when the name is not a built-in (so a user's ``stt.openai``
    # block still means the OpenAI provider, not a custom command).
    if name.lower() not in BUILTIN_STT_PROVIDERS:
        legacy = _get_stt_section(stt_config, name)
        if legacy:
            return legacy
    return {}


def _is_command_stt_provider_config(config: Dict[str, Any]) -> bool:
    """Return True when *config* declares a command-type STT provider."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = config.get("command")
    return isinstance(command, str) and bool(command.strip())


def _resolve_command_stt_provider_config(
    provider: str,
    stt_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the provider config if *provider* resolves to a command type.

    Built-in provider names are rejected (they have native handlers).
    Returns None when the name is a built-in, ``"none"``, unknown, or not
    a command type.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_STT_PROVIDERS or key == "none":
        return None
    config = _get_named_stt_provider_config(stt_config, key)
    if _is_command_stt_provider_config(config):
        return config
    return None


def _iter_command_stt_providers(stt_config: Dict[str, Any]):
    """Yield (name, config) pairs for every declared command-type STT provider."""
    if not isinstance(stt_config, dict):
        return
    providers = _get_stt_section(stt_config, "providers")
    for name, cfg in (providers or {}).items():
        if isinstance(name, str) and name.lower() not in BUILTIN_STT_PROVIDERS:
            if _is_command_stt_provider_config(cfg):
                yield name, cfg


def _has_any_command_stt_provider(stt_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when any command-type STT provider is configured."""
    if stt_config is None:
        stt_config = _load_stt_config()
    for _name, _cfg in _iter_command_stt_providers(stt_config):
        return True
    return False


def _get_command_stt_timeout(config: Dict[str, Any]) -> float:
    """Return timeout in seconds, falling back when invalid."""
    raw = config.get("timeout", config.get("timeout_seconds", DEFAULT_COMMAND_STT_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_COMMAND_STT_TIMEOUT_SECONDS)
    if value <= 0:
        return float(DEFAULT_COMMAND_STT_TIMEOUT_SECONDS)
    return value


def _get_command_stt_output_format(config: Dict[str, Any]) -> str:
    """Return the validated output format (txt/json/srt/vtt)."""
    raw = (
        config.get("format")
        or config.get("output_format")
        or DEFAULT_COMMAND_STT_OUTPUT_FORMAT
    )
    fmt = str(raw).lower().strip().lstrip(".")
    return fmt if fmt in COMMAND_STT_OUTPUT_FORMATS else DEFAULT_COMMAND_STT_OUTPUT_FORMAT


def _shell_quote_context_stt(command_template: str, position: int) -> Optional[str]:
    """Return the shell quote character active right before *position*.

    Mirrors ``tools.tts_tool._shell_quote_context`` — kept local to avoid
    cross-module import of a private helper. Returns ``"'"`` / ``'"'`` when
    inside a quoted region, ``None`` for bare context.
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


def _quote_command_stt_placeholder(value: str, quote_context: Optional[str]) -> str:
    """Quote a placeholder value for its position in a shell command template.

    Mirrors ``tools.tts_tool._quote_command_tts_placeholder``.
    """
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


def _render_command_stt_template(
    command_template: str,
    placeholders: Dict[str, str],
) -> str:
    """Replace supported placeholders while preserving ``{{`` / ``}}``.

    Mirrors ``tools.tts_tool._render_command_tts_template``. Placeholders
    are shell-quote-aware: ``{voice}`` inside single quotes gets
    single-quote-safe escaping, inside double quotes gets ``$``/`` ` ``/`` " ``
    escaping, outside quotes gets ``shlex.quote``. Doubled braces ``{{`` and
    ``}}`` are preserved as literal ``{`` / ``}`` for users who want to
    embed JSON snippets in their command.
    """
    import re

    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(
        rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})"
    )
    replacements: list[tuple[str, str]] = []

    def replace_match(match: "re.Match[str]") -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_STT_PLACEHOLDER_{len(replacements)}__"
        replacements.append((
            token,
            _quote_command_stt_placeholder(
                placeholders[name],
                _shell_quote_context_stt(command_template, match.start()),
            ),
        ))
        return token

    rendered = pattern.sub(replace_match, command_template)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


def _terminate_command_stt_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children.

    Mirrors ``tools.tts_tool._terminate_command_tts_process_tree``.
    """
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            proc.kill()
        return

    try:
        import psutil  # type: ignore
    except ImportError:
        # psutil is optional — fall back to single-process terminate/kill
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

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


def _run_command_stt(command: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a command-provider shell command with process-tree timeout cleanup.

    Mirrors ``tools.tts_tool._run_command_tts``.
    """
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

    proc = subprocess.Popen(command, **popen_kwargs, stdin=subprocess.DEVNULL)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_command_stt_process_tree(proc)
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


def _read_command_stt_output(output_path: Path, stdout: str, fmt: str) -> str:
    """Return the transcript text from a command-provider invocation.

    Resolution:
      1. If ``output_path`` exists and is non-empty → read it (raw text).
      2. Else if ``stdout`` is non-empty → use stdout (lets users write
         curl-style one-liners that emit transcript to stdout instead of
         writing a file).
      3. Else → raise RuntimeError (no usable output produced).

    For JSON format, we still return the raw bytes — extracting a
    ``text`` field is out of scope; users either configure ``format: txt``
    or post-process JSON downstream. (Same trade-off as TTS: the runner
    doesn't try to be clever about output shape.)
    """
    if output_path.exists():
        try:
            content = output_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = output_path.read_bytes().decode("utf-8", errors="replace").strip()
        if content:
            return content
    if stdout and stdout.strip():
        return stdout.strip()
    raise RuntimeError(
        f"Command STT provider wrote no output file at {output_path} "
        f"and produced no stdout"
    )


def _transcribe_command_stt(
    file_path: str,
    provider_name: str,
    config: Dict[str, Any],
    stt_config: Dict[str, Any],
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe via a user-declared ``stt.providers.<name>: type: command``.

    Placeholder grammar:

    | Placeholder       | Substituted with                                          |
    |-------------------|-----------------------------------------------------------|
    | ``{input_path}``  | absolute path to the audio file (original location)       |
    | ``{output_path}`` | absolute path the provider should write its transcript to |
    | ``{output_dir}``  | parent dir of ``{output_path}``                           |
    | ``{format}``      | configured output format (``txt`` / ``json`` / ``srt`` / ``vtt``) |
    | ``{language}``    | configured language code (default ``en``)                 |
    | ``{model}``       | configured model id (empty when not set)                  |

    All placeholders are shell-quote-aware (see ``_render_command_stt_template``).
    Doubled braces ``{{`` and ``}}`` are preserved as literal braces.

    Returns the standard transcribe-response envelope (``success``,
    ``transcript``, ``provider``, ``error``).
    """
    command_template = str(config.get("command") or "").strip()
    if not command_template:
        return {
            "success": False,
            "transcript": "",
            "provider": provider_name,
            "error": f"stt.providers.{provider_name}.command is not configured",
        }

    audio = Path(file_path).expanduser()
    if not audio.exists():
        return {
            "success": False,
            "transcript": "",
            "provider": provider_name,
            "error": f"Audio file not found: {file_path}",
        }

    timeout = _get_command_stt_timeout(config)
    output_format = _get_command_stt_output_format(config)
    language = (
        config.get("language")
        or stt_config.get("language")
        or DEFAULT_COMMAND_STT_LANGUAGE
    )
    model = model_override or config.get("model") or ""

    try:
        with tempfile.TemporaryDirectory(prefix=f"hermes-cmd-stt-{provider_name}-") as tmpdir:
            output_path = Path(tmpdir) / f"transcript.{output_format}"
            placeholders = {
                "input_path": str(audio.resolve()),
                "output_path": str(output_path),
                "output_dir": str(output_path.parent),
                "format": output_format,
                "language": str(language),
                "model": str(model),
            }
            command = _render_command_stt_template(command_template, placeholders)
            logger.info(
                "Transcribing %s via command STT provider '%s'...",
                audio.name, provider_name,
            )
            try:
                result = _run_command_stt(command, timeout)
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "transcript": "",
                    "provider": provider_name,
                    "error": (
                        f"STT command provider '{provider_name}' timed out after "
                        f"{timeout:g}s"
                    ),
                }
            except subprocess.CalledProcessError as exc:
                detail_parts = []
                if exc.stderr:
                    detail_parts.append(f"stderr: {exc.stderr.strip()}")
                if exc.stdout:
                    detail_parts.append(f"stdout: {exc.stdout.strip()}")
                detail = "; ".join(detail_parts) or "no command output"
                return {
                    "success": False,
                    "transcript": "",
                    "provider": provider_name,
                    "error": (
                        f"STT command provider '{provider_name}' exited with code "
                        f"{exc.returncode}: {detail}"
                    ),
                }

            try:
                transcript_text = _read_command_stt_output(
                    output_path, result.stdout or "", output_format,
                )
            except RuntimeError as exc:
                return {
                    "success": False,
                    "transcript": "",
                    "provider": provider_name,
                    "error": str(exc),
                }

    except OSError as exc:
        return {
            "success": False,
            "transcript": "",
            "provider": provider_name,
            "error": f"STT command provider '{provider_name}' failed: {exc}",
        }

    logger.info(
        "Transcribed %s via command STT provider '%s' (%d chars)",
        audio.name, provider_name, len(transcript_text),
    )
    return {
        "success": True,
        "transcript": transcript_text,
        "provider": provider_name,
    }


def _get_provider(stt_config: dict) -> str:
    """Determine which STT provider to use.

    When ``stt.provider`` is explicitly set in config, that choice is
    honoured — no silent cloud fallback.  When no provider is configured,
    auto-detect tries: local > groq (free) > openai (paid).
    """
    if not is_stt_enabled(stt_config):
        return "none"

    explicit = "provider" in stt_config
    provider = stt_config.get("provider", DEFAULT_PROVIDER)

    # --- Explicit provider: respect the user's choice ----------------------

    if explicit:
        if provider == "local":
            if _HAS_FASTER_WHISPER:
                return "local"
            if _has_local_command():
                return "local_command"
            # Try lazy-install before giving up
            if _try_lazy_install_stt():
                return "local"
            logger.warning(
                "STT provider 'local' configured but unavailable "
                "(install faster-whisper or set HERMES_LOCAL_STT_COMMAND)"
            )
            return "none"

        if provider == "local_command":
            if _has_local_command():
                return "local_command"
            if _HAS_FASTER_WHISPER:
                logger.info("Local STT command unavailable, using local faster-whisper")
                return "local"
            logger.warning(
                "STT provider 'local_command' configured but unavailable"
            )
            return "none"

        if provider == "groq":
            if _HAS_OPENAI and get_env_value("GROQ_API_KEY"):
                return "groq"
            logger.warning(
                "STT provider 'groq' configured but GROQ_API_KEY not set"
            )
            return "none"

        if provider == "openai":
            if _HAS_OPENAI and _has_openai_audio_backend():
                return "openai"
            logger.warning(
                "STT provider 'openai' configured but no API key available"
            )
            return "none"

        if provider == "mistral":
            if _HAS_MISTRAL and get_env_value("MISTRAL_API_KEY"):
                return "mistral"
            logger.warning(
                "STT provider 'mistral' configured but mistralai package "
                "not installed or MISTRAL_API_KEY not set"
            )
            return "none"

        if provider == "xai":
            from tools.xai_http import resolve_xai_http_credentials

            if resolve_xai_http_credentials().get("api_key"):
                return "xai"
            logger.warning(
                "STT provider 'xai' configured but no xAI credentials are available"
            )
            return "none"

        if provider == "elevenlabs":
            if get_env_value("ELEVENLABS_API_KEY"):
                return "elevenlabs"
            logger.warning(
                "STT provider 'elevenlabs' configured but ELEVENLABS_API_KEY not set"
            )
            return "none"

        return provider  # Unknown — let it fail downstream

    # --- Auto-detect (no explicit provider): local > groq > openai > xai > elevenlabs -
    # mistral is intentionally skipped while `mistralai` is quarantined on
    # PyPI (malicious 2.4.6 release on 2026-05-12).

    if _HAS_FASTER_WHISPER:
        return "local"
    if _has_local_command():
        return "local_command"
    # Try lazy-install before falling through to cloud providers
    if _try_lazy_install_stt():
        return "local"
    if _HAS_OPENAI and get_env_value("GROQ_API_KEY"):
        logger.info("No local STT available, using Groq Whisper API")
        return "groq"
    if _HAS_OPENAI and _has_openai_audio_backend():
        logger.info("No local STT available, using OpenAI Whisper API")
        return "openai"
    # Only auto-select Mistral if the SDK is already present — don't trigger a
    # lazy-install during passive auto-detection. Explicit `provider: mistral`
    # (above) does lazy-install on first transcription call.
    if _HAS_MISTRAL and get_env_value("MISTRAL_API_KEY"):
        logger.info("No local STT available, using Mistral Voxtral Transcribe API")
        return "mistral"
    try:
        from tools.xai_http import resolve_xai_http_credentials

        if resolve_xai_http_credentials().get("api_key"):
            logger.info("No local STT available, using xAI Grok STT API")
            return "xai"
    except Exception:
        pass
    if get_env_value("ELEVENLABS_API_KEY"):
        logger.info("No local STT available, using ElevenLabs Scribe STT API")
        return "elevenlabs"
    return "none"


# ---------------------------------------------------------------------------
# Plugin provider dispatch (issue follow-up to #30398 — STT pluggability)
# ---------------------------------------------------------------------------


def _dispatch_to_plugin_provider(
    file_path: str,
    provider: str,
    stt_config: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route the call to a plugin-registered transcription provider, or
    return None.

    Returns the transcribe-response dict on dispatch, or ``None`` to
    fall through to the legacy "No STT provider available" error path.

    Resolution invariants enforced here:

    1. Built-in provider names short-circuit — never reach the plugin
       registry. The caller (``transcribe_audio``) handles ``local``,
       ``groq``, ``openai``, etc. via its existing elif chain; this
       function defensively rejects those names so a plugin can't be
       silently dispatched under a built-in name even if it somehow
       slipped past the registry's built-in shadow guard.
    2. Same-name command-type provider declared under
       ``stt.providers.<name>: type: command`` wins over a plugin. The
       caller short-circuits to the command runner before reaching us,
       but we re-verify here so a refactor of the caller can't silently
       break the invariant (matches TTS PR #17843 precedence rule).
    3. Plugin dispatch fires only when ``provider`` matches a
       registered :class:`TranscriptionProvider` whose ``name`` equals
       the configured value. Unknown names with no plugin registered
       return None (caller surfaces the legacy "No STT provider"
       message).
    4. Availability gating: when the matched plugin reports
       ``is_available() == False`` (missing API key, missing optional
       SDK, etc.) this returns an error envelope identifying the
       plugin as unavailable — **not** ``None`` — because the user
       explicitly opted into this plugin via ``stt.provider`` and the
       generic fallthrough message would be misleading.

    Provider exceptions are caught and converted into the standard
    error envelope (matches the legacy built-in error shapes — the
    gateway/CLI caller already expects ``{success: False, error:
    "...", transcript: ""}`` on failure).
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_STT_PROVIDERS or key == "none":
        return None
    # Defense in depth: command-provider check should already have
    # short-circuited the caller. If a same-name command config exists,
    # bail so the command path wins.
    if stt_config is not None and _is_command_stt_provider_config(
        _get_named_stt_provider_config(stt_config, key)
    ):
        return None
    try:
        from agent.transcription_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Long-lived sessions may have discovered plugins before a
            # bundled backend was patched in or before config changed.
            # Retry once with a forced refresh before surfacing fall-
            # through. Mirrors the image_gen / browser dispatcher
            # recovery pattern.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("STT plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None

    # Availability gate: when a plugin reports it's not configured
    # (missing API key, missing optional SDK, etc.) surface a clean
    # error envelope **instead of** falling through to the generic
    # "No STT provider" message. The user explicitly set
    # ``stt.provider: <plugin>`` in config — surfacing the plugin's
    # own availability failure is more actionable than the generic
    # auto-detect-failure error, and avoids routing the call into a
    # plugin that's about to crash messily.
    #
    # ``is_available()`` MUST NOT raise per the ABC contract; defend
    # anyway so a buggy plugin can't break dispatch for everyone.
    try:
        available = plugin_provider.is_available()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' is_available() raised: %s — "
            "treating as unavailable", key, exc, exc_info=True,
        )
        available = False
    if not available:
        logger.info(
            "STT plugin provider '%s' reports not available; returning "
            "unavailability envelope.", key,
        )
        return {
            "success": False,
            "transcript": "",
            "error": (
                f"STT plugin '{key}' is not available — check that its "
                "required credentials / dependencies are configured."
            ),
            "provider": key,
        }

    logger.info("Transcribing with plugin STT provider '%s'...", key)
    try:
        result = plugin_provider.transcribe(
            file_path,
            model=model,
            language=language,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' raised: %s", key, exc, exc_info=True,
        )
        return {
            "success": False,
            "transcript": "",
            "error": f"STT plugin '{key}' raised: {exc}",
            "provider": key,
        }

    # Defensive: plugins should return a dict matching the contract. If
    # they don't, surface a clear error envelope rather than leaking a
    # weird object back to the gateway.
    if not isinstance(result, dict):
        return {
            "success": False,
            "transcript": "",
            "error": f"STT plugin '{key}' returned a non-dict result",
            "provider": key,
        }
    # Stamp provider if the plugin forgot to.
    result.setdefault("provider", key)
    return result


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_audio_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Validate the audio file.  Returns an error dict or None if OK."""
    audio_path = Path(file_path)

    if os.path.islink(audio_path):
        return {"success": False, "transcript": "", "error": f"Path is a symbolic link: {file_path}"}
    if not audio_path.exists():
        return {"success": False, "transcript": "", "error": f"Audio file not found: {file_path}"}
    if not audio_path.is_file():
        return {"success": False, "transcript": "", "error": f"Path is not a file: {file_path}"}
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported format: {audio_path.suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    try:
        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "transcript": "",
                "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024):.0f}MB)",
            }
    except OSError as e:
        return {"success": False, "transcript": "", "error": f"Failed to access file: {e}"}

    return None

# ---------------------------------------------------------------------------
# Provider: local (faster-whisper)
# ---------------------------------------------------------------------------


# Substrings that identify a missing/unloadable CUDA runtime library.  When
# ctranslate2 (the backend for faster-whisper) cannot dlopen one of these, the
# "auto" device picker has already committed to CUDA and the model can no
# longer be used — we fall back to CPU and reload.
#
# Deliberately narrow: we match on library-name tokens and dlopen phrasing so
# we DO NOT accidentally catch legitimate runtime failures like "CUDA out of
# memory" — those should surface to the user, not silently fall back to CPU
# (a 32GB audio clip on CPU at int8 isn't useful either).
_CUDA_LIB_ERROR_MARKERS = (
    "libcublas",
    "libcudnn",
    "libcudart",
    "cannot be loaded",
    "cannot open shared object",
    "no kernel image is available",
    "no CUDA-capable device",
    "CUDA driver version is insufficient",
)


def _looks_like_cuda_lib_error(exc: BaseException) -> bool:
    """Heuristic: is this exception a missing/broken CUDA runtime library?

    ctranslate2 raises plain RuntimeError with messages like
    ``Library libcublas.so.12 is not found or cannot be loaded``.  We want to
    catch missing/unloadable shared libs and driver-mismatch errors, NOT
    legitimate runtime failures ("CUDA out of memory", model bugs, etc.).
    """
    msg = str(exc)
    return any(marker in msg for marker in _CUDA_LIB_ERROR_MARKERS)


def _load_local_whisper_model(model_name: str):
    """Load faster-whisper with graceful CUDA → CPU fallback.

    faster-whisper's ``device="auto"`` picks CUDA when the ctranslate2 wheel
    ships CUDA shared libs, even on hosts where the NVIDIA runtime
    (``libcublas.so.12`` / ``libcudnn*``) isn't installed — common on WSL2
    without CUDA-on-WSL, headless servers, and CPU-only developer machines.
    On those hosts the load itself sometimes succeeds and the dlopen failure
    only surfaces at first ``transcribe()`` call.

    We try ``auto`` first (fast CUDA path when it works), and on any CUDA
    library load failure fall back to CPU + int8.
    """
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(model_name, device="auto", compute_type="auto")
    except Exception as exc:
        if not _looks_like_cuda_lib_error(exc):
            raise
        logger.warning(
            "faster-whisper CUDA load failed (%s) — falling back to CPU (int8). "
            "Install the NVIDIA CUDA runtime (libcublas/libcudnn) to use GPU.",
            exc,
        )
        return WhisperModel(model_name, device="cpu", compute_type="int8")


def _transcribe_local(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using faster-whisper (local, free)."""
    global _local_model, _local_model_name

    if not _HAS_FASTER_WHISPER:
        if not _try_lazy_install_stt():
            return {"success": False, "transcript": "", "error": "faster-whisper not installed"}

    try:
        # Lazy-load the model (downloads on first use, ~150 MB for 'base')
        if _local_model is None or _local_model_name != model_name:
            logger.info("Loading faster-whisper model '%s' (first load downloads the model)...", model_name)
            _local_model = _load_local_whisper_model(model_name)
            _local_model_name = model_name

        # Language: config.yaml (stt.local.language) > env var > auto-detect.
        local_config = _load_stt_config().get("local") or {}
        _forced_lang = (
            local_config.get("language")
            or os.getenv(LOCAL_STT_LANGUAGE_ENV)
            or None
        )
        transcribe_kwargs = {"beam_size": 5}
        if _forced_lang:
            transcribe_kwargs["language"] = _forced_lang
        initial_prompt = local_config.get("initial_prompt")
        if isinstance(initial_prompt, str) and initial_prompt.strip():
            transcribe_kwargs["initial_prompt"] = initial_prompt

        try:
            segments, info = _local_model.transcribe(file_path, **transcribe_kwargs)
            transcript = " ".join(segment.text.strip() for segment in segments)
        except Exception as exc:
            # CUDA runtime libs sometimes only fail at dlopen-on-first-use,
            # AFTER the model loaded successfully.  Evict the broken cached
            # model, reload on CPU, retry once.  Without this the module-
            # global `_local_model` is poisoned and every subsequent voice
            # message on this process fails identically until restart.
            if not _looks_like_cuda_lib_error(exc):
                raise
            logger.warning(
                "faster-whisper CUDA runtime failed mid-transcribe (%s) — "
                "evicting cached model and retrying on CPU (int8).",
                exc,
            )
            _local_model = None
            _local_model_name = None
            from faster_whisper import WhisperModel
            _local_model = WhisperModel(model_name, device="cpu", compute_type="int8")
            _local_model_name = model_name
            segments, info = _local_model.transcribe(file_path, **transcribe_kwargs)
            transcript = " ".join(segment.text.strip() for segment in segments)

        logger.info(
            "Transcribed %s via local whisper (%s, lang=%s, %.1fs audio)",
            Path(file_path).name, model_name, info.language, info.duration,
        )

        return {"success": True, "transcript": transcript, "provider": "local"}

    except Exception as e:
        logger.error("Local transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Local transcription failed: {e}"}


def _prepare_local_audio(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Normalize audio for local CLI STT when needed."""
    audio_path = Path(file_path)
    if audio_path.suffix.lower() in LOCAL_NATIVE_AUDIO_FORMATS:
        return file_path, None

    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "Local STT fallback requires ffmpeg for non-WAV inputs, but ffmpeg was not found"

    converted_path = os.path.join(work_dir, f"{audio_path.stem}.wav")
    command = [ffmpeg, "-y", "-i", file_path, converted_path]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
        return converted_path, None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out for %s", file_path)
        return None, "Audio conversion for local STT timed out"
    except subprocess.CalledProcessError as e:
        details = e.stderr.strip() or e.stdout.strip() or str(e)
        logger.error("ffmpeg conversion failed for %s: %s", file_path, details)
        return None, f"Failed to convert audio for local STT: {details}"


def _transcribe_local_command(file_path: str, model_name: str) -> Dict[str, Any]:
    """Run the configured local STT command template and read back a .txt transcript."""
    command_template = _get_local_command_template()
    if not command_template:
        return {
            "success": False,
            "transcript": "",
            "error": (
                f"{LOCAL_STT_COMMAND_ENV} not configured and no local whisper binary was found"
            ),
        }

    # Language: config.yaml (stt.local.language) > env var > "en" default.
    language = (
        (_load_stt_config().get("local") or {}).get("language")
        or os.getenv(LOCAL_STT_LANGUAGE_ENV)
        or DEFAULT_LOCAL_STT_LANGUAGE
    )
    normalized_model = _normalize_local_command_model(model_name)

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-local-stt-") as output_dir:
            prepared_input, prep_error = _prepare_local_audio(file_path, output_dir)
            if prep_error:
                return {"success": False, "transcript": "", "error": prep_error}

            command = command_template.format(
                input_path=shlex.quote(prepared_input),
                output_dir=shlex.quote(output_dir),
                language=shlex.quote(language),
                model=shlex.quote(normalized_model),
            )
            # User-provided templates (env var) may contain shell syntax; auto-detected commands are safe for list mode.
            use_shell = bool(os.getenv(LOCAL_STT_COMMAND_ENV, "").strip())
            if use_shell:
                subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            else:
                subprocess.run(shlex.split(command), check=True, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            

            txt_files = sorted(Path(output_dir).glob("*.txt"))
            if not txt_files:
                return {
                    "success": False,
                    "transcript": "",
                    "error": "Local STT command completed but did not produce a .txt transcript",
                }

            transcript_text = txt_files[0].read_text(encoding="utf-8").strip()
            logger.info(
                "Transcribed %s via local STT command (%s, %d chars)",
                Path(file_path).name,
                normalized_model,
                len(transcript_text),
            )
            return {"success": True, "transcript": transcript_text, "provider": "local_command"}

    except KeyError as e:
        return {
            "success": False,
            "transcript": "",
            "error": f"Invalid {LOCAL_STT_COMMAND_ENV} template, missing placeholder: {e}",
        }
    except subprocess.CalledProcessError as e:
        details = e.stderr.strip() or e.stdout.strip() or str(e)
        logger.error("Local STT command failed for %s: %s", file_path, details)
        return {"success": False, "transcript": "", "error": f"Local STT failed: {details}"}
    except Exception as e:
        logger.error("Unexpected error during local command transcription: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Local transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Provider: groq (Whisper API — free tier)
# ---------------------------------------------------------------------------


def _transcribe_groq(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using Groq Whisper API (free tier available)."""
    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "GROQ_API_KEY not set"}

    if not _HAS_OPENAI:
        return {"success": False, "transcript": "", "error": "openai package not installed"}

    # Auto-correct model if caller passed an OpenAI-only model
    if model_name in OPENAI_MODELS:
        logger.info("Model %s not available on Groq, using %s", model_name, DEFAULT_GROQ_STT_MODEL)
        model_name = DEFAULT_GROQ_STT_MODEL

    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30, max_retries=0)
        try:
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    response_format="text",
                )

            transcript_text = str(transcription).strip()
            logger.info("Transcribed %s via Groq API (%s, %d chars)",
                         Path(file_path).name, model_name, len(transcript_text))

            return {"success": True, "transcript": transcript_text, "provider": "groq"}
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except APIConnectionError as e:
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except APITimeoutError as e:
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except APIError as e:
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("Groq transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Provider: openai (Whisper API)
# ---------------------------------------------------------------------------


def _transcribe_openai(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using OpenAI Whisper API (paid)."""
    try:
        api_key, base_url = _resolve_openai_audio_client_config()
    except ValueError as exc:
        return {
            "success": False,
            "transcript": "",
            "error": str(exc),
        }

    if not _HAS_OPENAI:
        return {"success": False, "transcript": "", "error": "openai package not installed"}

    # Auto-correct model if caller passed a Groq-only model
    if model_name in GROQ_MODELS:
        logger.info("Model %s not available on OpenAI, using %s", model_name, DEFAULT_STT_MODEL)
        model_name = DEFAULT_STT_MODEL

    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)
        try:
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    response_format="text" if model_name == "whisper-1" else "json",
                )

            transcript_text = _extract_transcript_text(transcription)
            logger.info("Transcribed %s via OpenAI API (%s, %d chars)",
                         Path(file_path).name, model_name, len(transcript_text))

            return {"success": True, "transcript": transcript_text, "provider": "openai"}
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except APIConnectionError as e:
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except APITimeoutError as e:
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except APIError as e:
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("OpenAI transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Provider: mistral (Voxtral Transcribe API)
# ---------------------------------------------------------------------------


def _transcribe_mistral(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using Mistral Voxtral Transcribe API.

    Uses the ``mistralai`` Python SDK to call ``/v1/audio/transcriptions``.
    Requires ``MISTRAL_API_KEY`` environment variable.
    """
    api_key = get_env_value("MISTRAL_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "MISTRAL_API_KEY not set"}

    try:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("stt.mistral", prompt=False)
        except ImportError:
            pass
        from mistralai.client import Mistral

        with Mistral(api_key=api_key) as client:
            with open(file_path, "rb") as audio_file:
                result = client.audio.transcriptions.complete(
                    model=model_name,
                    file={"content": audio_file, "file_name": Path(file_path).name},
                )

            transcript_text = _extract_transcript_text(result)
            logger.info(
                "Transcribed %s via Mistral API (%s, %d chars)",
                Path(file_path).name, model_name, len(transcript_text),
            )
            return {"success": True, "transcript": transcript_text, "provider": "mistral"}

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except Exception as e:
        logger.error("Mistral transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Mistral transcription failed: {type(e).__name__}"}


# ---------------------------------------------------------------------------
# Provider: xAI (Grok STT API)
# ---------------------------------------------------------------------------


def _transcribe_xai(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using xAI Grok STT API.

    Uses the ``POST /v1/stt`` REST endpoint with multipart/form-data.
    Supports Inverse Text Normalization, diarization, and word-level timestamps.
    Requires ``XAI_API_KEY`` environment variable.
    """
    from tools.xai_http import resolve_xai_http_credentials

    creds = resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        return {
            "success": False,
            "transcript": "",
            "error": "No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY",
        }

    stt_config = _load_stt_config()
    xai_config = stt_config.get("xai") or {}
    base_url = str(
        xai_config.get("base_url")
        or get_env_value("XAI_STT_BASE_URL")
        or creds.get("base_url")
        or XAI_STT_BASE_URL
    ).strip().rstrip("/")
    language = str(
        xai_config.get("language")
        or os.getenv("HERMES_LOCAL_STT_LANGUAGE")
        or DEFAULT_LOCAL_STT_LANGUAGE
    ).strip()
    # .get("format", True) already defaults to True when the key is absent;
    # is_truthy_value only normalizes truthy/falsy strings from config.
    use_format = is_truthy_value(xai_config.get("format", True))
    use_diarize = is_truthy_value(xai_config.get("diarize", False))

    try:
        import requests
        from tools.xai_http import hermes_xai_user_agent

        data: Dict[str, str] = {}
        if language:
            data["language"] = language
        if use_format:
            data["format"] = "true"
        if use_diarize:
            data["diarize"] = "true"

        with open(file_path, "rb") as audio_file:
            response = requests.post(
                f"{base_url}/stt",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": hermes_xai_user_agent(),
                },
                files={
                    "file": (Path(file_path).name, audio_file),
                },
                data=data,
                timeout=120,
            )

        if response.status_code != 200:
            detail = ""
            try:
                err_body = response.json()
                detail = err_body.get("error", {}).get("message", "") or response.text[:300]
            except Exception:
                detail = response.text[:300]
            return {
                "success": False,
                "transcript": "",
                "error": f"xAI STT API error (HTTP {response.status_code}): {detail}",
            }

        result = response.json()
        transcript_text = result.get("text", "").strip()

        if not transcript_text:
            return {
                "success": False,
                "transcript": "",
                "error": "xAI STT returned empty transcript",
            }

        logger.info(
            "Transcribed %s via xAI Grok STT (lang=%s, %.1fs audio, %d chars)",
            Path(file_path).name,
            result.get("language", language),
            result.get("duration", 0),
            len(transcript_text),
        )

        return {"success": True, "transcript": transcript_text, "provider": "xai"}

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except Exception as e:
        logger.error("xAI STT transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"xAI STT transcription failed: {e}"}


# ---------------------------------------------------------------------------
# Provider: ElevenLabs (Scribe STT API)
# ---------------------------------------------------------------------------


def _transcribe_elevenlabs(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using ElevenLabs Scribe STT API."""
    api_key = get_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "ELEVENLABS_API_KEY not set"}

    stt_config = _load_stt_config()
    elevenlabs_config = stt_config.get("elevenlabs") or {}
    base_url = str(
        elevenlabs_config.get("base_url")
        or get_env_value("ELEVENLABS_STT_BASE_URL")
        or ELEVENLABS_STT_BASE_URL
    ).strip().rstrip("/")
    language_code = str(elevenlabs_config.get("language_code") or "").strip()
    tag_audio_events = is_truthy_value(elevenlabs_config.get("tag_audio_events", False))
    diarize = is_truthy_value(elevenlabs_config.get("diarize", False))

    try:
        import requests

        data: Dict[str, str] = {
            "model_id": model_name,
            "tag_audio_events": "true" if tag_audio_events else "false",
            "diarize": "true" if diarize else "false",
        }
        if language_code:
            data["language_code"] = language_code

        with open(file_path, "rb") as audio_file:
            response = requests.post(
                f"{base_url}/speech-to-text",
                headers={"xi-api-key": api_key},
                files={"file": (Path(file_path).name, audio_file)},
                data=data,
                timeout=120,
            )

        if response.status_code != 200:
            detail = ""
            try:
                err_body = response.json()
                error_value = err_body.get("detail") or err_body.get("error")
                if isinstance(error_value, dict):
                    detail = str(error_value.get("message") or error_value)
                elif error_value:
                    detail = str(error_value)
                else:
                    detail = response.text[:300]
            except Exception:
                detail = response.text[:300]
            return {
                "success": False,
                "transcript": "",
                "error": f"ElevenLabs STT API error (HTTP {response.status_code}): {detail}",
            }

        result = response.json()
        transcript_text = _extract_transcript_text(result)
        if not transcript_text:
            return {
                "success": False,
                "transcript": "",
                "error": "ElevenLabs STT returned empty transcript",
            }

        logger.info(
            "Transcribed %s via ElevenLabs Scribe (%s, %d chars)",
            Path(file_path).name,
            model_name,
            len(transcript_text),
        )

        return {"success": True, "transcript": transcript_text, "provider": "elevenlabs"}

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except Exception as e:
        logger.error("ElevenLabs STT transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"ElevenLabs STT transcription failed: {e}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_audio(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file using the configured STT provider.

    Provider priority:
      1. User config (``stt.provider`` in config.yaml)
      2. Auto-detect: local > Groq > OpenAI > Mistral > xAI > ElevenLabs

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Override the model. If None, uses config or provider default.

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
          - "provider" (str, optional): Which provider was used
    """
    # Validate input
    error = _validate_audio_file(file_path)
    if error:
        return error

    # Load config and determine provider
    stt_config = _load_stt_config()
    if not is_stt_enabled(stt_config):
        return {
            "success": False,
            "transcript": "",
            "error": "STT is disabled in config.yaml (stt.enabled: false).",
        }

    provider = _get_provider(stt_config)

    if provider == "local":
        local_cfg = stt_config.get("local") or {}
        model_name = _normalize_local_model(
            model or local_cfg.get("model", DEFAULT_LOCAL_MODEL)
        )
        return _transcribe_local(file_path, model_name)

    if provider == "local_command":
        local_cfg = stt_config.get("local") or {}
        model_name = _normalize_local_command_model(
            model or local_cfg.get("model", DEFAULT_LOCAL_MODEL)
        )
        return _transcribe_local_command(file_path, model_name)

    if provider == "groq":
        model_name = model or DEFAULT_GROQ_STT_MODEL
        return _transcribe_groq(file_path, model_name)

    if provider == "openai":
        openai_cfg = stt_config.get("openai") or {}
        model_name = model or openai_cfg.get("model", DEFAULT_STT_MODEL)
        return _transcribe_openai(file_path, model_name)

    if provider == "mistral":
        mistral_cfg = stt_config.get("mistral") or {}
        model_name = model or mistral_cfg.get("model", DEFAULT_MISTRAL_STT_MODEL)
        return _transcribe_mistral(file_path, model_name)

    if provider == "xai":
        # xAI Grok STT doesn't use a model parameter — pass through for logging
        model_name = model or "grok-stt"
        return _transcribe_xai(file_path, model_name)

    if provider == "elevenlabs":
        elevenlabs_cfg = stt_config.get("elevenlabs") or {}
        model_name = model or elevenlabs_cfg.get("model_id", DEFAULT_ELEVENLABS_STT_MODEL)
        return _transcribe_elevenlabs(file_path, model_name)

    # User-declared command-type provider
    # (``stt.providers.<name>: type: command``). Fires after the built-in
    # elif chain — built-in names short-circuit upstream so a user's
    # ``stt.providers.openai.command`` can't override the real OpenAI
    # handler — and BEFORE the plugin dispatcher, because config is more
    # local than a plugin install (same precedence rule as TTS PR #17843).
    command_provider_config = _resolve_command_stt_provider_config(provider, stt_config)
    if command_provider_config is not None:
        return _transcribe_command_stt(
            file_path,
            provider,
            command_provider_config,
            stt_config,
            model_override=model,
        )

    # Plugin-registered STT backend (e.g. OpenRouter, SenseAudio,
    # Gemini-STT). Fires only when ``provider`` is neither a built-in
    # nor ``"none"`` AND there is no same-name command provider. The
    # dispatcher enforces built-ins-always-win + command-wins-over-plugin
    # defensively. Returns None when no plugin is registered for the
    # configured name, falling through to the legacy "No STT provider"
    # error message below.
    #
    # Plugin-scoped config namespace mirrors the built-in pattern
    # (``stt.openai.model``, ``stt.mistral.model``): plugins read their
    # per-provider config under ``stt.<provider>`` and the dispatcher
    # forwards ``language`` from there. Top-level ``model`` argument
    # overrides any config-set model.
    plugin_cfg = stt_config.get(provider, {}) if isinstance(stt_config.get(provider), dict) else {}
    plugin_language = plugin_cfg.get("language")
    plugin_model = model or plugin_cfg.get("model")
    plugin_result = _dispatch_to_plugin_provider(
        file_path,
        provider,
        stt_config,
        model=plugin_model,
        language=plugin_language,
    )
    if plugin_result is not None:
        return plugin_result

    # No provider available
    return {
        "success": False,
        "transcript": "",
        "error": (
            "No STT provider available. Install faster-whisper for free local "
            f"transcription, configure {LOCAL_STT_COMMAND_ENV} or install a local whisper CLI, "
            "set GROQ_API_KEY for free Groq Whisper, set MISTRAL_API_KEY for Mistral "
            "Voxtral Transcribe, configure xAI OAuth or set XAI_API_KEY for xAI Grok STT, "
            "set ELEVENLABS_API_KEY for ElevenLabs Scribe, or set VOICE_TOOLS_OPENAI_KEY "
            "or OPENAI_API_KEY for the OpenAI Whisper API."
        ),
    }


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """Return direct OpenAI audio config or a managed gateway fallback."""
    stt_config = _load_stt_config()
    openai_cfg = stt_config.get("openai") or {}
    cfg_api_key = openai_cfg.get("api_key", "")
    cfg_base_url = openai_cfg.get("base_url", "")
    if cfg_api_key:
        return cfg_api_key, (cfg_base_url or OPENAI_BASE_URL)

    direct_api_key = resolve_openai_audio_api_key()
    if direct_api_key:
        return direct_api_key, OPENAI_BASE_URL

    managed_gateway = resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        message = "Neither stt.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        if managed_nous_tools_enabled():
            message += (
                ". "
                + nous_tool_gateway_unavailable_message(
                    "managed OpenAI audio for transcription",
                )
            )
        raise ValueError(message)

    return managed_gateway.nous_user_token, urljoin(
        f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"
    )


def _extract_transcript_text(transcription: Any) -> str:
    """Normalize text and JSON transcription responses to a plain string."""
    if isinstance(transcription, str):
        return transcription.strip()

    if hasattr(transcription, "text"):
        value = getattr(transcription, "text")
        if isinstance(value, str):
            return value.strip()

    if isinstance(transcription, dict):
        value = transcription.get("text")
        if isinstance(value, str):
            return value.strip()

    return str(transcription).strip()
