"""User-declared and plugin STT providers.

``stt.providers.<name>: type: command`` registry, plugin-registered
``TranscriptionProvider`` dispatch, and the ``pre_transcription`` hook that
threads prompt/language/model overrides into every backend. ``_resolve_stt_language``
is read lazily from ``tools.transcription_tools``.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from tools.tts_command_provider import (
    _command_output_format, _command_timeout, _is_command_provider_config as _is_command_stt_provider_config,
    _named_provider_config, _resolve_command_config, command_env_passthrough as _command_stt_env_passthrough,
    command_failure_detail, render_command_template as _render_command_stt_template,
    run_command_provider as _run_command_stt)
from tools.transcription_common import (
    BUILTIN_STT_PROVIDERS, _error_result, _log_prompt_unsupported, _ok_result)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")


# ---- Command-provider registry (``stt.providers.<name>: type: command``) ---
#
# Mirrors the TTS command-provider registry (same placeholder grammar, quote-aware
# rendering, process-tree termination on timeout). Resolution order: built-in name
# (always wins) > stt.providers.<name> command > plugin TranscriptionProvider >
# "No STT provider available". The single-env-var HERMES_LOCAL_STT_COMMAND escape
# hatch stays untouched via the built-in ``local_command`` path.
# Lets any whisper CLI / ASR CLI / curl pipeline become an STT backend with zero Python. 1. Built-in
# (``local``, ``local_command``, ``groq``, ``openai``, ``mistral``, ``xai``)              → native handler.
# **Always wins.** 2. 3. 4. Use the command-provider registry when you want MULTIPLE shell-driven STT
# engines, or you want a named provider you can pick via ``stt.provider`` in config.yaml. See #17843.
DEFAULT_COMMAND_STT_TIMEOUT_SECONDS = 300
DEFAULT_COMMAND_STT_LANGUAGE = "en"
DEFAULT_COMMAND_STT_OUTPUT_FORMAT = "txt"
COMMAND_STT_OUTPUT_FORMATS = frozenset({"txt", "json", "srt", "vtt"})
_NON_COMMAND_STT_NAMES = frozenset(BUILTIN_STT_PROVIDERS | {"none"})


# ``stt.providers.<name>`` (canonical), else ``stt.<name>`` for non-built-in names only.
_get_named_stt_provider_config = partial(_named_provider_config, builtins=BUILTIN_STT_PROVIDERS)
# The provider config if it is a command type; None for built-ins, ``none``, unknown.
_resolve_command_stt_provider_config = partial(_resolve_command_config,
                                               reserved=_NON_COMMAND_STT_NAMES)
_get_command_stt_timeout = partial(_command_timeout, default=DEFAULT_COMMAND_STT_TIMEOUT_SECONDS)
_get_command_stt_output_format = partial(_command_output_format, formats=COMMAND_STT_OUTPUT_FORMATS,
                                         default=DEFAULT_COMMAND_STT_OUTPUT_FORMAT)


def _read_command_stt_output(output_path: Path, stdout: str, fmt: str) -> str:
    """Transcript: non-empty output file > non-empty stdout (curl one-liners) > RuntimeError. JSON is returned raw."""
    content = (output_path.read_bytes().decode("utf-8", errors="replace").strip()
               if output_path.exists() else "")
    if content or (stdout or "").strip():
        return content or stdout.strip()
    raise RuntimeError(f"Command STT provider wrote no output file at {output_path} and produced no stdout")


def _transcribe_command_stt(
    file_path: str, provider_name: str, config: Dict[str, Any], stt_config: Dict[str, Any],
    model_override: Optional[str] = None, language_override: Optional[str] = None,
    prompt: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe via a user-declared ``stt.providers.<name>: type: command``. Placeholders
    (shell-quote-aware; ``{{``/``}}`` stay literal): ``{input_path}``, ``{output_path}`` (transcript
    file), ``{output_dir}``, ``{format}`` txt/json/srt/vtt, ``{language}`` (default ``en``),
    ``{model}`` (empty when unset)."""
    from tools.transcription_tools import _resolve_stt_language
    if prompt:
        _log_prompt_unsupported(f"Command STT provider '{provider_name}'")

    def fail(error: str) -> Dict[str, Any]:
        return _error_result(error, provider=provider_name)
    command_template = str(config.get("command") or "").strip()
    if not command_template:
        return fail(f"stt.providers.{provider_name}.command is not configured")
    audio = Path(file_path).expanduser()
    if not audio.exists():
        return fail(f"Audio file not found: {file_path}")
    timeout = _get_command_stt_timeout(config)
    output_format = _get_command_stt_output_format(config)
    language = (language_override or config.get("language")
                or _resolve_stt_language(provider_name, stt_config) or DEFAULT_COMMAND_STT_LANGUAGE)
    try:
        with tempfile.TemporaryDirectory(prefix=f"hermes-cmd-stt-{provider_name}-") as tmpdir:
            output_path = Path(tmpdir) / f"transcript.{output_format}"
            command = _render_command_stt_template(command_template, {
                "input_path": str(audio.resolve()), "output_path": str(output_path),
                "output_dir": str(output_path.parent), "format": output_format,
                "language": str(language), "model": str(model_override or config.get("model") or ""),
            })
            logger.info("Transcribing %s via command STT provider '%s'...", audio.name, provider_name)
            result = _run_command_stt(command, timeout, env_passthrough=_command_stt_env_passthrough(config))
            transcript_text = _read_command_stt_output(output_path, result.stdout or "", output_format)
    except subprocess.TimeoutExpired:
        return fail(f"STT command provider '{provider_name}' timed out after {timeout:g}s")
    except subprocess.CalledProcessError as exc:
        return fail(
            f"STT command provider '{provider_name}' exited with code {exc.returncode}: {command_failure_detail(exc)}"
        )
    except RuntimeError as exc:
        return fail(str(exc))
    except OSError as exc:
        return fail(f"STT command provider '{provider_name}' failed: {exc}")
    logger.info("Transcribed %s via command STT provider '%s' (%d chars)", audio.name, provider_name, len(transcript_text))
    return _ok_result(transcript_text, provider_name)


def _unregistered_stt_provider_error(provider: str) -> Dict[str, Any]:
    key = str(provider or "").strip()
    return _error_result(
        f"stt.provider='{key}' is set but no built-in, command, or plugin "
        "provider registered that name. Run `hermes plugins list` to see "
        "installed STT plugins, or configure a command provider under "
        f"`stt.providers.{key}.command`.",
        provider=key,
        error_type="provider_not_registered")


# --------------------------------------------------------------------------- Plugin provider dispatch
# (issue follow-up to #30398 — STT pluggability)
# ---------------------------------------------------------------------------
def _dispatch_to_plugin_provider(
    file_path: str, provider: str, stt_config: Optional[Dict[str, Any]] = None, *,
    model: Optional[str] = None, language: Optional[str] = None, prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route to a plugin-registered transcription provider; None when no plugin claims the name.
    Invariants re-verified here so a caller refactor can't break them: built-in names never reach
    the registry; a same-name command provider wins over a plugin. A matched plugin with
    ``is_available() == False`` returns an error envelope — not None — because the user
    explicitly opted in via ``stt.provider``. Provider exceptions become the error envelope."""
    key = (provider or "").lower().strip()
    if not key or key in _NON_COMMAND_STT_NAMES:
        return None
    if stt_config is not None and _is_command_stt_provider_config(_get_named_stt_provider_config(stt_config, key)):
        return None
    try:
        from agent.transcription_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Long-lived sessions may have discovered plugins before a backend
            # was patched in or config changed — retry once with a forced refresh.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("STT plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None
    # ``is_available()`` MUST NOT raise per the ABC contract; defend anyway so
    # a buggy plugin can't break dispatch for everyone.
    try:
        available = plugin_provider.is_available()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' is_available() raised: %s — treating as unavailable", key, exc, exc_info=True,
        )
        available = False
    if not available:
        logger.info("STT plugin provider '%s' reports not available; returning unavailability envelope.", key)
        return _error_result(
            f"STT plugin '{key}' is not available — check that its required credentials / dependencies are configured.",
            provider=key)
    logger.info("Transcribing with plugin STT provider '%s'...", key)
    # The prompt travels via the ABC's ``**extra`` kwargs and is only sent when
    # set, so pre-prompt providers see byte-identical calls on the no-prompt path.
    extra_kwargs: Dict[str, Any] = {} if prompt is None else {"prompt": prompt}
    try:
        result = plugin_provider.transcribe(file_path, model=model, language=language, **extra_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("STT plugin provider '%s' raised: %s", key, exc, exc_info=True)
        return _error_result(f"STT plugin '{key}' raised: {exc}", provider=key)
    if not isinstance(result, dict):
        return _error_result(f"STT plugin '{key}' returned a non-dict result", provider=key)
    result.setdefault("provider", key)
    return result


# Fields a pre_transcription hook may mutate; ``file_path`` is read-only (logged and dropped).
# --------------------------------------------------------------------------- pre_transcription plugin hook
# (issue #64168 — STT prompt/vocab threading)
# ---------------------------------------------------------------------------
_PRE_TRANSCRIPTION_MUTABLE_FIELDS = ("prompt", "language", "model")

# Whisper-family models only use the final ~224 tokens of the prompt; longer values
# waste upload bytes and can trip stricter OpenAI-compatible servers. Enforced
# client-side (truncate with a warning, never error), ~4 chars/token.
_WHISPER_PROMPT_TOKEN_CAP = 224
_PROMPT_CHARS_PER_TOKEN = 4
_WHISPER_PROMPT_CAPPED_PROVIDERS = frozenset({"local", "openai", "groq", "deepinfra"})


def _enforce_prompt_length_limit(prompt: Optional[str], provider: str) -> Optional[str]:
    """Truncate *prompt* to the whisper-family token cap, keeping the TAIL (whisper conditions
    on the final context window, so the newest hints survive). Other providers self-validate."""
    max_chars = _WHISPER_PROMPT_TOKEN_CAP * _PROMPT_CHARS_PER_TOKEN
    if not prompt or provider not in _WHISPER_PROMPT_CAPPED_PROVIDERS or len(prompt) <= max_chars:
        return prompt
    logger.warning(
        "Transcription prompt is ~%d tokens; whisper-family provider '%s' "
        "only uses the final ~%d — truncating to the last %d characters.",
        len(prompt) // _PROMPT_CHARS_PER_TOKEN, provider, _WHISPER_PROMPT_TOKEN_CAP, max_chars)
    return prompt[-max_chars:]


def _apply_pre_transcription_hook(
    *, file_path: str, provider: str, model: Optional[str], language: Optional[str],
    prompt: Optional[str], source: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fire the ``pre_transcription`` plugin hook; returns ``(model, language_override, prompt)``.
    Gated on ``has_hook`` (the no-hook path never builds kwargs) and fail-open: any plumbing error
    leaves the dispatch untouched. Results apply field-by-field in registration order (last hook
    wins). ``language_override`` is None unless a hook explicitly set ``language``, so backends keep
    their own config/env resolution."""
    try:
        from hermes_cli.plugins import has_hook, invoke_hook
        if not has_hook("pre_transcription"):
            return model, None, prompt
        hook_results = invoke_hook(
            "pre_transcription", file_path=file_path, provider=provider,
            model=model, language=language, prompt=prompt, source=source)
        overrides: Dict[str, Any] = {}
        for hook_result in hook_results:
            for key, value in (hook_result.items() if isinstance(hook_result, dict) else ()):
                if key == "file_path":
                    logger.warning(
                        "pre_transcription hook attempted to change "
                        "file_path (read-only) — ignoring the attempt.")
                elif key not in _PRE_TRANSCRIPTION_MUTABLE_FIELDS:
                    logger.debug("pre_transcription hook returned unsupported field %r — ignoring.", key)
                elif not isinstance(value, str):
                    logger.debug(
                        "pre_transcription hook returned non-string value %r for field %r — ignoring.", value, key,
                    )
                else:
                    overrides[key] = value
        # Hooks win over the static ``stt.prompt`` config; "" clears it.
        if "prompt" in overrides:
            prompt = overrides["prompt"] or None
        return overrides.get("model", model), overrides.get("language") or None, prompt
    except Exception as _hook_err:  # noqa: BLE001 — hook plumbing is fail-open
        logger.debug("pre_transcription hook error: %s", _hook_err)
        return model, None, prompt
