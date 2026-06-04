#!/usr/bin/env python3
"""
Audio Generation Tool
=====================

Single ``audio_generate`` tool that dispatches to a plugin-registered
audio generation provider. Mirrors the ``video_generate`` design:

- ``agent/audio_gen_provider.py`` defines the :class:`AudioGenProvider` ABC.
- ``agent/audio_gen_registry.py`` holds the active providers (populated by
  plugins at import time).
- Each provider lives under ``plugins/audio_gen/<name>/``.

The tool itself is intentionally backend-agnostic and ships **no in-tree
provider** — turn on a backend by enabling a plugin (``hermes plugins
enable audio_gen/<name>``) and selecting it in ``hermes tools`` → Audio
Generation.

Audio generation vs. text-to-speech
------------------------------------
``audio_generate`` composes **new** audio (a song, a soundscape, a sound
effect) from a creative prompt. It is distinct from ``text_to_speech``,
which reads a fixed string aloud in a chosen voice. Reach for this tool
when the user asks for music, a jingle, ambient sound, or an SFX —
not for narration.

Unified surface
---------------
One tool covers the common cases with a compact schema:

    prompt              text instruction (required)
    duration            seconds (provider clamps)
    audio_format        "mp3" | "wav" | "ogg" | "flac"
    negative_prompt     optional (content to avoid)
    lyrics              optional (lyrics for vocal music models)
    seed                optional
    model               optional, override the active provider's default

Providers ignore parameters they do not support. The tool layer does
**lightweight** validation (required prompt) and lets each provider do its
own clamping inside :meth:`AudioGenProvider.generate`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.audio_gen_provider import (
    COMMON_AUDIO_FORMATS,
    DEFAULT_AUDIO_FORMAT,
    error_response,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


AUDIO_GENERATE_SCHEMA: Dict[str, Any] = {
    "name": "audio_generate",
    # Placeholder — the real description is built dynamically at
    # get_tool_definitions() time so it reflects the active backend's
    # actual capabilities. See _build_dynamic_audio_schema() below.
    "description": "(rebuilt at get_definitions() time — see _build_dynamic_audio_schema)",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text instruction describing the desired audio — the "
                    "genre, mood, instruments, tempo, or sound to compose. "
                    "Examples: 'upbeat lo-fi hip hop with a mellow piano', "
                    "'rain falling on a tin roof with distant thunder', "
                    "'an 8-bit victory jingle'."
                ),
            },
            "duration": {
                "type": "integer",
                "description": (
                    "Desired audio duration in seconds. Providers clamp to "
                    "their supported range. Omit to use the provider's "
                    "default."
                ),
            },
            "audio_format": {
                "type": "string",
                "enum": list(COMMON_AUDIO_FORMATS),
                "description": (
                    "Output container/codec. Providers clamp to their "
                    "supported set."
                ),
                "default": DEFAULT_AUDIO_FORMAT,
            },
            "negative_prompt": {
                "type": "string",
                "description": (
                    "Optional negative prompt — content to avoid in the "
                    "output. Supported by some backends; ignored by "
                    "providers that do not support it."
                ),
            },
            "lyrics": {
                "type": "string",
                "description": (
                    "Optional lyrics for vocal music models (e.g. Lyria's "
                    "song mode). Ignored by instrumental-only or SFX "
                    "backends."
                ),
            },
            "seed": {
                "type": "integer",
                "description": (
                    "Optional seed for reproducible outputs (provider-"
                    "dependent)."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override. If omitted, the user's "
                    "configured ``audio_gen.model`` (set via `hermes tools` "
                    "→ Audio Generation) is used. Models that the active "
                    "provider does not know are rejected."
                ),
            },
        },
        "required": ["prompt"],
    },
}


# ---------------------------------------------------------------------------
# Config readers (mirror video_generation_tool.py)
# ---------------------------------------------------------------------------


def _read_audio_gen_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("audio_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not read audio_gen config: %s", exc)
        return {}


def _read_configured_audio_provider() -> Optional[str]:
    value = _read_audio_gen_section().get("provider")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_configured_audio_model() -> Optional[str]:
    value = _read_audio_gen_section().get("model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def check_audio_generation_requirements() -> bool:
    """Return True when at least one registered provider reports available.

    Triggers plugin discovery (idempotent) so user-installed plugins are
    visible to the toolset gate.
    """
    try:
        from agent.audio_gen_registry import list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        for provider in list_providers():
            try:
                if provider.is_available():
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _resolve_active_provider():
    """Return the active provider object or None.

    Forces plugin discovery before checking the registry — handles cases
    where a long-lived session was started before a plugin was installed.
    """
    try:
        from agent.audio_gen_registry import get_active_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_active_provider()
        if provider is None:
            _ensure_plugins_discovered(force=True)
            provider = get_active_provider()
        return provider
    except Exception as exc:
        logger.debug("audio_gen provider resolution failed: %s", exc)
        return None


def _missing_provider_error(configured: Optional[str]) -> str:
    if configured:
        msg = (
            f"audio_gen.provider='{configured}' is set but no plugin "
            f"registered that name. Run `hermes plugins list` to see "
            f"installed audio gen backends, or `hermes tools` → Audio "
            f"Generation to pick one."
        )
        return json.dumps(error_response(
            error=msg, error_type="provider_not_registered",
            provider=configured,
        ))
    msg = (
        "No audio generation backend is configured. Run `hermes tools` → "
        "Audio Generation to enable one (e.g. OpenRouter for Lyria music)."
    )
    return json.dumps(error_response(
        error=msg, error_type="no_provider_configured",
    ))


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str:
    """Return a stripped string, or '' for non-string/None inputs.

    Tool-call args are advisory — a caller may send a non-string despite the
    schema. Coercing here keeps the handler from raising AttributeError on
    ``.strip()`` and lets it return a clean tool_error instead.
    """
    return value.strip() if isinstance(value, str) else ""


def _handle_audio_generate(args: Dict[str, Any], **_kw: Any) -> str:
    prompt = _coerce_str(args.get("prompt"))
    duration = _coerce_int(args.get("duration"))
    audio_format = _coerce_str(args.get("audio_format")) or DEFAULT_AUDIO_FORMAT
    negative_prompt = _coerce_str(args.get("negative_prompt")) or None
    lyrics = _coerce_str(args.get("lyrics")) or None
    seed = _coerce_int(args.get("seed"))
    model_override = _coerce_str(args.get("model")) or None

    if not prompt:
        return tool_error("prompt is required for audio generation")

    configured = _read_configured_audio_provider()
    provider = _resolve_active_provider()
    if provider is None:
        return _missing_provider_error(configured)

    # Resolve model: explicit arg wins, then config, then provider default.
    model = model_override or _read_configured_audio_model() or provider.default_model()

    kwargs: Dict[str, Any] = {
        "model": model,
        "_model_override_explicit": bool(model_override),
        "duration": duration,
        "audio_format": audio_format,
        "negative_prompt": negative_prompt,
        "lyrics": lyrics,
        "seed": seed,
    }
    # Drop None entries so providers see clean defaults.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        result = provider.generate(prompt=prompt, **kwargs)
    except TypeError as exc:
        logger.warning(
            "audio_gen provider '%s' rejected kwargs (signature too narrow): %s",
            getattr(provider, "name", "?"), exc,
        )
        return json.dumps(error_response(
            error=(
                f"Provider '{getattr(provider, 'name', '?')}' signature is "
                f"out of date with the audio_generate schema. Report this "
                f"to the plugin author."
            ),
            error_type="provider_contract",
            provider=getattr(provider, "name", ""),
            model=model or "",
            prompt=prompt,
        ))
    except Exception as exc:
        logger.warning(
            "audio_gen provider '%s' raised: %s",
            getattr(provider, "name", "?"), exc,
        )
        return json.dumps(error_response(
            error=f"Provider '{getattr(provider, 'name', '?')}' error: {exc}",
            error_type="provider_exception",
            provider=getattr(provider, "name", ""),
            model=model or "",
            prompt=prompt,
        ))

    if not isinstance(result, dict):
        return json.dumps(error_response(
            error="Provider returned a non-dict result",
            error_type="provider_contract",
            provider=getattr(provider, "name", ""),
            model=model or "",
            prompt=prompt,
        ))

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Dynamic schema — reflect the active backend's actual capabilities
# ---------------------------------------------------------------------------
#
# Why dynamic: the user's configured backend determines which kinds
# (music / sfx), formats, duration ranges, and lyric/negative-prompt flags
# are real. Surfacing the per-model surface in the description means the
# model usually gets the call right on the first try.
#
# Memoization: model_tools.get_tool_definitions() keys its cache on
# config.yaml mtime, so when the user changes provider/model via
# `hermes tools`, the schema rebuilds automatically.


_GENERIC_DESCRIPTION = (
    "Generate audio — music, a soundscape, or a sound effect — from a text "
    "prompt using the user's configured audio generation backend. This is "
    "NOT text-to-speech: use `text_to_speech` to read a fixed string aloud, "
    "and use this tool to compose new audio from a creative prompt. The "
    "backend and model family are user-configured via `hermes tools` → "
    "Audio Generation; the agent does not pick them. Generation may take "
    "several seconds — the call blocks until the audio is ready. Returns "
    "either an HTTP URL or an absolute file path in the `audio` field; "
    "display it with markdown and the gateway will deliver it."
)


def _build_dynamic_audio_schema() -> Dict[str, Any]:
    """Build a description that reflects the active backend's actual surface.

    Cheap: reads config (already memoized by the caller), asks the active
    provider for ``capabilities()``, and formats a few lines of prose.
    Falls back to the generic description when no provider is configured
    or registered.
    """
    parts = [_GENERIC_DESCRIPTION]

    configured = _read_configured_audio_provider()
    configured_model = _read_configured_audio_model()

    if not configured:
        parts.append(
            "\nNo audio backend is configured. Calls will return an error "
            "until the user picks one via `hermes tools` → Audio Generation."
        )
        return {"description": "\n".join(parts)}

    try:
        from agent.audio_gen_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider(configured)
    except Exception:
        provider = None

    if provider is None:
        parts.append(
            f"\nActive backend: {configured} (plugin not yet loaded — the "
            f"tool will retry discovery on first call)."
        )
        return {"description": "\n".join(parts)}

    try:
        caps = provider.capabilities() or {}
    except Exception:
        caps = {}

    active_model = configured_model or provider.default_model()
    line = f"\nActive backend: {provider.display_name}"
    if active_model:
        line += f" · model: {active_model}"
    parts.append(line)

    if caps.get("kinds"):
        parts.append(f"- kinds: {', '.join(caps['kinds'])}")
    if caps.get("formats"):
        parts.append(f"- formats: {', '.join(caps['formats'])}")
    if caps.get("min_duration") and caps.get("max_duration"):
        parts.append(
            f"- duration range: {caps['min_duration']}-{caps['max_duration']}s"
        )
    if caps.get("supports_lyrics"):
        parts.append("- lyrics: pass `lyrics` for vocal/song output")
    if caps.get("supports_negative_prompt"):
        parts.append("- negative_prompt: supported")

    return {"description": "\n".join(parts)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


registry.register(
    name="audio_generate",
    toolset="audio_gen",
    schema=AUDIO_GENERATE_SCHEMA,
    handler=_handle_audio_generate,
    check_fn=check_audio_generation_requirements,
    requires_env=[],
    is_async=False,
    emoji="🎵",
    dynamic_schema_overrides=_build_dynamic_audio_schema,
)
