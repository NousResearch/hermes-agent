#!/usr/bin/env python3
"""Multi-modal config tools.

Two agent-facing tools let the LLM read and update the per-modality
provider/model selections that drive image_gen, video_gen, tts, stt,
web search, and audio generation. Useful when the user says things
like "from now on use Veo 3.1 for video" or "what model are we using
for images?" — the agent can act on it without the user opening a
settings page.

Both tools manipulate ``$HERMES_HOME/config.yaml`` via the existing
:func:`hermes_cli.config.load_config` / :func:`save_config` helpers,
so the webui side (which reads the same file) sees the new values on
its next render. Bi-directional sync is implicit through the shared
file rather than a separate IPC channel.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


# Modality → (config section, supported keys, description)
# Each modality maps to a top-level section in config.yaml; the `provider`
# and `model` keys are settable; other tool-specific keys (voice, etc.)
# can be set via the optional `extra` argument.
_MODALITY_MAP: Dict[str, Dict[str, Any]] = {
    "image": {
        "section": "image_gen",
        "description": "Image generation (image_generate, image_upscale, image_edit, image_compose, image_remove_background)",
    },
    "video": {
        "section": "video_gen",
        "description": "Video generation (video_generate text-to-video + image-to-video) and audio_generate (music/SFX)",
    },
    "tts": {
        "section": "tts",
        "description": "Text-to-speech (the agent speaking back)",
    },
    "stt": {
        "section": "stt",
        "description": "Speech-to-text (transcribing uploaded audio)",
    },
    "web_search": {
        "section": "web",
        "description": "Web search + page extract (web_search, web_extract)",
    },
}


_VALID_MODALITIES = sorted(_MODALITY_MAP.keys())


def _load_config_safe() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.debug("Could not load config.yaml: %s", exc)
        return {}


def _save_config_safe(cfg: Dict[str, Any]) -> Optional[str]:
    """Persist *cfg* to config.yaml. Returns error string on failure."""
    try:
        from hermes_cli.config import save_config

        save_config(cfg)
        return None
    except Exception as exc:
        logger.warning("Could not save config.yaml: %s", exc)
        return str(exc)


def _list_image_gen_providers() -> List[Dict[str, Any]]:
    try:
        from agent.image_gen_registry import list_providers

        return [
            {"name": p.name, "display_name": getattr(p, "display_name", p.name), "is_available": bool(p.is_available())}
            for p in list_providers()
        ]
    except Exception:
        return []


def _list_video_gen_providers() -> List[Dict[str, Any]]:
    try:
        from agent.video_gen_registry import list_providers

        return [
            {"name": p.name, "display_name": getattr(p, "display_name", p.name), "is_available": bool(p.is_available())}
            for p in list_providers()
        ]
    except Exception:
        return []


def _list_web_providers() -> List[Dict[str, Any]]:
    try:
        from agent.web_search_registry import list_providers

        return [
            {"name": p.name, "display_name": getattr(p, "display_name", p.name), "is_available": bool(p.is_available())}
            for p in list_providers()
        ]
    except Exception:
        return []


def _tts_provider_options() -> List[Dict[str, Any]]:
    # tts_tool.py uses a frozenset rather than a plugin registry; reflect
    # availability by env var presence to keep the picker honest.
    candidates = [
        ("edge", "Edge TTS (free, no key)", True),
        ("elevenlabs", "ElevenLabs", bool(os.environ.get("ELEVENLABS_API_KEY"))),
        ("openai", "OpenAI", bool(os.environ.get("OPENAI_API_KEY"))),
        ("minimax", "MiniMax", bool(os.environ.get("MINIMAX_API_KEY"))),
        ("xai", "xAI", bool(os.environ.get("XAI_API_KEY"))),
        ("venice", "Venice", bool(os.environ.get("VENICE_API_KEY"))),
        ("gemini", "Google Gemini", bool(os.environ.get("GEMINI_API_KEY"))),
        ("piper", "Piper (local)", True),
        ("kittentts", "KittenTTS (local)", True),
        ("neutts", "NeuTTS (local)", True),
    ]
    return [
        {"name": name, "display_name": display, "is_available": available}
        for name, display, available in candidates
    ]


def _stt_provider_options() -> List[Dict[str, Any]]:
    candidates = [
        ("local", "Local (faster-whisper)", True),
        ("groq", "Groq Whisper (free tier)", bool(os.environ.get("GROQ_API_KEY"))),
        ("openai", "OpenAI Whisper", bool(os.environ.get("OPENAI_API_KEY"))),
        ("venice", "Venice", bool(os.environ.get("VENICE_API_KEY"))),
        ("xai", "xAI Grok STT", bool(os.environ.get("XAI_API_KEY"))),
    ]
    return [
        {"name": name, "display_name": display, "is_available": available}
        for name, display, available in candidates
    ]


def _modality_providers(modality: str) -> List[Dict[str, Any]]:
    if modality == "image":
        return _list_image_gen_providers()
    if modality == "video":
        return _list_video_gen_providers()
    if modality == "web_search":
        return _list_web_providers()
    if modality == "tts":
        return _tts_provider_options()
    if modality == "stt":
        return _stt_provider_options()
    return []


def multimodal_get_settings_tool() -> str:
    """Return the agent's current multi-modal provider/model selections.

    Reads config.yaml + reports which providers are currently
    available based on registry membership and env-var presence.
    """
    cfg = _load_config_safe()
    report: Dict[str, Any] = {"success": True, "modalities": {}}

    for modality, meta in _MODALITY_MAP.items():
        section_name = meta["section"]
        section = cfg.get(section_name) if isinstance(cfg.get(section_name), dict) else {}
        modality_block: Dict[str, Any] = {
            "description": meta["description"],
            "section": section_name,
            "active_provider": section.get("provider"),
            "active_model": section.get("model"),
            "available_providers": _modality_providers(modality),
        }
        # For modalities with sub-keys (e.g. tts.voice, video_gen.duration),
        # surface a few extras so the LLM doesn't lose context.
        if modality == "tts":
            modality_block["voice"] = section.get("voice")
        if modality == "video":
            modality_block["duration"] = section.get("duration")
            modality_block["resolution"] = section.get("resolution")
        report["modalities"][modality] = modality_block

    return json.dumps(report)


def multimodal_set_model_tool(
    modality: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Update the active provider/model for a modality, persisting to config.yaml.

    Args:
        modality: one of image | video | tts | stt | web_search.
        provider: optional new provider name (e.g. "venice").
        model:    optional new model id within that provider.
        extra:    optional dict of additional keys to set on the section
                  (e.g. {"voice": "af_sky"} for tts).

    Either provider or model (or both) should be specified. extra alone
    is allowed when the caller wants to nudge a voice/duration without
    changing the provider.
    """
    modality_key = (modality or "").strip().lower()
    if modality_key not in _MODALITY_MAP:
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown modality '{modality}'. Valid: {', '.join(_VALID_MODALITIES)}",
                "error_type": "bad_modality",
            }
        )

    if not provider and not model and not extra:
        return json.dumps(
            {
                "success": False,
                "error": "Specify at least one of: provider, model, extra",
                "error_type": "bad_input",
            }
        )

    meta = _MODALITY_MAP[modality_key]
    section_name = meta["section"]

    cfg = _load_config_safe()
    section = cfg.get(section_name) if isinstance(cfg.get(section_name), dict) else {}
    if not isinstance(section, dict):
        section = {}

    if isinstance(provider, str) and provider.strip():
        section["provider"] = provider.strip()
    if isinstance(model, str) and model.strip():
        section["model"] = model.strip()
    if isinstance(extra, dict):
        for k, v in extra.items():
            if isinstance(k, str) and k.strip():
                section[k] = v

    cfg[section_name] = section
    err = _save_config_safe(cfg)
    if err is not None:
        return json.dumps(
            {"success": False, "error": f"Could not save config.yaml: {err}", "error_type": "io_error"}
        )

    return json.dumps(
        {
            "success": True,
            "modality": modality_key,
            "section": section_name,
            "active_provider": section.get("provider"),
            "active_model": section.get("model"),
        }
    )


MULTIMODAL_GET_SETTINGS_SCHEMA = {
    "name": "multimodal_get_settings",
    "description": (
        "Report the agent's current multi-modal provider/model selections "
        "for image, video, TTS, STT, and web search. Use this when the user "
        "asks 'what model are we using for video?' or before suggesting a "
        "switch. Returns the active provider/model per modality plus the "
        "list of available alternatives (filtered by which env keys are set)."
    ),
    "parameters": {"type": "object", "properties": {}},
}


MULTIMODAL_SET_MODEL_SCHEMA = {
    "name": "multimodal_set_model",
    "description": (
        "Update the active provider and/or model for a modality. Use when "
        "the user says things like 'use Veo 3.1 for video from now on' or "
        "'switch image gen to qwen-image-2'. Writes config.yaml; the next "
        "tool call uses the new selection. The webui settings page sees "
        "the change immediately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "modality": {
                "type": "string",
                "enum": _VALID_MODALITIES,
                "description": "Which modality to update.",
            },
            "provider": {
                "type": "string",
                "description": "New provider name (e.g. 'venice', 'fal', 'edge'). Optional.",
            },
            "model": {
                "type": "string",
                "description": "New model id within the chosen provider. Optional.",
            },
            "extra": {
                "type": "object",
                "description": "Optional extra section keys (e.g. {voice: 'af_sky'} for tts).",
            },
        },
        "required": ["modality"],
    },
}


registry.register(
    name="multimodal_get_settings",
    toolset="config",
    schema=MULTIMODAL_GET_SETTINGS_SCHEMA,
    handler=lambda **kw: multimodal_get_settings_tool(),
    check_fn=lambda: True,
    requires_env=[],
    is_async=False,
    emoji="⚙️",
)


registry.register(
    name="multimodal_set_model",
    toolset="config",
    schema=MULTIMODAL_SET_MODEL_SCHEMA,
    handler=lambda **kw: multimodal_set_model_tool(**kw),
    check_fn=lambda: True,
    requires_env=[],
    is_async=False,
    emoji="🎛️",
)
