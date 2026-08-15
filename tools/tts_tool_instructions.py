"""Resolve and describe style instructions for text-to-speech providers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS, _get_named_provider_config, _get_provider_section,
)

TTS_INSTRUCTIONS_MAX_CHARS = 200
_MINIMAX_TTS_EMOTIONS = frozenset({
    "happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm", "neutral",
})
_COMMAND_TTS_INSTRUCTIONS_PLACEHOLDER_RE = re.compile(r"(?<!\$)\{\{?instructions\}\}?")
_BRACKETS_RE = re.compile(r"[<>\[\]{}]")
_WHITESPACE_RE = re.compile(r"\s+")


def _sanitize_tts_instructions(value: Any) -> str:
    if value is None:
        return ""
    clean = _WHITESPACE_RE.sub(" ", _BRACKETS_RE.sub(" ", str(value))).strip()
    return clean[:TTS_INSTRUCTIONS_MAX_CHARS].strip()


def _resolve_tts_instructions(
    provider: Optional[str], tts_config: Optional[Dict[str, Any]] = None,
    instructions_override: Optional[str] = None,
) -> str:
    if instructions_override is not None:
        return _sanitize_tts_instructions(instructions_override)
    key = (provider or "").lower().strip()
    config = tts_config if isinstance(tts_config, dict) else {}
    section = _get_provider_section(config, key)
    if not section and key and key not in BUILTIN_TTS_PROVIDERS:
        section = _get_named_provider_config(config, key)
    value = section.get("instructions") if isinstance(section, dict) else None
    return _sanitize_tts_instructions(config.get("instructions") if value is None else value)


def _tts_instructions_channel(tts_config: Optional[Dict[str, Any]]) -> str:
    if not isinstance(tts_config, dict):
        return ""
    value = tts_config.get("instructions")
    return value.strip() if isinstance(value, str) else ""


def _xai_instructions_wrap_tag(instructions: str) -> str:
    from tools.tts_tool_providers import _XAI_WRAPPING_SPEECH_TAGS
    candidate = instructions.lower().strip()
    return candidate if candidate in _XAI_WRAPPING_SPEECH_TAGS else ""


def _elevenlabs_supports_instruction_tags(model_id: str) -> bool:
    return "v3" in (model_id or "").strip().lower()


def _tts_instructions_overhead(
    provider: Optional[str], instructions: str, tts_config: Optional[Dict[str, Any]] = None,
) -> int:
    if not instructions:
        return 0
    key = (provider or "").lower().strip()
    if key == "xai":
        tag = _xai_instructions_wrap_tag(instructions)
        return 2 * len(tag) + 5 if tag else 0
    if key == "elevenlabs":
        from tools.tts_tool_providers import DEFAULT_ELEVENLABS_MODEL_ID
        section = _get_provider_section(tts_config or {}, "elevenlabs")
        model_id = str(section.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID))
        if _elevenlabs_supports_instruction_tags(model_id):
            return len(instructions) + 3
    return 0


def _tts_instructions_applied(
    provider: Optional[str], instructions: str, tts_config: Optional[Dict[str, Any]],
    command_provider_config: Optional[Dict[str, Any]] = None,
) -> bool:
    if not instructions:
        return False
    key = (provider or "").lower().strip()
    config = tts_config if isinstance(tts_config, dict) else {}
    if command_provider_config is not None:
        template = str(command_provider_config.get("command") or "")
        return bool(_COMMAND_TTS_INSTRUCTIONS_PLACEHOLDER_RE.search(template))
    if key in {"openai", "deepinfra", "gemini"}:
        return True
    if key == "xai":
        from tools.tts_tool_providers import DEFAULT_XAI_AUTO_SPEECH_TAGS, _config_bool
        section = _get_provider_section(config, "xai")
        auto_tags = _config_bool(
            section.get("auto_speech_tags", section.get("speech_tags")), DEFAULT_XAI_AUTO_SPEECH_TAGS,
        )
        return bool(_xai_instructions_wrap_tag(instructions)) or auto_tags
    if key == "elevenlabs":
        from tools.tts_tool_providers import DEFAULT_ELEVENLABS_MODEL_ID
        section = _get_provider_section(config, "elevenlabs")
        return _elevenlabs_supports_instruction_tags(str(section.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID)))
    if key == "minimax":
        return instructions.lower() in _MINIMAX_TTS_EMOTIONS
    if key and key not in BUILTIN_TTS_PROVIDERS:
        from agent.tts_registry import get_provider
        return get_provider(key) is not None
    return False
