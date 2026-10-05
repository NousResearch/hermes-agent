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
_WHITESPACE_RE = re.compile(r"\s+")


def _sanitize_tts_instructions(value: Any) -> str:
    if value is None:
        return ""
    clean = _WHITESPACE_RE.sub(" ", str(value)).strip()
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


def _elevenlabs_text_with_instructions(text: str, instructions: str, model_id: str) -> str:
    if not instructions or not _elevenlabs_supports_instruction_tags(model_id):
        return text
    audio_tag = re.sub(r"\s+", " ", re.sub(r"[<>\[\]{}]", " ", instructions)).strip()
    return f"[{audio_tag}] {text}" if audio_tag else text


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


def _tts_text_chunk_limit(
    provider: Optional[str], instructions: str, tts_config: Optional[Dict[str, Any]],
    max_length: int,
) -> int:
    key = (provider or "").lower().strip()
    config = tts_config if isinstance(tts_config, dict) else {}
    if key == "gemini":
        from tools.tts_tool_providers import _compose_gemini_tts_prompt, _read_gemini_persona_prompt
        section = _get_provider_section(config, "gemini")
        persona = _read_gemini_persona_prompt(section)
        if persona or instructions:
            lower, upper = 0, max_length
            while lower < upper:
                candidate = (lower + upper + 1) // 2
                prompt = _compose_gemini_tts_prompt(
                    "x" * candidate, section, persona_prompt=persona,
                    instructions=instructions,
                )
                if len(prompt) <= max_length:
                    lower = candidate
                else:
                    upper = candidate - 1
            return max(1, lower)
    return max(1, max_length - _tts_instructions_overhead(provider, instructions, config))


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
        return bool(_xai_instructions_wrap_tag(instructions))
    if key == "elevenlabs":
        from tools.tts_tool_providers import DEFAULT_ELEVENLABS_MODEL_ID
        section = _get_provider_section(config, "elevenlabs")
        return _elevenlabs_supports_instruction_tags(str(section.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID)))
    if key == "minimax":
        return instructions.lower() in _MINIMAX_TTS_EMOTIONS
    return False
