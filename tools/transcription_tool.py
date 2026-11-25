"""``transcribe_audio`` agent tool — speech-to-text on an audio file.

The same STT provider stack that transcribes gateway voice messages
(:func:`tools.transcription_tools.transcribe_audio`) is exposed to the model as a tool, so a
recording dropped into the chat as an ``@file:`` reference can be turned into text on demand
instead of dead-ending as a "binary file" stub. Availability is gated by the existing
``stt.enabled`` switch (plus a resolvable provider) — the tool never appears when STT is off.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# A long recording can transcribe to a lot of text; bound the history a single call can add.
MAX_RESULT_CHARS = 100_000


def _check_transcription_requirements() -> bool:
    """True when ``stt.enabled`` is on AND a provider (built-in / command / plugin) resolves.

    Mirrors the voice-mode availability probe: a disabled switch or an unconfigured provider
    keeps the schema out of the model's tool list rather than advertising a call that can only
    fail. Cached ~30 s by the registry's ``check_fn`` TTL, so flipping ``stt.enabled`` lands
    within a turn or two.
    """
    try:
        from tools.transcription_tools import (
            BUILTIN_STT_PROVIDERS,
            _get_provider,
            _load_stt_config,
            _resolve_command_stt_provider_config,
            is_stt_enabled,
        )
    except Exception:
        return False
    try:
        stt_config = _load_stt_config()
        if not is_stt_enabled(stt_config):
            return False
        provider = _get_provider(stt_config)
        if provider in BUILTIN_STT_PROVIDERS:
            return True
        if _resolve_command_stt_provider_config(provider, stt_config) is not None:
            return True
        # Plugin providers are resolved in voice_mode, which pulls heavy deps (numpy, audio
        # backends): import it only once the built-in and command providers have both missed,
        # so a missing/heavy module cannot take the whole probe down with it.
        from tools.voice_mode import _check_plugin_stt_provider
        return bool(_check_plugin_stt_provider(provider))
    except Exception:
        logger.debug("transcribe_audio availability probe failed", exc_info=True)
        return False


def transcribe_audio_tool(path: str, model: str | None = None) -> str:
    """Transcribe one audio file with the configured STT provider and return the envelope as JSON."""
    from tools.registry import tool_error
    from tools.transcription_tools import transcribe_audio

    if not path or not str(path).strip():
        return tool_error("Missing required path.")
    try:
        result = transcribe_audio(str(path).strip(), model=model, source="tool")
    except Exception as exc:  # never raise into the tool loop
        logger.error("transcribe_audio tool failed for %s: %s", path, exc, exc_info=True)
        return tool_error(f"Transcription failed: {exc}")
    return json.dumps(result, ensure_ascii=False)


TRANSCRIBE_AUDIO_SCHEMA = {
    "name": "transcribe_audio",
    "description": (
        "Transcribe a speech/audio file to text with the configured STT provider "
        "(local faster-whisper by default; cloud providers when configured). Use this on audio "
        "attachments and voice notes — .m4a, .mp3, .wav, .ogg, .opus, .flac, .aac, .wma, .silk, "
        ".caf and similar. Call it with the file's path (the path shown in the attachment's "
        "📎 context block works). Returns a JSON envelope: success=true carries the transcript "
        "in `transcript`; success=false carries an `error` and a configured `provider`. Do not "
        "call read_file on an audio file — it cannot decode audio."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the audio file to transcribe.",
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override (e.g. a whisper model name for the local provider). "
                    "Omit to use the configured default."
                ),
            },
        },
        "required": ["path"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="transcribe_audio",
    toolset="file",
    schema=TRANSCRIBE_AUDIO_SCHEMA,
    handler=lambda args, **kw: transcribe_audio_tool(
        args.get("path"), model=args.get("model")),
    check_fn=_check_transcription_requirements,
    emoji="🎙️",
    max_result_size_chars=MAX_RESULT_CHARS,
)
