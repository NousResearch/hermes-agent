#!/usr/bin/env python3
"""Photon voice-note tool — send a recorded audio clip over iMessage.

The Photon Spectrum (iMessage) adapter already exposes ``send_voice()`` and the
sidecar already converts source audio to a native voice note (see
``plugins/platforms/photon/sidecar/voice-send.mjs``). There was, however, no
agent-callable surface for it: ``send_message`` is intentionally NOT
agent-callable (cross-platform blast protection), so a voice note had no tool.

This module fills that gap with a *service-gated* tool: it only registers its
schema when a live Photon adapter is in the running gateway (zero prompt tokens
on every other install), and routes through the live adapter's ``send_voice``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error


# ---------------------------------------------------------------------------
# Live-adapter resolution (mirrors tools/send_message_tool.py)
# ---------------------------------------------------------------------------


def _live_photon_adapter():
    """Return the in-process Photon adapter, or ``None``.

    Photon is a plugin platform, so the enum member is resolved dynamically
    via ``Platform("photon")`` rather than a hard-coded ``Platform.PHOTON``.
    """
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            return None
        adapter = runner.adapters.get(Platform("photon"))
        return adapter
    except Exception:
        return None


def _photon_voice_check() -> bool:
    """check_fn: the tool exists only when a live Photon adapter is in process."""
    return _live_photon_adapter() is not None


def _resolve_chat_id(explicit: Optional[str]) -> Optional[str]:
    """Chat resolution order: explicit arg -> session chat -> photon home."""
    if explicit:
        return explicit.strip() or None
    # Conversation being answered (cron/standalone callers may set this).
    try:
        from gateway.session_context import get_session_env

        env_chat = get_session_env("HERMES_SESSION_CHAT_ID", "")
        if env_chat:
            return env_chat
    except Exception:
        pass
    # Photon home channel from gateway config.
    try:
        from gateway.config import Platform, load_gateway_config

        home = load_gateway_config().get_home_channel(Platform("photon"))
        if home and getattr(home, "chat_id", None):
            return home.chat_id
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


PHOTON_VOICE_SCHEMA: Dict[str, Any] = {
    "name": "photon_voice",
    "description": (
        "Send a voice note (audio clip) to a Photon Spectrum (iMessage) chat. "
        "USE SPARSINGLY — voice notes are high-friction for the recipient, so "
        "only send one when the user explicitly asked for a spoken message or "
        "when tone/inflection genuinely matters. Requires a pre-recorded audio "
        "file (e.g. produced with the tts tool or `edge-tts`); pass its local "
        "path. The sidecar converts the file to a native iMessage voice note. "
        "Only available on installs with a live Photon gateway adapter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": (
                    "Absolute path to the audio file to send (mp3/wav/m4a/etc.). "
                    "The sidecar re-encodes to a native voice-note format, so the "
                    "source codec is not strict."
                ),
            },
            "chat_id": {
                "type": "string",
                "description": (
                    "Optional target chat id (Photon spaceId, e.g. "
                    "'any;-;+15551234567'). Defaults to the Photon home channel "
                    "when omitted."
                ),
            },
            "caption": {
                "type": "string",
                "description": "Optional text caption delivered after the voice note.",
            },
        },
        "required": ["audio_path"],
    },
}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_photon_voice(args: Dict[str, Any], **_kw: Any) -> str:
    audio_path = args.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path.strip():
        return tool_error("audio_path is required (absolute path to an audio file)")
    audio_path = audio_path.strip()

    # Defense-in-depth: the sidecar re-validates, but refuse obviously missing
    # paths early so we don't spin up the gateway call for nothing.
    if not os.path.isfile(audio_path):
        return tool_error(f"audio_path does not exist: {audio_path}")

    chat_id = _resolve_chat_id(args.get("chat_id"))
    if not chat_id:
        return tool_error(
            "No chat_id supplied and no Photon home channel configured. "
            "Pass an explicit chat_id (Photon spaceId)."
        )

    adapter = _live_photon_adapter()
    if adapter is None:
        return tool_error(
            "photon_voice requires a live Photon adapter in the running gateway "
            "(not available from cron/standalone contexts)."
        )
    send_voice = getattr(adapter, "send_voice", None)
    if not callable(send_voice):
        return tool_error("The Photon adapter does not support send_voice.")

    caption = args.get("caption")
    caption = caption.strip() if isinstance(caption, str) and caption.strip() else None

    from model_tools import _run_async

    try:
        result = _run_async(send_voice(chat_id=chat_id, audio_path=audio_path, caption=caption))
    except Exception as e:  # surface transport failures to the model, not a crash
        return json.dumps({
            "success": False,
            "error": f"photon_voice send failed: {e}",
            "error_type": "send_failed",
        })

    # adapter.send_voice returns a gateway SendResult dataclass — serialize the
    # public attributes rather than leaking the object.
    success = bool(getattr(result, "success", False))
    payload: Dict[str, Any] = {
        "success": success,
        "message_id": getattr(result, "message_id", None),
    }
    if not success:
        payload["error"] = getattr(result, "error", "unknown send failure")
        payload["retryable"] = getattr(result, "retryable", False)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Registration — own toolset so non-Photon installs pay zero prompt tokens.
# ---------------------------------------------------------------------------


# NOTE: the handler is synchronous but internally bridges the adapter's async
# send_voice via model_tools._run_async. Register is_async=False so dispatch
# does NOT double-wrap it (registry.dispatch only calls _run_async for async
# handlers). This mirrors tools/send_message_tool.py's react/send handlers.
registry.register(
    name="photon_voice",
    toolset="photon_voice",
    schema=PHOTON_VOICE_SCHEMA,
    handler=_handle_photon_voice,
    check_fn=_photon_voice_check,
    is_async=False,
    emoji="🎙️",
)
