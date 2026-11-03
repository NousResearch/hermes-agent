"""Configuration, credentials, and readiness for the xAI Grok Live voice engine.

``voice.voice_chat_mode: grok-live`` gives xAI's realtime model the microphone
and speaker while Hermes remains the agent that answers delegated requests. The
gateway holds the xAI websocket; this module owns only configuration, auth, and
the non-secret status surface. The relay itself lives in
``tools.voice_live_grok_bridge``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.voice_live import GROK_LIVE_MODE, _voice_section, voice_chat_mode

logger = logging.getLogger(__name__)

DEFAULT_GROK_LIVE_MODEL = "grok-voice-latest"
DEFAULT_GROK_LIVE_VOICE = "eve"

NO_CREDENTIAL_REASON = "no xAI credential (SuperGrok login via `hermes login` or set XAI_API_KEY)"

# Short voice-layer persona; Hermes supplies the real system prompt, tools, and
# memory only after delegation. This text is sent as xAI session instructions.
GROK_LIVE_PERSONA = (
    "You are Hermes, a calm and friendly voice assistant. Speak naturally at an unhurried pace. "
    "Be clear and direct, not overly cheerful. If the user is frustrated, acknowledge it briefly "
    "and focus on the next helpful step.\n\n"
    "Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with "
    "the main response.\n\n"
    "Interruption policy: Stop speaking when the user interrupts. Listen to what they say.\n\n"
    "Delegation policy:\n"
    "Backend tools:\n"
    "- Hermes agent: a full AI agent with tools — it can run commands, read and edit files, "
    "browse the web, search, remember things across sessions, schedule tasks, and reason "
    "carefully about anything. It is the one who actually does work and knows facts.\n\n"
    "Delegate to the backend when:\n"
    "- The user asks a question that needs facts, current information, or careful reasoning.\n"
    "- The user asks you to do, check, find, make, fix, run or remember anything.\n"
    "- A correction changes work already requested.\n\n"
    "Do not delegate to the backend when:\n"
    "- The user greets you, makes small talk, or asks you to repeat a result already provided.\n"
    "- You need a brief clarification to understand the request.\n\n"
    "Delegate before giving an answer that depends on backend work. Do not guess the result "
    "while waiting; say briefly that you are checking, then wait for the result."
)


def _grok_live_section(voice: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    section = (voice if voice is not None else _voice_section()).get("grok_live")
    return section if isinstance(section, dict) else {}


def _resolve_grok_credentials(live: Optional[dict[str, Any]] = None) -> str:
    """Resolve a realtime bearer through Hermes's shared xAI credential path.

    ``auto`` prefers the managed OAuth pool, then falls back to XAI_API_KEY;
    ``oauth`` and ``apikey`` restrict the result to that credential kind. The
    shared resolver owns profile scoping, refresh locking, atomic persistence,
    multi-account selection, and origin validation.
    """
    from tools.xai_http import resolve_xai_http_credentials

    live = live if live is not None else _grok_live_section()
    auth = str(live.get("auth") or "auto").strip().lower()
    credentials = resolve_xai_http_credentials(prefer_api_key=auth == "apikey")
    provider = credentials.get("provider")
    if auth == "oauth" and provider != "xai-oauth":
        return ""
    if auth == "apikey" and provider != "xai":
        return ""
    return str(credentials.get("api_key") or "").strip()


def live_instructions(live: Optional[dict[str, Any]] = None) -> str:
    """Return the voice-layer persona plus configured extra instructions."""
    extra = str((live if live is not None else _grok_live_section()).get("instructions") or "").strip()
    return f"{GROK_LIVE_PERSONA}\n\n{extra}" if extra else GROK_LIVE_PERSONA


def resolve_grok_live_status() -> dict[str, Any]:
    """Return the selected mode and whether a credential resolves, without secrets."""
    voice = _voice_section()
    live = _grok_live_section(voice)
    credential = _resolve_grok_credentials(live)
    return {
        "mode": voice_chat_mode(voice),
        "available": bool(credential),
        "reason": None if credential else NO_CREDENTIAL_REASON,
        "model": str(live.get("model") or DEFAULT_GROK_LIVE_MODEL),
        "voice": str(live.get("voice") or DEFAULT_GROK_LIVE_VOICE),
    }
