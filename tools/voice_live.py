"""GPT-Live voice chat mode: the full-duplex voice frontend that delegates to Hermes.

``voice.voice_chat_mode: gpt-live`` replaces the chained STT → turn → TTS loop with ONE
full-duplex voice model (OpenAI ``gpt-live-1``) that owns the microphone and the speaker and
delegates every real request to Hermes as its *client-delegation* backend. Hermes stays the
agent: whatever model/provider the session has selected answers, with the full toolset.

Division of labour (the Live API has no tools of its own in client mode):

* the desktop renderer holds the WebRTC media session (mic in, speech out) and the data channel;
* this module resolves WHICH credentials/voice/persona to use and performs the one server-side
  step the API requires — exchanging the browser's SDP offer for an answer with the project key
  (``POST /v1/live/sessions``), so the key never reaches the client;
* the renderer turns each ``session.delegation.created`` into a normal ``prompt.submit`` on the
  active session (surface ``voice-live``) and streams the reply back as
  ``session.commentary.append`` — Hermes' answer is what the voice speaks.

Vendor contract: https://developers.openai.com/api/docs/guides/live (+ live-delegation,
voice-webrtc). Billing is $0.05/min of session time on the OpenAI key, separate from the
Hermes turn.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GPT_LIVE_MODE = "gpt-live"
CHAINED_MODE = "chained"
DEFAULT_LIVE_MODEL = "gpt-live-1"
DEFAULT_LIVE_VOICE = "marin"
DEFAULT_LIVE_BASE_URL = "https://api.openai.com/v1"
# Voices the vendor lists for gpt-live-1 (live-conversations guide) plus the realtime defaults it
# accepts; free text stays allowed for custom voices.
GPT_LIVE_VOICES = (
    "marin", "cedar", "quartz", "ripple", "vesper", "willow", "stone", "gleam", "meridian",
    "bossa", "tempo", "beacon", "delta", "cinder",
)

# Persona for the voice layer. Short on purpose: the live model has a small context window and
# the vendor guide asks for role + style + a labelled delegation policy, nothing more. The
# backend (Hermes) carries the real instructions, tools and memory.
LIVE_PERSONA = (
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

# Per-turn note prepended to the MODEL INPUT (never the byte-stable system prompt) when a turn
# arrives from the live voice layer. Same seam as the HUD note.
VOICE_LIVE_TURN_NOTE = (
    "[Note: this message is a delegation from a live spoken conversation. The text is a voice "
    "transcript (it may contain mis-hearings, hesitations and later corrections; use the latest "
    "intent). Your reply will be spoken aloud by a voice model that paraphrases it: answer in plain "
    "conversational sentences, keep it short (a few sentences unless the user asked for detail), no "
    "markdown, no lists, no code blocks, no URLs read out character by character. Do the work with "
    "your tools as usual; only the final facts need to be spoken. Do not claim an action succeeded "
    "before it actually did.]"
)


def voice_live_turn_note(context: str = "") -> str:
    """The per-turn note plus, when the client sent one, the recent spoken exchange the delegation
    refers to (the user's last words alone are often "yes" or "Thursday, not Friday")."""
    context = context.strip()
    if not context:
        return VOICE_LIVE_TURN_NOTE
    return f"{VOICE_LIVE_TURN_NOTE}\n[Recent spoken conversation, newest last:\n{context}]"


def _voice_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        voice = load_config().get("voice")
    except Exception:
        return {}
    return voice if isinstance(voice, dict) else {}


def _live_section(voice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    section = (voice if voice is not None else _voice_section()).get("gpt_live")
    return section if isinstance(section, dict) else {}


def voice_chat_mode(voice: Optional[Dict[str, Any]] = None) -> str:
    """``chained`` (default) or ``gpt-live``. Accepts the underscore spelling too."""
    raw = (voice if voice is not None else _voice_section()).get("voice_chat_mode")
    mode = str(raw or CHAINED_MODE).strip().lower().replace("_", "-")
    return GPT_LIVE_MODE if mode in {GPT_LIVE_MODE, "gptlive", "live"} else CHAINED_MODE


def _resolve_credentials(live: Dict[str, Any]) -> tuple[str, str]:
    """``(api_key, base_url)`` — ``voice.gpt_live.api_key`` first, else the same OpenAI audio
    chain the STT/TTS providers use (``VOICE_TOOLS_OPENAI_KEY`` → ``OPENAI_API_KEY`` → pool).

    The Nous-managed audio proxy does not carry ``/live/sessions``; this mode is direct-key only.
    """
    from tools.tool_backend_helpers import resolve_openai_audio_api_key
    api_key = str(live.get("api_key") or "").strip() or resolve_openai_audio_api_key()
    base_url = str(live.get("base_url") or DEFAULT_LIVE_BASE_URL).strip().rstrip("/")
    return api_key, base_url


def live_instructions(live: Optional[Dict[str, Any]] = None) -> str:
    extra = str((live if live is not None else _live_section()).get("instructions") or "").strip()
    return f"{LIVE_PERSONA}\n\n{extra}" if extra else LIVE_PERSONA


def resolve_gpt_live_status() -> Dict[str, Any]:
    """Non-secret readiness verdict for the client: which mode is selected and whether GPT-Live
    can start (a key resolves). Never returns the key."""
    voice = _voice_section()
    mode = voice_chat_mode(voice)
    live = _live_section(voice)
    api_key, _base = _resolve_credentials(live)
    return {
        "mode": mode,
        "available": bool(api_key),
        "reason": None if api_key else "no OpenAI API key (set OPENAI_API_KEY or voice.gpt_live.api_key)",
        "model": str(live.get("model") or DEFAULT_LIVE_MODEL),
        "voice": str(live.get("voice") or DEFAULT_LIVE_VOICE),
    }


def build_session_config(history: Optional[list] = None) -> Dict[str, Any]:
    """The ``session`` object for ``POST /v1/live/sessions`` (client delegation, WebRTC — the
    transport negotiates the audio format, so none is set)."""
    live = _live_section()
    config: Dict[str, Any] = {
        "model": str(live.get("model") or DEFAULT_LIVE_MODEL),
        "instructions": live_instructions(live),
        "audio": {"output": {"voice": str(live.get("voice") or DEFAULT_LIVE_VOICE)}},
        "delegation": {"type": "client"},
    }
    if history:
        config["input"] = history
    return config


def create_webrtc_session(sdp_offer: str, history: Optional[list] = None) -> Dict[str, Any]:
    """Exchange the renderer's SDP offer for the Live session answer.

    Returns the vendor response ``{"session": {"id": ...}, "transport": {"type": "webrtc",
    "sdp": ...}}``. Raises ``ValueError`` for a missing key and ``RuntimeError`` (with the vendor
    status/detail) for a rejected request.
    """
    live = _live_section()
    api_key, base_url = _resolve_credentials(live)
    if not api_key:
        raise ValueError("GPT-Live needs an OpenAI API key (OPENAI_API_KEY or voice.gpt_live.api_key)")
    body = json.dumps({
        "session": build_session_config(history),
        "transport": {"type": "webrtc", "sdp": sdp_offer},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/live/sessions", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        logger.warning("GPT-Live session creation failed: %s %s", exc.code, detail)
        raise RuntimeError(f"GPT-Live session creation failed ({exc.code}): {detail}") from exc
