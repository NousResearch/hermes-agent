"""GPT-Live voice chat mode: the full-duplex voice frontend that delegates to Hermes.

``voice.voice_chat_mode: gpt-live`` replaces the chained STT → turn → TTS loop with ONE
full-duplex voice model (OpenAI GPT-Live) that owns the microphone and the speaker and
delegates every real request to Hermes as its *client-delegation* backend. Hermes stays the
agent: whatever model/provider the session has selected answers, with the full toolset.

Division of labour (the Live API has no tools of its own in client mode):

* the desktop renderer holds the WebRTC media session (mic in, speech out) and the data channel;
* this module resolves WHICH credentials/voice/persona to use and performs the one server-side
  step the API requires — exchanging the browser's SDP offer for an answer with the project key
  (``POST /v1/live/sessions``), or using Hermes Codex OAuth for explicit subscription mode;
  long-lived credentials and the subscription account identity never reach the client;
* the renderer turns each ``session.delegation.created`` into a normal ``prompt.submit`` on the
  active session (surface ``voice-live``) and streams the reply back as
  ``session.commentary.append`` — Hermes' answer is what the voice speaks.

Vendor contract: https://developers.openai.com/api/docs/guides/live (+ live-delegation,
voice-webrtc). API billing is separate from the Hermes turn. Subscription mode uses the
Codex frameless WebRTC contract (OpenAI Codex c4017a87aacc7558002b7cb510025e967c1d765e,
realtime_call.rs and realtime_websocket/methods_frameless_bidi.rs); it never falls back to API.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

GPT_LIVE_MODE = "gpt-live"
CHAINED_MODE = "chained"
DEFAULT_LIVE_MODEL = "gpt-live-1"
DEFAULT_LIVE_VOICE = "marin"
DEFAULT_LIVE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SUBSCRIPTION_MODEL = "gpt-live-1-codex"
DEFAULT_SUBSCRIPTION_VOICE = "cove"
CODEX_LIVE_URL = (
    "https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas"
)
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


def _live_auth(live: Dict[str, Any]) -> str:
    auth = str(live.get("auth", "api")).strip().lower()
    if auth not in {"api", "subscription"}:
        raise ValueError("voice.gpt_live.auth must be api or subscription; no fallback was used")
    return auth


def _live_model_voice(live: Dict[str, Any], auth: str) -> tuple[str, str]:
    if auth == "subscription":
        return (str(live.get("subscription_model") or DEFAULT_SUBSCRIPTION_MODEL),
                str(live.get("subscription_voice") or DEFAULT_SUBSCRIPTION_VOICE))
    return str(live.get("model") or DEFAULT_LIVE_MODEL), str(live.get("voice") or DEFAULT_LIVE_VOICE)


def _subscription_credentials(*, refresh_if_expiring: bool = True) -> tuple[str, str]:
    from hermes_cli.auth_codex import resolve_codex_runtime_credentials
    from hermes_cli.auth_constants import AuthError, _decode_jwt_claims

    try:
        credentials = resolve_codex_runtime_credentials(refresh_if_expiring=refresh_if_expiring)
    except AuthError:
        raise ValueError(
            "GPT-Live subscription needs a working Codex sign-in on the Hermes host. "
            "Run `hermes auth` and choose OpenAI Codex. No API fallback was used."
        ) from None
    token = str(credentials.get("api_key") or "").strip()
    claims = _decode_jwt_claims(token).get("https://api.openai.com/auth", {})
    account = claims.get("chatgpt_account_id") if isinstance(claims, dict) else None
    if (not token or not isinstance(account, str) or not account.strip()
            or any(c in token + account for c in "\r\n")):
        raise ValueError("GPT-Live subscription needs Codex OAuth with an account identity; no API fallback was used")
    return token, account.strip()


def live_instructions(live: Optional[Dict[str, Any]] = None) -> str:
    extra = str((live if live is not None else _live_section()).get("instructions") or "").strip()
    return f"{LIVE_PERSONA}\n\n{extra}" if extra else LIVE_PERSONA


def resolve_gpt_live_status() -> Dict[str, Any]:
    """Credential readiness only; subscription entitlement is checked when the call starts."""
    voice = _voice_section()
    live = _live_section(voice)
    auth = str(live.get("auth", "api")).strip().lower()
    reason = None
    try:
        auth = _live_auth(live)
        if auth == "subscription":
            _subscription_credentials(refresh_if_expiring=False)
        elif not _resolve_credentials(live)[0]:
            reason = "no OpenAI API key (set OPENAI_API_KEY or voice.gpt_live.api_key)"
    except ValueError as exc:
        reason = str(exc)
    model, selected_voice = _live_model_voice(live, auth)
    return {
        "mode": voice_chat_mode(voice), "auth": auth,
        "available": reason is None, "reason": reason,
        "model": model, "voice": selected_voice,
    }


def build_session_config(history: Optional[list] = None, *, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Client delegation and history for the explicitly selected Live protocol."""
    live = _live_section() if live is None else live
    auth = _live_auth(live)
    model, selected_voice = _live_model_voice(live, auth)
    config: Dict[str, Any] = {
        "model": model,
        "instructions": live_instructions(live),
        "audio": {"output": {"voice": selected_voice}},
        "delegation": {"type": "client"},
    }
    if history:
        config["initial_items" if auth == "subscription" else "input"] = history
    return config


def _create_subscription_session(sdp_offer: str, config: Dict[str, Any]) -> Dict[str, Any]:
    token, account = _subscription_credentials()
    # Codex frameless WebRTC: the bearer and its account identity never reach the renderer.
    # Do not follow redirects with subscription credentials or retry through the API route.
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.post(CODEX_LIVE_URL, json={"sdp": sdp_offer, "session": config}, headers={
                "Authorization": f"Bearer {token}", "ChatGPT-Account-Id": account,
                "OpenAI-Alpha": "quicksilver=v2",
            })
    except httpx.RequestError:
        raise RuntimeError("GPT-Live subscription connection failed; no API fallback was used") from None
    if not response.is_success:
        raise RuntimeError(
            f"GPT-Live subscription session was rejected (HTTP {response.status_code}); "
            "check Codex sign-in and voice access. No API fallback was used."
        )
    sdp = response.text
    call_id = urlparse(response.headers.get("Location", "")).path.rstrip("/").rsplit("/", 1)[-1]
    if not sdp.startswith("v=0") or not re.fullmatch(r"rtc_[A-Za-z0-9_-]+|[0-9a-fA-F-]{36}", call_id):
        raise RuntimeError("GPT-Live subscription returned an invalid WebRTC answer; no API fallback was used")
    return {"auth": "subscription", "session": {"id": call_id},
            "transport": {"type": "webrtc", "sdp": sdp}}


def create_webrtc_session(sdp_offer: str, history: Optional[list] = None) -> Dict[str, Any]:
    """Exchange the renderer's SDP offer for the Live session answer.

    Returns the vendor response ``{"session": {"id": ...}, "transport": {"type": "webrtc",
    "sdp": ...}}``. Raises ``ValueError`` for a missing key and ``RuntimeError`` (with the vendor
    status/detail) for a rejected request.
    """
    live = _live_section()
    auth = _live_auth(live)
    config = build_session_config(history, live=live)
    if auth == "subscription":
        return _create_subscription_session(sdp_offer, config)
    api_key, base_url = _resolve_credentials(live)
    if not api_key:
        raise ValueError("GPT-Live needs an OpenAI API key (OPENAI_API_KEY or voice.gpt_live.api_key)")
    body = json.dumps({
        "session": config,
        "transport": {"type": "webrtc", "sdp": sdp_offer},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/live/sessions", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            result["auth"] = "api"
            return result
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        logger.warning("GPT-Live session creation failed: %s %s", exc.code, detail)
        raise RuntimeError(f"GPT-Live session creation failed ({exc.code}): {detail}") from exc
