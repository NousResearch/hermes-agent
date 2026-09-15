"""Grok-Live voice chat mode: the xAI realtime sibling engine that delegates to Hermes.

``voice.voice_chat_mode: grok-live`` replaces the chained STT → turn → TTS loop with ONE
full-duplex voice model (xAI Grok Voice) that owns the microphone and the speaker and delegates
every real request to Hermes, exactly like gpt-live. Hermes stays the agent: whatever
model/provider the session has selected answers, with the full toolset.

Architectural difference from gpt-live: the backend (gateway process) holds the xAI websocket
exclusively (``wss://api.x.ai/v1/realtime``) — there is no renderer-negotiated WebRTC leg. The
renderer relays base64 PCM16 chunks over the existing JSON-RPC channel (``voice.grok.*``
methods, see docs/grok-live-voice/SPEC.md §4). The websocket bridge itself is a sibling card;
THIS module owns only the config/credential/status surface:

* ``resolve_grok_live_status()`` — non-secret readiness verdict for the client;
* ``_resolve_grok_credentials()`` — AUTHORITATIVE auth order (SPEC §8): OAuth-first in ``auto``
  mode, NEVER XAI_API_KEY-preferred (on this box the BSM-injected XAI_API_KEY belongs to a
  zero-credit team and 403s everywhere);
* ``live_instructions()`` — the short voice-layer persona shipped as xAI
  ``session.update.instructions`` (never the Hermes system prompt).

``voice_chat_mode()`` is intentionally NOT re-implemented here — the shared resolver in
``tools/voice_live.py`` is the single source of truth for all three engines (SPEC §1).
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from tools.voice_live import GROK_LIVE_MODE, _voice_section, voice_chat_mode

logger = logging.getLogger(__name__)

DEFAULT_GROK_LIVE_MODEL = "grok-voice-latest"
DEFAULT_GROK_LIVE_VOICE = "eve"

NO_CREDENTIAL_REASON = "no xAI credential (SuperGrok login via `hermes login` or set XAI_API_KEY)"

# Persona for the voice layer, same shape as gpt-live's LIVE_PERSONA: short on purpose (role +
# style + a labelled delegation policy), because the live model has a small context window and
# the backend (Hermes) carries the real instructions, tools and memory. This text is shipped to
# xAI as ``session.update.instructions`` — it is never the Hermes system prompt, so it has no
# prompt-cache footprint on the Hermes side.
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


def _grok_live_section(voice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    section = (voice if voice is not None else _voice_section()).get("grok_live")
    return section if isinstance(section, dict) else {}


def _hermes_home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _oauth_token() -> str:
    """Load the xai-oauth (SuperGrok device-code) access token from ``$HERMES_HOME/auth.json``,
    refreshing it via the discovery token endpoint when expired (2-minute skew). Refreshed
    tokens are written back to the shared store so the CLI's own xai-oauth lane keeps working.

    Ported from the proven standalone reference (``/home/ericio/grok_live_voice.py``) with one
    hardening change (SPEC §13 risk 4): the write-back is atomic (temp file + ``os.replace``)
    because this runs inside a long-lived multi-request gateway process where a torn write
    could be observed by a concurrent reader.

    Never logged, never returned to the renderer — this value only ever builds the
    ``Authorization`` header of the backend's own websocket to xAI.
    """
    auth_path = _hermes_home() / "auth.json"
    try:
        data = json.loads(auth_path.read_text())
        prov = data["providers"]["xai-oauth"]
        toks = prov["tokens"]
    except Exception:
        return ""

    def _expired() -> bool:
        try:
            lr = datetime.datetime.fromisoformat(
                str(prov.get("last_refresh", "")).replace("Z", "+00:00"))
            exp = lr + datetime.timedelta(seconds=int(toks.get("expires_in", 0)))
            skew = datetime.timedelta(minutes=2)
            return exp - skew <= datetime.datetime.now(datetime.timezone.utc)
        except Exception:
            return False

    if toks.get("access_token") and not _expired():
        return toks["access_token"]

    # Refresh path (device-code OAuth: public client, no secret). The client_id is the ``aud``
    # claim of the id_token JWT — undocumented but working, same as the reference script.
    try:
        payload = toks["id_token"].split(".")[1]
        payload += "=" * (-len(payload) % 4)
        client_id = json.loads(base64.urlsafe_b64decode(payload)).get("aud", "")
        body = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": toks["refresh_token"],
            "client_id": client_id,
        }).encode()
        req = urllib.request.Request(
            prov["discovery"]["token_endpoint"], data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        new = json.loads(urllib.request.urlopen(req, timeout=15).read())
        toks.update({k: new[k] for k in
                     ("access_token", "refresh_token", "id_token", "expires_in")
                     if k in new})
        prov["last_refresh"] = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        tmp_path = auth_path.with_name(auth_path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        os.replace(tmp_path, auth_path)
        return str(toks.get("access_token", ""))
    except Exception as exc:
        logger.debug("grok-live: xai-oauth token refresh failed: %s", exc)
        return ""


def _xai_api_key() -> str:
    """``XAI_API_KEY`` from the process env, else ``$HERMES_HOME/.env``. Fallback only — in
    ``auto`` mode the OAuth token is always preferred (see ``_resolve_grok_credentials``)."""
    key = os.environ.get("XAI_API_KEY", "").strip().strip('"').strip("'")
    if key:
        return key
    env_path = _hermes_home() / ".env"
    if env_path.exists():
        for line in env_path.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == "XAI_API_KEY":
                return v.strip().strip('"').strip("'")
    return ""


def _resolve_grok_credentials(live: Optional[Dict[str, Any]] = None) -> str:
    """The bearer credential for the xAI realtime websocket, per ``voice.grok_live.auth``
    (SPEC §8 — AUTHORITATIVE order):

    * ``oauth``  → SuperGrok OAuth token only;
    * ``apikey`` → ``XAI_API_KEY`` only;
    * ``auto`` (default) → OAuth token FIRST (refreshed if needed), THEN ``XAI_API_KEY``.
      Never the reverse: a zero-credit API key sitting in the environment must not shadow a
      working OAuth token.
    """
    live = live if live is not None else _grok_live_section()
    auth = str(live.get("auth") or "auto").strip().lower()
    if auth == "oauth":
        return _oauth_token()
    if auth == "apikey":
        return _xai_api_key()
    return _oauth_token() or _xai_api_key()


def live_instructions(live: Optional[Dict[str, Any]] = None) -> str:
    """The voice-layer persona plus any ``voice.grok_live.instructions`` sentences appended."""
    extra = str((live if live is not None else _grok_live_section()).get("instructions") or "").strip()
    return f"{GROK_LIVE_PERSONA}\n\n{extra}" if extra else GROK_LIVE_PERSONA


def resolve_grok_live_status() -> Dict[str, Any]:
    """Non-secret readiness verdict for the client: which mode is selected and whether
    Grok-Live can start (a credential resolves per ``_resolve_grok_credentials``).

    Mirrors ``tools/voice_live.py::resolve_gpt_live_status`` 1:1 in shape. ``available=True``
    does NOT mean the websocket will succeed — network/xAI-side failures are still possible —
    it means a credential resolves. Never includes the credential itself. Note: an expired
    OAuth token triggers ONE token-endpoint round-trip here (bounded 15s timeout) to decide
    refreshability — that is the "present and refreshable" half of the contract.
    """
    voice = _voice_section()
    mode = voice_chat_mode(voice)
    live = _grok_live_section(voice)
    credential = _resolve_grok_credentials(live)
    return {
        "mode": mode,
        "available": bool(credential),
        "reason": None if credential else NO_CREDENTIAL_REASON,
        "model": str(live.get("model") or DEFAULT_GROK_LIVE_MODEL),
        "voice": str(live.get("voice") or DEFAULT_GROK_LIVE_VOICE),
    }
