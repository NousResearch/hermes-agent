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

Azure: when ``voice.gpt_live.base_url`` points at an Azure OpenAI resource
(``*.openai.azure.com`` / ``*.cognitiveservices.azure.com``), the same mode switches to the
Azure Realtime API, which has no ``/live/sessions`` SDP exchange — the session is created over
a WebSocket (``wss://<resource>.openai.azure.com/openai/realtime?api-version=...&deployment=...``,
authenticated with the ``api-key`` header) via Azure-flavored ``session.update`` / session
events. ``voice.gpt_live.model`` is then the Azure *deployment* name and the key is the Azure
resource key. Default (OpenAI) behavior is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GPT_LIVE_MODE = "gpt-live"
CHAINED_MODE = "chained"
DEFAULT_LIVE_MODEL = "gpt-live-1"
DEFAULT_LIVE_VOICE = "marin"
DEFAULT_LIVE_BASE_URL = "https://api.openai.com/v1"
# Endpoint families recognized as Azure OpenAI resources (Foundry / cognitiveservices).
AZURE_HOST_SUFFIXES = (".openai.azure.com", ".cognitiveservices.azure.com")
# Azure Realtime API version used for the WebSocket session handshake.
AZURE_REALTIME_API_VERSION = "2025-04-01-preview"
BACKEND_OPENAI = "openai"
BACKEND_AZURE = "azure"
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


def _voice_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        voice = load_config().get("voice")
    except Exception:
        return {}
    return voice if isinstance(voice, dict) else {}


def _live_section(voice: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    section = (voice if voice is not None else _voice_section()).get("gpt_live")
    return section if isinstance(section, dict) else {}


def voice_chat_mode(voice: Optional[dict[str, Any]] = None) -> str:
    """``chained`` (default) or ``gpt-live``. Accepts the underscore spelling too."""
    raw = (voice if voice is not None else _voice_section()).get("voice_chat_mode")
    mode = str(raw or CHAINED_MODE).strip().lower().replace("_", "-")
    return GPT_LIVE_MODE if mode in {GPT_LIVE_MODE, "gptlive", "live"} else CHAINED_MODE


def _resolve_credentials(live: dict[str, Any]) -> tuple[str, str]:
    """``(api_key, base_url)`` — ``voice.gpt_live.api_key`` first, else the same OpenAI audio
    chain the STT/TTS providers use (``VOICE_TOOLS_OPENAI_KEY`` → ``OPENAI_API_KEY`` → pool).
    On an Azure endpoint the key is the Azure resource key (``api-key`` header, not Bearer).

    The Nous-managed audio proxy does not carry ``/live/sessions``; this mode is direct-key only.
    """
    from tools.tool_backend_helpers import resolve_openai_audio_api_key
    api_key = str(live.get("api_key") or "").strip() or resolve_openai_audio_api_key()
    base_url = str(live.get("base_url") or DEFAULT_LIVE_BASE_URL).strip().rstrip("/")
    return api_key, base_url


def live_backend(base_url: str) -> str:
    """Which vendor the configured ``base_url`` belongs to: ``"azure"`` for an Azure OpenAI
    resource (``*.openai.azure.com`` / ``*.cognitiveservices.azure.com``), else ``"openai"``."""
    host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    if any(host.endswith(suffix) for suffix in AZURE_HOST_SUFFIXES):
        return BACKEND_AZURE
    return BACKEND_OPENAI


def azure_realtime_ws_url(base_url: str, deployment: str, api_version: str = AZURE_REALTIME_API_VERSION) -> str:
    """``wss://<resource>.openai.azure.com/openai/realtime?api-version=...&deployment=...``
    built from the configured https ``base_url`` (an ``.../openai`` path is tolerated)."""
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    path = parsed.path.rstrip("/")
    if path and not path.endswith("/openai"):
        path = f"{path}/openai"
    elif not path:
        path = "/openai"
    query = urllib.parse.urlencode({"api-version": api_version, "deployment": deployment})
    return urllib.parse.urlunparse((scheme, parsed.netloc, f"{path}/realtime", "", query, ""))


def live_instructions(live: Optional[dict[str, Any]] = None) -> str:
    extra = str((live if live is not None else _live_section()).get("instructions") or "").strip()
    return f"{LIVE_PERSONA}\n\n{extra}" if extra else LIVE_PERSONA


def resolve_gpt_live_status() -> dict[str, Any]:
    """Non-secret readiness verdict for the client: which mode is selected and whether GPT-Live
    can start (a key resolves). Never returns the key."""
    voice = _voice_section()
    mode = voice_chat_mode(voice)
    live = _live_section(voice)
    api_key, base_url = _resolve_credentials(live)
    backend = live_backend(base_url)
    return {
        "mode": mode,
        "available": bool(api_key),
        "reason": None if api_key else "no API key (set voice.gpt_live.api_key, VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)",
        "backend": backend,
        "model": str(live.get("model") or DEFAULT_LIVE_MODEL),
        "voice": str(live.get("voice") or DEFAULT_LIVE_VOICE),
    }


def build_session_config(history: Optional[list] = None) -> dict[str, Any]:
    """The ``session`` object for ``POST /v1/live/sessions`` (client delegation, WebRTC — the
    transport negotiates the audio format, so none is set)."""
    live = _live_section()
    config: dict[str, Any] = {
        "model": str(live.get("model") or DEFAULT_LIVE_MODEL),
        "instructions": live_instructions(live),
        "audio": {"output": {"voice": str(live.get("voice") or DEFAULT_LIVE_VOICE)}},
        "delegation": {"type": "client"},
    }
    if history:
        config["input"] = history
    return config


def build_azure_session_config(history: Optional[list] = None) -> dict[str, Any]:
    """The Azure-flavored ``session`` object for the ``session.update`` event: the deployment
    rides in the WebSocket URL (not in the payload), the voice sits at session top level, and
    PCM is pinned explicitly because there is no WebRTC transport to negotiate it. Azure's
    realtime API has no ``delegation`` client mode — Hermes turns are driven as ordinary
    conversation items over the socket instead."""
    live = _live_section()
    config: dict[str, Any] = {
        "instructions": live_instructions(live),
        "voice": str(live.get("voice") or DEFAULT_LIVE_VOICE),
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {"type": "server_vad"},
    }
    if history:
        config["input"] = history
    return config


def _azure_ws_connect(url: str, api_key: str, timeout: float):
    """Open the Azure realtime WebSocket with the ``api-key`` header. Returns a minimal
    send/recv/close wrapper; lazily imports the client lib (``websockets`` is a core dep on
    Python 3.14+, ``websocket-client`` is the fallback elsewhere)."""
    try:
        from websockets.sync.client import connect as _ws_connect

        class _Sync:
            def __init__(self, ws):
                self._ws = ws

            def send(self, message: str) -> None:
                self._ws.send(message)

            def recv(self, timeout: Optional[float] = None) -> str:
                return str(self._ws.recv(timeout=timeout))

            def close(self) -> None:
                self._ws.close()

        return _Sync(_ws_connect(url, additional_headers={"api-key": api_key}, open_timeout=timeout))
    except ImportError:
        pass
    try:
        import websocket as _websocket_client  # type: ignore[import-not-found]

        class _SyncFallback:
            def __init__(self, ws):
                self._ws = ws

            def send(self, message: str) -> None:
                self._ws.send(message)

            def recv(self, timeout: Optional[float] = None) -> str:
                if timeout:
                    self._ws.settimeout(timeout)
                return str(self._ws.recv())

            def close(self) -> None:
                self._ws.close()

        return _SyncFallback(_websocket_client.create_connection(url, header=[f"api-key: {api_key}"], timeout=timeout))
    except ImportError:
        raise RuntimeError(
            "Azure realtime transport needs the 'websockets' (or 'websocket-client') package, "
            "which is not installed in this environment"
        )


def create_azure_realtime_session(sdp_offer: Optional[str] = None, history: Optional[list] = None,
                                  timeout: float = 30.0) -> dict[str, Any]:
    """Azure path for gpt-live session creation: Azure Realtime has no ``/live/sessions`` SDP
    exchange, so the session is created over a WebSocket — connect to
    ``wss://<resource>.openai.azure.com/openai/realtime?api-version=...&deployment=<model>``
    with the ``api-key`` header, read ``session.created``, send ``session.update`` and wait for
    its ``session.updated`` ack (or an ``error`` event).

    Mirrors the ``create_webrtc_session`` contract as closely as Azure allows: returns
    ``{"session": {"id": ...}, "transport": {"type": "websocket", "url": ...}}`` (no SDP —
    negotiation happens over the socket). The ``sdp_offer`` argument is accepted for call-site
    compatibility and ignored. The API key never leaves this host. Raises ``ValueError`` for a
    missing key and ``RuntimeError`` (with the vendor error detail) for a rejected session.
    """
    live = _live_section()
    api_key, base_url = _resolve_credentials(live)
    if not api_key:
        raise ValueError("GPT-Live on an Azure endpoint needs an API key (voice.gpt_live.api_key, VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)")
    deployment = str(live.get("model") or DEFAULT_LIVE_MODEL)
    ws_url = azure_realtime_ws_url(base_url, deployment)
    ws = _azure_ws_connect(ws_url, api_key, timeout)
    session_id: Optional[str] = None
    try:
        deadline = time.monotonic() + timeout

        def _next_event() -> dict[str, Any]:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                raw = ws.recv(timeout=remaining)
            except Exception as exc:
                raise RuntimeError(f"Azure realtime session creation failed (socket closed): {exc}") from exc
            try:
                return json.loads(raw)
            except ValueError:
                return {"type": "_unparsed", "raw": raw[:200]}

        # Azure greets a new socket with session.created (carries the session id).
        while session_id is None:
            event = _next_event()
            if event.get("type") == "session.created":
                session = event.get("session") or {}
                session_id = session.get("id")
            elif event.get("type") == "error":
                err = event.get("error") or {}
                raise RuntimeError(f"Azure realtime session creation failed: {err.get('message') or event}")

        ws.send(json.dumps({"type": "session.update", "session": build_azure_session_config(history)}))
        while True:
            event = _next_event()
            kind = event.get("type")
            if kind == "session.updated":
                break
            if kind == "error":
                err = event.get("error") or {}
                raise RuntimeError(f"Azure realtime session.update rejected: {err.get('message') or event}")
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return {
        "session": {"id": session_id, "model": deployment},
        "transport": {"type": "websocket", "url": ws_url, "api_version": AZURE_REALTIME_API_VERSION},
    }


def create_webrtc_session(sdp_offer: str, history: Optional[list] = None) -> dict[str, Any]:
    """Exchange the renderer's SDP offer for the Live session answer (OpenAI) or create the
    session over the Azure Realtime WebSocket when ``voice.gpt_live.base_url`` is an Azure
    endpoint.

    Returns the vendor response ``{"session": {"id": ...}, "transport": {"type": "webrtc",
    "sdp": ...}}`` on OpenAI, or ``{"session": {"id": ...}, "transport": {"type": "websocket",
    "url": ...}}`` on Azure. Raises ``ValueError`` for a missing key and ``RuntimeError`` (with
    the vendor status/detail) for a rejected request.
    """
    live = _live_section()
    api_key, base_url = _resolve_credentials(live)
    if not api_key:
        raise ValueError("GPT-Live needs an API key (voice.gpt_live.api_key, VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)")
    if live_backend(base_url) == BACKEND_AZURE:
        return create_azure_realtime_session(sdp_offer, history)
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
