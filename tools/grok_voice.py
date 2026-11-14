"""Grok Voice session minting. The long-lived xAI credential never leaves the gateway.

``voice.voice_chat_mode: grok-live`` talks to xAI's realtime voice model
(``grok-voice-latest``, today ``grok-voice-3.0``). The desktop opens
``wss://api.x.ai/v1/realtime`` with a 300s client secret; this module is the only
place that holds the bearer used to mint it.

Auth order: the SuperGrok / X Premium+ sign-in
(``hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials``), then
``XAI_API_KEY`` from the profile env only when that sign-in is missing or raises.
``/v1/tts`` is a different endpoint and is not used here (the subscription bearer
403s on it).

Vendor shape, adapted from the hermes-xai voice plugin: POST
``{base}/realtime/client_secrets`` with ``{"expires_after": {"seconds": 300}}``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

DEFAULT_BASE = "https://api.x.ai/v1"
PROTOCOL_PREFIX = "xai-client-secret."
SECRET_TTL_S = 300
DEFAULT_MODEL = "grok-voice-latest"
DEFAULT_VOICE = "eve"
DEFAULT_REASONING = "none"
REASONING_EFFORTS = ("none", "high")
_MODEL_RE = re.compile(r"^grok-voice-[a-z0-9.\-]+$")
_VOICE_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

PERSONA = """# Role
You are Hermes, a calm, friendly voice assistant. You are the voice of an agent that works on the user's computer: the agent does the work, you hand it the tasks.

# Objective
Understand what the user wants, hand real work to the agent with `hermes_task`, and say the result when it arrives.

# Flow
- Anything that needs facts, current information, files, the web, memory, or an action: call `hermes_task` with a clear, complete request in the user's own words.
- When you call it, say one very short sentence ("on it"), then wait. Never guess a result.
- A message starting with "[Hermes result]" is the agent's answer: say it in one or two plain sentences. If it says the task failed, say so simply.
- A correction ("no, Thursday") on a running task: call `hermes_task` again with the corrected request.
- Greetings, small talk, or repeating what was just said: answer directly, no tool.
- When the user clearly says goodbye, say goodbye in one sentence, then call `end_call`.

# Limits
- Never say something is done before the result arrives.
- Do not name the tools or the agent; say "let me check", "it's sent".
- If the request is unclear, ask one short question.

# Style
Speak in the user's language. One or two short spoken sentences per turn, no lists, no symbols."""

TOOLS = [
    {
        "type": "function",
        "name": "hermes_task",
        "description": (
            "Hand a task to the Hermes agent (act, search, check, write, remember, use a service). "
            "Returns a receipt at once; the result arrives later in a message starting with "
            "\"[Hermes result]\"."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The complete request, as the user would type it, with every useful detail.",
                }
            },
            "required": ["request"],
        },
    },
    {
        "type": "function",
        "name": "end_call",
        "description": "End the conversation, only after a clear goodbye from the user and your own goodbye.",
        "parameters": {"type": "object", "properties": {}},
    },
]

Poster = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, Any]]


class GrokVoiceError(Exception):
    """Typed refusal. ``message`` is safe to show: it never contains the bearer or the secret."""

    def __init__(self, reason: str, message: str, status: int = 502):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status = status


def grok_live_settings(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """``voice.grok_live`` with defaults. Invalid model/voice/effort fall back locally
    so a typo never reaches xAI (the vendor accepts an unknown voice without error)."""
    raw = raw if isinstance(raw, dict) else {}
    model = str(raw.get("model") or "").strip()
    voice = str(raw.get("voice") or "").strip()
    effort = str(raw.get("reasoning_effort") or "").strip().lower()
    return {
        "model": model if _MODEL_RE.match(model) else DEFAULT_MODEL,
        "voice": voice if _VOICE_RE.match(voice) else DEFAULT_VOICE,
        "reasoning_effort": effort if effort in REASONING_EFFORTS else DEFAULT_REASONING,
        "instructions": str(raw.get("instructions") or "").strip()[:4000],
    }


def session_update(cfg: dict[str, Any]) -> dict[str, Any]:
    """The ``session`` object of the first ``session.update``. The desktop sends it unchanged."""
    extra = cfg.get("instructions") or ""
    return {
        "voice": cfg["voice"],
        "instructions": f"{PERSONA}\n\n{extra}" if extra else PERSONA,
        "reasoning": {"effort": cfg["reasoning_effort"]},
        "turn_detection": {"type": "server_vad"},
        "tools": TOOLS,
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": 24000}},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
    }


def _scrub(text: str, secret: str) -> str:
    if secret and secret in (text or ""):
        return text.replace(secret, "***")
    return text or ""


def _http_base(value: str) -> str:
    base = (value or "").strip().rstrip("/")
    if base.startswith("https://") or base.startswith("http://"):
        return base
    return DEFAULT_BASE


def _realtime_url(base: str, model: str) -> str:
    if base.startswith("https://"):
        ws = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        ws = "ws://" + base[len("http://"):]
    else:
        ws = "wss://api.x.ai/v1"
    return f"{ws}/realtime?model={urllib.parse.quote(model, safe='')}"


def _profile_secret(name: str) -> str:
    """Profile env via ``get_secret``. An unscoped multiplex read is a miss, not another profile's env."""
    from agent.secret_scope import UnscopedSecretError, get_secret

    try:
        return str(get_secret(name, "") or "").strip()
    except UnscopedSecretError:
        return ""


def _oauth_credential() -> Optional[tuple[str, str]]:
    """``(bearer, base_url)`` when the xAI OAuth sign-in resolves. None if it is missing or raises."""
    try:
        from hermes_cli.auth_xai import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
    except Exception:
        return None
    token = str((creds or {}).get("api_key") or "").strip()
    if not token:
        return None
    return token, _http_base(str((creds or {}).get("base_url") or ""))


def _api_key_credential() -> Optional[tuple[str, str]]:
    key = _profile_secret("XAI_API_KEY")
    if not key:
        return None
    return key, _http_base(_profile_secret("XAI_BASE_URL") or DEFAULT_BASE)


def resolve_grok_credential() -> tuple[str, str, str]:
    """``(bearer, auth, base_url)``. OAuth wins whenever it resolves; the API key is only the fallback.

    ``auth`` is ``oauth``, ``api_key``, or ``none``. The bearer is for the mint call only —
    callers that build a client response must not put it on the wire back to the desktop.
    """
    oauth = _oauth_credential()
    if oauth:
        token, base = oauth
        return token, "oauth", base
    api_key = _api_key_credential()
    if api_key:
        token, base = api_key
        return token, "api_key", base
    return "", "none", DEFAULT_BASE


def _grok_section() -> dict[str, Any]:
    from tools.voice_live import _voice_section

    section = _voice_section().get("grok_live")
    return section if isinstance(section, dict) else {}


def resolve_grok_voice_status() -> dict[str, Any]:
    """Non-secret readiness. ``auth`` is ``oauth``, ``api_key``, or ``none``. Never the bearer."""
    from tools.voice_live import voice_chat_mode

    cfg = grok_live_settings(_grok_section())
    _token, auth, _base = resolve_grok_credential()
    available = auth != "none"
    return {
        "mode": voice_chat_mode(),
        "available": available,
        "reason": None if available else "no xAI credential (hermes auth add xai-oauth or XAI_API_KEY)",
        "model": cfg["model"],
        "voice": cfg["voice"],
        "auth": auth,
    }


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")[:300]}


def _secret_from_body(body: Any, bearer: str) -> tuple[str, Optional[int]]:
    if not isinstance(body, dict):
        raise GrokVoiceError("rejected", "Unexpected answer from xAI.", 502)
    nested = body.get("client_secret")
    nested = nested if isinstance(nested, dict) else {}
    value = body.get("value") or nested.get("value")
    expires = body.get("expires_at")
    if expires is None:
        expires = nested.get("expires_at")
    if not isinstance(value, str) or not value or value == bearer:
        raise GrokVoiceError("rejected", "Unexpected answer from xAI.", 502)
    expires_at = int(expires) if isinstance(expires, (int, float)) else None
    return value, expires_at


def mint_grok_voice_session(poster: Optional[Poster] = None) -> dict[str, Any]:
    """Client secret + URL + ``session.update`` body. Raises :class:`GrokVoiceError`.

    The returned object contains the minted secret under ``client_secret.value`` and nowhere
    else. The long-lived bearer is not in the result and not in the error text.
    """
    bearer, _auth, base = resolve_grok_credential()
    if not bearer:
        raise GrokVoiceError(
            "no_credential",
            "Grok Voice needs a Grok sign-in (hermes auth add xai-oauth) or XAI_API_KEY on this profile.",
            503,
        )
    cfg = grok_live_settings(_grok_section())
    post = poster or _post_json
    url = f"{base}/realtime/client_secrets"
    try:
        status, body = post(
            url,
            {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
            {"expires_after": {"seconds": SECRET_TTL_S}},
            20.0,
        )
    except Exception as exc:
        raise GrokVoiceError("unreachable", _scrub(f"xAI is unreachable ({type(exc).__name__}).", bearer), 502) from None
    if status >= 400:
        raise GrokVoiceError("rejected", _scrub(f"xAI refused the client secret request ({status}).", bearer), 502)
    value, expires_at = _secret_from_body(body, bearer)
    return {
        "url": _realtime_url(base, cfg["model"]),
        "protocol_prefix": PROTOCOL_PREFIX,
        "client_secret": {"value": value, "expires_at": expires_at},
        "model": cfg["model"],
        "session": session_update(cfg),
    }
