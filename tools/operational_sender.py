"""Fail-closed sender for verified SoLo Hermes operational notifications."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

EXPECTED_USERNAME = "solo_hermes_bot"
DEFAULT_CHAT_ID = "8148316720"


def _api_call(token: str, method: str, data: dict[str, str]) -> dict:
    request = Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=urlencode(data).encode(),
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Telegram operational sender failed: {type(exc).__name__}") from exc
    if not payload.get("ok"):
        raise RuntimeError("Telegram operational sender rejected the request")
    return payload


def send_operational_message(message: str, chat_id: str = DEFAULT_CHAT_ID) -> dict:
    """Verify @solo_hermes_bot and send one bounded operational message.

    There is deliberately no fallback to TELEGRAM_BOT_TOKEN or a gateway
    adapter: operational notifications must never silently use Halo.
    """
    token = os.environ.get("SOLO_HERMES_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SOLO_HERMES_BOT_TOKEN is not configured")
    identity = _api_call(token, "getMe", {})
    username = str(identity.get("result", {}).get("username", "")).lower()
    if username != EXPECTED_USERNAME:
        raise RuntimeError("operational sender identity verification failed")
    return _api_call(token, "sendMessage", {"chat_id": str(chat_id), "text": message})