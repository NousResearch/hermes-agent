"""Fail-closed sender for verified Hermes operational notifications."""
from __future__ import annotations

import json
import mimetypes
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
        raise RuntimeError(
            f"Telegram operational sender failed: {type(exc).__name__}"
        ) from exc
    if not payload.get("ok"):
        raise RuntimeError("Telegram operational sender rejected the request")
    return payload


def send_operational_message(message: str) -> dict:
    """Verify @solo_hermes_bot and send one operational message.

    There is deliberately no fallback to TELEGRAM_BOT_TOKEN or a gateway
    adapter: operational notifications must never silently use a profile bot.
    """
    # Cron ticks run under the owning profile's secret scope in a multiplexed
    # gateway. Resolve that scope first; the process environment is only a
    # compatibility fallback for standalone invocations.
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None
    token = (
        get_secret("SOLO_HERMES_BOT_TOKEN", "") if get_secret is not None else ""
    ) or ""
    if not token:
        import os
        token = os.environ.get("SOLO_HERMES_BOT_TOKEN", "")
    token = token.strip()
    if not token:
        raise RuntimeError("SOLO_HERMES_BOT_TOKEN is not configured")
    identity = _api_call(token, "getMe", {})
    username = str(identity.get("result", {}).get("username", "")).lower()
    if username != EXPECTED_USERNAME:
        raise RuntimeError("operational sender identity verification failed")
    return _api_call(token, "sendMessage", {"chat_id": DEFAULT_CHAT_ID, "text": message})


def send_operational_document(file_path: str) -> dict:
    """Send one artifact to the verified operational DM, without a topic."""
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None
    token = (get_secret("SOLO_HERMES_BOT_TOKEN", "") if get_secret else "") or os.environ.get("SOLO_HERMES_BOT_TOKEN", "")
    token = token.strip()
    if not token:
        raise RuntimeError("SOLO_HERMES_BOT_TOKEN is not configured")
    identity = _api_call(token, "getMe", {})
    if str(identity.get("result", {}).get("username", "")).lower() != EXPECTED_USERNAME:
        raise RuntimeError("operational sender identity verification failed")
    path = os.fspath(file_path)
    boundary = "----hermes-operational-sender"
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as artifact:
        payload = artifact.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=chat_id\r\n\r\n{DEFAULT_CHAT_ID}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=document; filename={os.path.basename(path)}\r\n"
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    request = Request(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Telegram operational sender failed: {type(exc).__name__}") from exc
    if not result.get("ok"):
        raise RuntimeError("Telegram operational sender rejected the request")
    return result
