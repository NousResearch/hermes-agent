"""Fail-closed sender for verified Hermes operational notifications."""
from __future__ import annotations

import json
import os
from pathlib import Path

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

EXPECTED_USERNAME = "solo_hermes_bot"
EXPECTED_BOT_ID = 8611668567
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


def send_operational_message(message: str, chat_id: str = DEFAULT_CHAT_ID) -> dict:
    """Verify @solo_hermes_bot and send one operational message.

    There is deliberately no fallback to TELEGRAM_BOT_TOKEN or a gateway
    adapter: operational notifications must never silently use a profile bot.
    """
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None
    token = (get_secret("SOLO_HERMES_BOT_TOKEN", "") if get_secret else "") or os.environ.get(
        "SOLO_HERMES_BOT_TOKEN", ""
    )
    token = token.strip()
    if not token:
        raise RuntimeError("SOLO_HERMES_BOT_TOKEN is not configured")
    identity = _api_call(token, "getMe", {})
    bot = identity.get("result", {})
    if (bot.get("id") != EXPECTED_BOT_ID
            or str(bot.get("username", "")).lower() != EXPECTED_USERNAME
            or bot.get("is_bot") is not True):
        raise RuntimeError("operational sender identity verification failed")
    if str(chat_id) != DEFAULT_CHAT_ID:
        raise RuntimeError("operational sender destination verification failed")
    result = _api_call(token, "sendMessage", {"chat_id": DEFAULT_CHAT_ID, "text": message})
    sent = result.get("result", {})
    sent_from, sent_chat = sent.get("from", {}), sent.get("chat", {})
    if (sent_from.get("id") != EXPECTED_BOT_ID
            or str(sent_from.get("username", "")).lower() != EXPECTED_USERNAME
            or sent_from.get("is_bot") is not True
            or str(sent_chat.get("id")) != DEFAULT_CHAT_ID
            or "message_thread_id" in sent):
        raise RuntimeError("operational sender delivery proof failed")
    return result


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
    bot = identity.get("result", {})
    if (bot.get("id") != EXPECTED_BOT_ID
            or str(bot.get("username", "")).lower() != EXPECTED_USERNAME
            or bot.get("is_bot") is not True):
        raise RuntimeError("operational sender identity verification failed")
    path = Path(file_path)
    boundary = "----hermes-operational-sender"
    with path.open("rb") as artifact:
        payload = artifact.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=chat_id\r\n\r\n{DEFAULT_CHAT_ID}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=document; filename={path.name}\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
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
    sent = result.get("result", {})
    sent_from, sent_chat = sent.get("from", {}), sent.get("chat", {})
    if (sent_from.get("id") != EXPECTED_BOT_ID
            or str(sent_from.get("username", "")).lower() != EXPECTED_USERNAME
            or sent_from.get("is_bot") is not True
            or str(sent_chat.get("id")) != DEFAULT_CHAT_ID
            or "message_thread_id" in sent):
        raise RuntimeError("operational sender delivery proof failed")
    return result
