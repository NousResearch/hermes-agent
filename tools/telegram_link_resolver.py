"""Helpers for resolving private Telegram deep links from Hermes history."""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

from hermes_state import SessionDB


_PRIVATE_TELEGRAM_HOSTS = {
    "t.me",
    "www.t.me",
    "telegram.me",
    "www.telegram.me",
}


def parse_private_telegram_link(url: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse ``t.me/c/...`` links into Telegram chat/thread/message identifiers."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc.lower() not in _PRIVATE_TELEGRAM_HOSTS:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts or parts[0] != "c":
        return None

    if len(parts) == 3:
        _, internal_chat_id, message_id = parts
        thread_id = None
    elif len(parts) >= 4:
        _, internal_chat_id, thread_id, message_id = parts[:4]
    else:
        return None

    if not internal_chat_id.isdigit() or not message_id.isdigit():
        return None
    if thread_id is not None and not thread_id.isdigit():
        return None

    return {
        "chat_id": f"-100{internal_chat_id}",
        "thread_id": thread_id,
        "message_id": message_id,
        "internal_chat_id": internal_chat_id,
    }


def resolve_private_telegram_link(
    url: str,
    db_path=None,
) -> Optional[Dict[str, Any]]:
    """Resolve a private Telegram deep link from Hermes' local transcript store."""
    link = parse_private_telegram_link(url)
    if not link:
        return None

    db = SessionDB(db_path=db_path)
    try:
        message = db.find_telegram_message(
            chat_id=link["chat_id"],
            message_id=link["message_id"],
            thread_id=link["thread_id"],
        )
    finally:
        db.close()

    if not message:
        thread_text = (
            f" topic {link['thread_id']}" if link["thread_id"] is not None else ""
        )
        return {
            "url": url,
            "title": "Telegram private message",
            "content": "",
            "error": (
                "Private Telegram links are not directly web-accessible. "
                f"Hermes could not resolve chat {link['chat_id']}{thread_text} "
                f"message {link['message_id']} from its stored Telegram history."
            ),
            "metadata": {
                "platform": "telegram",
                **link,
            },
        }

    body = (message.get("content") or "").strip()
    if not body:
        body = (
            "[Hermes observed this Telegram message, but no text content was stored. "
            "It may have been media-only or reduced during ingestion.]"
        )

    return {
        "url": url,
        "title": f"Telegram message {link['message_id']}",
        "content": body,
        "metadata": {
            "platform": "telegram",
            "session_id": message.get("session_id"),
            "role": message.get("role"),
            "timestamp": message.get("timestamp"),
            **link,
        },
    }
