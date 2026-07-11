"""Telegram admin tools for pinning/unpinning messages.

Uses the configured Telegram bot token from the Hermes gateway config or
TELEGRAM_BOT_TOKEN environment variable. Never returns or logs the token.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from agent.redact import redact_sensitive_text
from tools.registry import registry

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


def _sanitize_error_text(text: Any) -> str:
    return redact_sensitive_text(str(text))


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": _sanitize_error_text(message)})


def _get_telegram_token() -> Optional[str]:
    """Resolve Telegram bot token from gateway config/env without exposing it."""
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.TELEGRAM)
        if pconfig and pconfig.enabled and pconfig.token:
            return str(pconfig.token).strip()
    except Exception:
        logger.debug("Could not load Telegram token from gateway config", exc_info=True)

    # Fallback: environment only. Do not print this value.
    try:
        import os

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        return token or None
    except Exception:
        return None


def _check_telegram_admin_requirements() -> bool:
    return bool(_get_telegram_token())


def _telegram_api_request(method: str, payload: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    token = _get_telegram_token()
    if not token:
        raise RuntimeError("Telegram bot token is not configured")

    url = f"{TELEGRAM_API_BASE}{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent TelegramAdminTool",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {"description": body}
        return {
            "ok": False,
            "error_code": exc.code,
            "description": parsed.get("description") or f"HTTP {exc.code}",
        }


def _parse_telegram_target(target: str = "", chat_id: str = "", thread_id: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Resolve a Telegram target string to (chat_id, thread_id)."""
    target = (target or "").strip()
    chat_id = (chat_id or "").strip()
    thread_id = str(thread_id or "").strip() or None

    if target:
        if target.startswith("telegram:"):
            ref = target.split(":", 1)[1]
        elif target == "telegram":
            ref = ""
        else:
            ref = target

        if ref:
            # Numeric explicit Telegram target: chat_id[:thread_id]
            parts = ref.split(":")
            if len(parts) >= 1 and parts[0].lstrip("-").isdigit():
                chat_id = parts[0]
                if len(parts) >= 2 and parts[1].isdigit():
                    thread_id = parts[1]
            else:
                try:
                    from gateway.channel_directory import resolve_channel_name

                    resolved = resolve_channel_name("telegram", ref)
                    if resolved:
                        parts = str(resolved).split(":")
                        chat_id = parts[0]
                        if len(parts) >= 2 and parts[1].isdigit():
                            thread_id = parts[1]
                except Exception:
                    pass

    if not chat_id:
        try:
            from gateway.config import Platform, load_gateway_config

            config = load_gateway_config()
            home = config.get_home_channel(Platform.TELEGRAM)
            if home:
                chat_id = str(home.chat_id)
        except Exception:
            pass

    return (chat_id or None), thread_id


def _extract_message_id(send_result: Dict[str, Any]) -> Optional[int]:
    mid = send_result.get("message_id")
    if mid is None:
        return None
    try:
        return int(mid)
    except (TypeError, ValueError):
        return None


def _pin(chat_id: str, message_id: int, disable_notification: bool = True) -> Dict[str, Any]:
    payload = {
        "chat_id": int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id,
        "message_id": int(message_id),
        "disable_notification": bool(disable_notification),
    }
    resp = _telegram_api_request("pinChatMessage", payload)
    if resp.get("ok"):
        return {
            "success": True,
            "action": "pin_message",
            "platform": "telegram",
            "chat_id": str(chat_id),
            "message_id": str(message_id),
        }
    return {
        "success": False,
        "action": "pin_message",
        "platform": "telegram",
        "chat_id": str(chat_id),
        "message_id": str(message_id),
        "error": _sanitize_error_text(resp.get("description") or resp),
        "error_code": resp.get("error_code"),
    }


def _unpin(chat_id: str, message_id: Optional[int] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "chat_id": int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id,
    }
    if message_id is not None:
        payload["message_id"] = int(message_id)
    resp = _telegram_api_request("unpinChatMessage", payload)
    if resp.get("ok"):
        return {
            "success": True,
            "action": "unpin_message",
            "platform": "telegram",
            "chat_id": str(chat_id),
            "message_id": str(message_id) if message_id is not None else None,
        }
    return {
        "success": False,
        "action": "unpin_message",
        "platform": "telegram",
        "chat_id": str(chat_id),
        "message_id": str(message_id) if message_id is not None else None,
        "error": _sanitize_error_text(resp.get("description") or resp),
        "error_code": resp.get("error_code"),
    }


def telegram_admin_tool(args, **kw) -> str:
    """Registry-compatible handler for Telegram admin actions."""
    action = (args.get("action") or "").strip()
    target = args.get("target") or ""
    chat_id_arg = args.get("chat_id") or ""
    thread_id_arg = args.get("thread_id") or ""
    message_id = args.get("message_id")
    disable_notification = args.get("disable_notification", True)
    message = args.get("message") or ""

    chat_id, thread_id = _parse_telegram_target(target, chat_id_arg, thread_id_arg)
    if not chat_id:
        return _error("No Telegram chat_id resolved. Provide target='telegram:CHAT_ID[:THREAD_ID]' or chat_id.")

    if action == "pin_message":
        if message_id in (None, ""):
            return _error("message_id is required for pin_message")
        try:
            result = _pin(chat_id, int(message_id), bool(disable_notification))
            return json.dumps(result)
        except Exception as exc:
            logger.exception("Telegram pin_message failed")
            return _error(f"Telegram pin_message failed: {exc}")

    if action == "unpin_message":
        try:
            mid = int(message_id) if message_id not in (None, "") else None
            result = _unpin(chat_id, mid)
            return json.dumps(result)
        except Exception as exc:
            logger.exception("Telegram unpin_message failed")
            return _error(f"Telegram unpin_message failed: {exc}")

    if action == "send_and_pin":
        if not message:
            return _error("message is required for send_and_pin")
        try:
            from tools.send_message_tool import _handle_send

            send_target = f"telegram:{chat_id}"
            if thread_id:
                send_target += f":{thread_id}"
            send_raw = _handle_send({"target": send_target, "message": message})
            try:
                send_result = json.loads(send_raw)
            except Exception:
                return _error(f"send_message returned non-JSON result: {send_raw}")
            if not send_result.get("success"):
                return json.dumps({
                    "success": False,
                    "action": "send_and_pin",
                    "send_result": send_result,
                    "error": _sanitize_error_text(send_result.get("error") or "send_message failed"),
                })
            sent_message_id = _extract_message_id(send_result)
            if sent_message_id is None:
                return json.dumps({
                    "success": False,
                    "action": "send_and_pin",
                    "send_result": send_result,
                    "error": "send_message succeeded but no message_id was returned to pin",
                })
            pin_result = _pin(chat_id, sent_message_id, bool(disable_notification))
            return json.dumps({
                "success": bool(pin_result.get("success")),
                "action": "send_and_pin",
                "platform": "telegram",
                "chat_id": str(chat_id),
                "thread_id": str(thread_id) if thread_id else None,
                "message_id": str(sent_message_id),
                "send_result": send_result,
                "pin_result": pin_result,
                "error": pin_result.get("error"),
            })
        except Exception as exc:
            logger.exception("Telegram send_and_pin failed")
            return _error(f"Telegram send_and_pin failed: {exc}")

    return json.dumps({
        "success": False,
        "error": f"Unknown action: {action}",
        "available_actions": ["pin_message", "unpin_message", "send_and_pin"],
    })


TELEGRAM_ADMIN_SCHEMA = {
    "name": "telegram_admin",
    "description": (
        "Telegram admin actions for pinning/unpinning messages. Requires the Telegram bot "
        "to be an admin with Pin Messages permission. For existing messages, use action='pin_message' "
        "with chat_id/target and message_id. To send a new pinned message, use action='send_and_pin'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["pin_message", "unpin_message", "send_and_pin"],
                "description": "Admin action to perform.",
            },
            "target": {
                "type": "string",
                "description": "Telegram target, e.g. 'telegram:-1001234567890:63' or a resolvable Telegram channel/topic name.",
            },
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID, e.g. -1001234567890. Alternative to target.",
            },
            "thread_id": {
                "type": "string",
                "description": "Optional Telegram forum topic/thread ID for send_and_pin targeting.",
            },
            "message_id": {
                "type": "integer",
                "description": "Telegram message_id to pin/unpin. Not needed for send_and_pin.",
            },
            "message": {
                "type": "string",
                "description": "Message text to send when action='send_and_pin'. Supports the same markdown/media handling as send_message.",
            },
            "disable_notification": {
                "type": "boolean",
                "description": "Whether to pin silently without notifying everyone. Defaults to true.",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="telegram_admin",
    toolset="telegram_admin",
    schema=TELEGRAM_ADMIN_SCHEMA,
    handler=telegram_admin_tool,
    check_fn=_check_telegram_admin_requirements,
    requires_env=["TELEGRAM_BOT_TOKEN"],
    emoji="📌",
)
