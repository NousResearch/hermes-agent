"""Single content-bearing Telegram attempt, using the native managed bot factory."""

from datetime import timedelta
import math
import re


def is_plain_telegram_request(args):
    if not isinstance(args, dict) or set(args) - {"target", "message", "action"}:
        return False
    target, message = args.get("target"), args.get("message")
    return (
        isinstance(target, str) and target.partition(":")[0].strip().lower() == "telegram"
        and isinstance(message, str) and bool(message.strip())
        and len(message.encode("utf-16-le", errors="replace")) <= 2 * 4096
        and re.search(r"MEDIA:|\[\[(?:as_document|audio_as_voice)\]\]", message, re.IGNORECASE) is None
        and args.get("action", "send") == "send"
    )


def sender_options(entry, args, *, single_attempt=False):
    options = {"args": args} if entry is not None and entry.send_message_handler is not None else {}
    if single_attempt:
        options["single_attempt"] = True
    return options


async def send_telegram_once(pconfig, chat_id, message, *, thread_id=None):
    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
    from tools.send_message_senders import _success, _telegram_bot, _telegram_thread_kwargs
    try:
        from telegram.error import RetryAfter
    except ImportError:
        return {"success": False, "error": "TELEGRAM_DEPENDENCY_UNAVAILABLE", "delivery_status": "not_sent"}

    try:
        async with _telegram_bot(pconfig.token) as bot:
            try:
                sent = await bot.send_message(
                    chat_id=normalize_telegram_chat_id(chat_id), text=message, parse_mode=None,
                    message_thread_id=_telegram_thread_kwargs(thread_id).get("message_thread_id"),
                    disable_web_page_preview=True,
                )
            except RetryAfter as exc:
                result = {"success": False, "error": "TELEGRAM_RATE_LIMITED", "delivery_status": "not_sent"}
                delay = exc.retry_after
                seconds = delay.total_seconds() if isinstance(delay, timedelta) else delay
                if isinstance(seconds, (int, float)) and math.isfinite(seconds) and 0 <= seconds <= 86400:
                    result["retry_after"] = float(seconds)
                return result
        return _success("telegram", chat_id, message_id=str(sent.message_id), delivery_status="accepted")
    except Exception:  # health: allow BLE001 -- SDK exceptions can contain token/body; fixed unknown code is the privacy boundary
        # Neither exception text nor a failed transport proves whether Telegram
        # accepted the attempt. The caller owns any authorized recovery policy.
        return {"success": False, "error": "TELEGRAM_SEND_UNCONFIRMED", "delivery_status": "unknown"}
