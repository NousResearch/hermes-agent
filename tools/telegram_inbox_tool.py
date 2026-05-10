"""Telegram zero-inbox housekeeping tool.

This tool is intentionally narrower than a generic Telegram admin surface.  It
exists for forum-topic inbox workflows: copy the original message and/or create a
canonical archive/action card in a configured target topic, then optionally delete
the source Inbox message only after the target write succeeds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Iterable

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_ARCHIVE_MODES = {
    "card_only",
    "copy_original_and_card",
    "copy_original_card_then_delete",
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_GENERAL_TOPIC_THREAD_ID = "1"


def _parse_csv_ids(raw: Any) -> set[str]:
    """Return normalized string IDs from comma/list-like config values."""
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        values: Iterable[Any] = raw
    else:
        values = str(raw).split(",")
    return {str(v).strip() for v in values if str(v).strip()}


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _load_hermes_dotenv_best_effort() -> None:
    """Load $HERMES_HOME/.env if python-dotenv is available.

    Tool availability and calls should work in the gateway container where the
    token may live in the Hermes dotenv file rather than the process env.
    """
    try:
        from dotenv import load_dotenv
        from hermes_constants import get_hermes_home
    except Exception:
        return
    env_path = get_hermes_home() / ".env"
    try:
        load_dotenv(str(env_path), override=False, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(str(env_path), override=False, encoding="latin-1")
    except Exception:
        return


def _load_config_telegram_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}
    try:
        cfg = load_config()
    except Exception:
        return {}
    telegram_cfg = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    return telegram_cfg if isinstance(telegram_cfg, dict) else {}


def _allowed_chats_from_config(telegram_cfg: dict[str, Any]) -> set[str]:
    allowed = set()
    for key in ("allowed_chats", "group_allowed_chats"):
        allowed.update(_parse_csv_ids(telegram_cfg.get(key)))
    allowed.update(_parse_csv_ids(os.getenv("TELEGRAM_ALLOWED_CHATS")))
    allowed.update(_parse_csv_ids(os.getenv("TELEGRAM_GROUP_ALLOWED_CHATS")))
    return allowed


def _zero_inbox_policy(telegram_cfg: dict[str, Any]) -> dict[str, Any]:
    zero_cfg = telegram_cfg.get("zero_inbox", {})
    if not isinstance(zero_cfg, dict):
        zero_cfg = {}

    source_threads = set()
    target_threads = set()
    for key in ("source_thread_ids", "inbox_thread_ids"):
        source_threads.update(_parse_csv_ids(zero_cfg.get(key)))
    for key in ("target_thread_ids", "archive_thread_ids"):
        target_threads.update(_parse_csv_ids(zero_cfg.get(key)))

    source_threads.update(_parse_csv_ids(os.getenv("TELEGRAM_ZERO_INBOX_SOURCE_THREADS")))
    target_threads.update(_parse_csv_ids(os.getenv("TELEGRAM_ZERO_INBOX_TARGET_THREADS")))

    delete_enabled = bool(zero_cfg.get("delete_enabled", False)) or _bool_env(
        "TELEGRAM_ZERO_INBOX_DELETE_ENABLED",
        False,
    )
    reactions_enabled = bool(zero_cfg.get("reactions", True))
    if os.getenv("TELEGRAM_ZERO_INBOX_REACTIONS") is not None:
        reactions_enabled = _bool_env("TELEGRAM_ZERO_INBOX_REACTIONS", True)

    return {
        "source_threads": source_threads,
        "target_threads": target_threads,
        "delete_enabled": delete_enabled,
        "reactions_enabled": reactions_enabled,
    }


def _validate_int_id(name: str, value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric Telegram ID")


def _message_thread_kwargs(thread_id: str | None) -> dict[str, int]:
    if not thread_id or str(thread_id) == _GENERAL_TOPIC_THREAD_ID:
        return {}
    return {"message_thread_id": _validate_int_id("thread_id", thread_id)}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": redact_sensitive_text(message)}


async def _archive_telegram_inbox_item_async(
    *,
    bot: Any,
    source_chat_id: str,
    source_message_id: str,
    source_thread_id: str | None,
    target_chat_id: str | None,
    target_thread_id: str,
    archive_card: str,
    mode: str = "copy_original_and_card",
    reaction: str = "✅",
    telegram_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive/copy an Inbox item using an already-constructed Telegram bot."""
    if mode not in _ARCHIVE_MODES:
        return _error(f"mode must be one of {sorted(_ARCHIVE_MODES)}")
    if not archive_card or not archive_card.strip():
        return _error("archive_card is required")

    source_chat = _validate_int_id("source_chat_id", source_chat_id)
    target_chat_raw = target_chat_id or source_chat_id
    target_chat = _validate_int_id("target_chat_id", target_chat_raw)
    source_message = _validate_int_id("source_message_id", source_message_id)
    source_thread = str(source_thread_id).strip() if source_thread_id is not None else ""
    target_thread = str(target_thread_id).strip() if target_thread_id is not None else ""
    if not target_thread:
        return _error("target_thread_id is required")

    telegram_cfg = telegram_cfg if telegram_cfg is not None else _load_config_telegram_section()
    allowed_chats = _allowed_chats_from_config(telegram_cfg)
    if allowed_chats and (str(source_chat) not in allowed_chats or str(target_chat) not in allowed_chats):
        return _error("source_chat_id and target_chat_id must be in Telegram allowed chats")

    policy = _zero_inbox_policy(telegram_cfg)
    if policy["source_threads"] and source_thread not in policy["source_threads"]:
        return _error("source_thread_id is not allowed for zero-inbox archiving")
    if policy["target_threads"] and target_thread not in policy["target_threads"]:
        return _error("target_thread_id is not allowed for zero-inbox archiving")
    if mode == "copy_original_card_then_delete" and not policy["delete_enabled"]:
        return _error("delete mode is disabled; set telegram.zero_inbox.delete_enabled or TELEGRAM_ZERO_INBOX_DELETE_ENABLED=true")

    copied_message_id = None
    card_message_id = None
    deleted_source = False
    reacted_source = False
    warnings: list[str] = []

    try:
        if mode in {"copy_original_and_card", "copy_original_card_then_delete"}:
            copied = await bot.copy_message(
                chat_id=target_chat,
                from_chat_id=source_chat,
                message_id=source_message,
                **_message_thread_kwargs(target_thread),
            )
            copied_message_id = getattr(copied, "message_id", None)

        sent = await bot.send_message(
            chat_id=target_chat,
            text=archive_card,
            **_message_thread_kwargs(target_thread),
        )
        card_message_id = getattr(sent, "message_id", None)

        if mode == "copy_original_card_then_delete":
            deleted_source = bool(await bot.delete_message(chat_id=source_chat, message_id=source_message))
            if not deleted_source:
                warnings.append("archive card was created, but source delete failed")

        if reaction and policy["reactions_enabled"] and not deleted_source:
            try:
                await bot.set_message_reaction(
                    chat_id=source_chat,
                    message_id=source_message,
                    reaction=reaction,
                )
                reacted_source = True
            except Exception as exc:  # pragma: no cover - best-effort marker
                warnings.append(f"reaction failed: {redact_sensitive_text(str(exc))}")

        return {
            "success": True,
            "mode": mode,
            "source_chat_id": str(source_chat),
            "source_message_id": str(source_message),
            "source_thread_id": source_thread or None,
            "target_chat_id": str(target_chat),
            "target_thread_id": target_thread,
            "copied_message_id": copied_message_id,
            "card_message_id": card_message_id,
            "deleted_source": deleted_source,
            "reacted_source": reacted_source,
            "warnings": warnings,
        }
    except Exception as exc:
        logger.warning("Telegram zero-inbox archive failed: %s", redact_sensitive_text(str(exc)))
        return _error(str(exc))


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A sync tool handler can occasionally be invoked from a context that
    # already owns an event loop. Run the coroutine in a one-off thread with its
    # own loop instead of deadlocking or trying nested asyncio.run().
    import threading

    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # propagate to caller thread
            box["error"] = exc

    thread = threading.Thread(target=runner, name="telegram-inbox-tool", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def telegram_inbox_tool(args, **kw):
    """Handle telegram_inbox tool calls."""
    action = args.get("action", "archive")
    if action != "archive":
        return json.dumps(_error("only action='archive' is supported"))

    required = ["source_chat_id", "source_message_id", "target_thread_id", "archive_card"]
    missing = [field for field in required if not args.get(field)]
    if missing:
        return json.dumps(_error(f"missing required field(s): {', '.join(missing)}"))

    _load_hermes_dotenv_best_effort()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return json.dumps(_error("TELEGRAM_BOT_TOKEN is not configured"))

    try:
        from telegram import Bot
    except Exception as exc:
        return json.dumps(_error(f"python-telegram-bot is not available: {exc}"))

    bot = Bot(token=token)
    try:
        result = _run_coro(
            _archive_telegram_inbox_item_async(
                bot=bot,
                source_chat_id=args.get("source_chat_id"),
                source_message_id=args.get("source_message_id"),
                source_thread_id=args.get("source_thread_id"),
                target_chat_id=args.get("target_chat_id") or args.get("source_chat_id"),
                target_thread_id=args.get("target_thread_id"),
                archive_card=args.get("archive_card"),
                mode=args.get("mode", "copy_original_and_card"),
                reaction=args.get("reaction", "✅"),
            )
        )
    except Exception as exc:
        result = _error(str(exc))
    return json.dumps(result)


def check_requirements() -> bool:
    _load_hermes_dotenv_best_effort()
    return bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip())


TELEGRAM_INBOX_SCHEMA = {
    "name": "telegram_inbox",
    "description": (
        "Scoped Telegram zero-inbox housekeeping for forum-topic workflows. "
        "Copies the original message and/or posts an archive/action card into a "
        "configured target topic, then optionally deletes the source Inbox message "
        "only after the target write succeeds. Use only for configured allowed "
        "Telegram chats/topics; not a general Telegram admin tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["archive"], "default": "archive"},
            "source_chat_id": {"type": "string", "description": "Telegram source supergroup/chat ID."},
            "source_thread_id": {"type": "string", "description": "Source topic/thread ID, e.g. Inbox topic."},
            "source_message_id": {"type": "string", "description": "Telegram message ID to archive/mark/delete."},
            "target_chat_id": {"type": "string", "description": "Target chat ID; defaults to source_chat_id."},
            "target_thread_id": {"type": "string", "description": "Target topic/thread ID for the archive/action card."},
            "archive_card": {"type": "string", "description": "Canonical triage/archive/action card to post in the target topic."},
            "mode": {
                "type": "string",
                "enum": ["card_only", "copy_original_and_card", "copy_original_card_then_delete"],
                "default": "copy_original_and_card",
                "description": "Delete mode requires telegram.zero_inbox.delete_enabled or TELEGRAM_ZERO_INBOX_DELETE_ENABLED=true.",
            },
            "reaction": {"type": "string", "default": "✅", "description": "Best-effort reaction marker on the source message."},
        },
        "required": ["source_chat_id", "source_message_id", "target_thread_id", "archive_card"],
    },
}


from tools.registry import registry

registry.register(
    name="telegram_inbox",
    toolset="messaging",
    schema=TELEGRAM_INBOX_SCHEMA,
    handler=telegram_inbox_tool,
    check_fn=check_requirements,
    requires_env=["TELEGRAM_BOT_TOKEN"],
    description="Telegram zero-inbox archive/copy/delete helper",
    emoji="📥",
)
