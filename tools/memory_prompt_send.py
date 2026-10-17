"""Immediate per-record memory review delivery, including from standalone workers."""

from __future__ import annotations

import logging
import re
import time
import threading
import hashlib
import json
from contextlib import contextmanager

from hermes_constants import get_hermes_home
from tools import write_approval as wa
from tools.memory_prompt_text import render_cards
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_held_locks = threading.local()


def review_buttons(pending_id: str, revision=1):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    actions = (("✅ Approve", "approve"), ("🗑 Disapprove", "reject"),
               ("⏰ Snooze", "snooze"), ("✏️ Edit", "edit"))
    buttons = [InlineKeyboardButton(label, callback_data=f"mem:{action}:{pending_id}:{revision}")
               for label, action in actions]
    return InlineKeyboardMarkup([buttons[:2], buttons[2:], [InlineKeyboardButton(
        '💬 Chat about it', callback_data=f'mem:discuss:{pending_id}:{revision}')]])


@contextmanager
def record_lock(pending_id: str):
    """Serialize delivery and user decisions across threads and gateway restarts."""
    if not re.fullmatch(r"[a-f0-9]{8}", pending_id):
        raise ValueError("Invalid memory review id")
    from hermes_cli.active_sessions import _FileLock

    path = get_hermes_home() / "pending" / ".memory-review-locks" / f"{pending_id}.lock"
    held = getattr(_held_locks, "paths", None)
    if held is None:
        held = _held_locks.paths = set()
    if path in held:
        yield
        return
    with _FileLock(path):
        held.add(path)
        try:
            yield
        finally:
            held.remove(path)


def _delivery_config():
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    home = config.get_home_channel(Platform.TELEGRAM)
    if home is None:
        logger.warning("Memory review has no Telegram binding in %s; delivery will retry", get_hermes_home())
        return None
    pconfig = config.platforms.get(Platform.TELEGRAM)
    # A shared-bot satellite's delivery resolver may supply the transport credential.
    from tools.send_message_senders import _live_adapter

    _, adapter = _live_adapter(Platform.TELEGRAM)
    if adapter is not None:
        pconfig = adapter.config
    if pconfig is None or not pconfig.token:
        logger.warning("Memory review has no Telegram transport in %s; delivery will retry", get_hermes_home())
        return None
    return pconfig, home


def _make_bot(pconfig):
    from telegram import Bot
    from tools.send_message_senders import _telegram_bot

    extra = pconfig.extra or {}
    if extra.get("base_url"):
        return Bot(token=pconfig.token, base_url=extra["base_url"],
                   base_file_url=extra.get("base_file_url", extra["base_url"]))
    return _telegram_bot(pconfig.token)


async def deliver_prompt(bot, record: dict, chat_id: str, thread_id=None, *, progress=None, quiet=False,
                         cards=None, buttons=None) -> dict:
    """One operation's contiguous native cards; only its decision card alerts audibly.

    Persist each known receipt before sending the next part. An accepted request whose
    response is lost remains ambiguous; the API has no idempotency key for sendMessage.
    """
    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
    from tools.send_message_senders import _telegram_thread_kwargs

    if cards is None:
        cards = render_cards(record, issue=record.get("review_issue"), limit=3700)
    signature = hashlib.sha256(json.dumps([(c.text, c.entities) for c in cards], ensure_ascii=False).encode()).hexdigest()
    previous = record.get("delivery_in_progress") or {}
    messages = list(previous.get("message_ids", [])) if (previous.get("signature") == signature
        and previous.get("chat_id") == str(chat_id) and previous.get("bot_id") == str(bot.id)
        and previous.get("thread_id") == (str(thread_id) if thread_id is not None else None)) else []
    receipt = {"chat_id": str(chat_id), "thread_id": str(thread_id) if thread_id is not None else None,
               "message_ids": messages, "bot_id": str(bot.id), "revision": record.get("revision", 1),
               "signature": signature, "card": {"text": cards[-1].text, "entities": cards[-1].entities}}
    for index in range(len(messages), len(cards)):
        card = cards[index]
        message = await bot.send_message(
            chat_id=normalize_telegram_chat_id(chat_id), text=card.text, entities=card.entities, parse_mode=None,
            disable_notification=quiet or index != len(cards) - 1, disable_web_page_preview=True,
            reply_markup=(buttons if buttons is not None else review_buttons(record["id"], record.get("revision", 1)))
                         if index == len(cards) - 1 else None,
            **_telegram_thread_kwargs(thread_id))
        messages.append(str(message.message_id))
        if progress:
            progress(receipt)
    return receipt


async def _deliver(record: dict, pconfig, home) -> dict:
    def progress(receipt):
        current = wa.get_pending(wa.MEMORY, record["id"])
        if current is not None and current.get("revision", 1) == record.get("revision", 1):
            current["delivery_in_progress"] = receipt
            atomic_json_write(wa._pending_path(wa.MEMORY, current["id"]), current)
    async with _make_bot(pconfig) as bot:
        return await deliver_prompt(bot, record, home.chat_id, home.thread_id, progress=progress)


@contextmanager
def delivery_lock():
    """Keep one profile's continuation groups contiguous across worker processes."""
    from hermes_cli.active_sessions import _FileLock
    with _FileLock(get_hermes_home() / "pending" / ".memory-delivery.lock"):
        yield


def send_prompt(subsystem: str, record: dict, *, due_before: float | None = None,
                nag_seconds: float = 7200, delivery=None) -> bool:
    """Synchronously attempt delivery; only a completed prompt advances the cadence clock.

    ``due_before`` rechecks cadence under the record lock, so two concurrent retries do not
    both send. No token or launch-profile fallback is used when an owning home has no binding.
    """
    if subsystem != wa.MEMORY:
        return False
    with delivery_lock(), record_lock(record["id"]):
        current = wa.get_pending(wa.MEMORY, record["id"])
        if current is None or current.get("payload", {}).get("action") == "batch" or current.get("conversion_pending"):
            return False
        if due_before is not None:
            snooze = current.get("snooze_until")
            if snooze is not None and float(snooze) > due_before:
                return False
            if snooze is None and current.get("last_prompt_at") is not None:
                if float(current["last_prompt_at"]) + nag_seconds > due_before:
                    return False
        if current.get("payload", {}).get("action") == "patch":
            from tools.memory_review_records import reviewed_payload
            current["payload"] = reviewed_payload(current["payload"])
            current["action"] = current["payload"]["action"]
            atomic_json_write(wa._pending_path(wa.MEMORY, current["id"]), current)
        delivery = delivery or _delivery_config()
        if delivery is None:
            return False
        from model_tools import _run_async
        from tools.memory_review_records import review_issue
        from tools.memory_tool import load_on_disk_store
        home_path = get_hermes_home()
        current["profile"] = home_path.name if home_path.parent.name == "profiles" else "default"
        store = load_on_disk_store()
        target_path = str(store._path_for(current.get("payload", {}).get("target", "memory")).absolute())
        current["target_path"] = target_path
        current["review_issue"] = review_issue(current, store)
        from tools.memory_space_plan import space_plan
        current["space_plan"] = space_plan(current, store)
        issue = current["review_issue"]
        plan = current["space_plan"]

        try:
            prompt = _run_async(_deliver(current, *delivery))
        except Exception:
            logger.debug("Memory review transport exception", exc_info=True)
            logger.warning("Memory review delivery failed for %s in %s; delivery will retry",
                           current["id"], get_hermes_home())
            return False
        # A synchronous consumer may have answered during an embedded delivery call.
        current = wa.get_pending(wa.MEMORY, record["id"])
        if current is None:
            return False
        current["review_issue"] = issue
        current["target_path"] = target_path
        current["space_plan"] = plan
        current.pop("delivery_in_progress", None)
        current["profile"] = home_path.name if home_path.parent.name == "profiles" else "default"
        current["prompt"] = prompt
        current.setdefault("prompts", []).append(prompt)
        current["last_prompt_at"] = time.time()
        current.pop("snooze_until", None)
        atomic_json_write(wa._pending_path(wa.MEMORY, current["id"]), current)
        return True
