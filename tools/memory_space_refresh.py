"""Keep make-room plans current on existing Telegram reviews, without notifications."""
from __future__ import annotations

import logging
import time

from tools import write_approval as wa
from utils import atomic_json_write

logger = logging.getLogger(__name__)


def refresh_space_reviews():
    from tools.memory_prompt_send import _delivery_config, _make_bot, delivery_lock, record_lock, review_buttons, deliver_prompt
    from tools.memory_prompt_text import render_cards
    from tools.memory_review_records import review_issue
    from tools.memory_space_plan import space_plan
    from tools.memory_tool import load_on_disk_store
    from model_tools import _run_async

    async def refresh():
        delivery = _delivery_config()
        if delivery is None:
            return 0
        changed = 0
        async with _make_bot(delivery[0]) as bot:
            for pending in wa.list_pending(wa.MEMORY):
                with record_lock(pending['id']):
                    record = wa.get_pending(wa.MEMORY, pending['id'])
                    if (not record or record.get('conversion_pending') or record['payload']['action'] == 'batch'
                            or float(record.get('snooze_until') or 0) > time.time()):
                        continue
                    receipt = record.get('prompt') or {}
                    if (not receipt.get('message_ids') or receipt.get('bot_id') != str(bot.id)
                            or receipt.get('revision', 1) != record.get('revision', 1)):
                        continue
                    store = load_on_disk_store()
                    plan = space_plan(record, store)
                    if not plan and not record.get('space_plan'):
                        continue
                    record['space_plan'] = plan
                    record['review_issue'] = review_issue(record, store)
                    cards = render_cards(record, limit=3700)
                    card = {'text': cards[-1].text, 'entities': cards[-1].entities}
                    if card == receipt.get('card'):
                        # References can change without changing their displayed labels.
                        if record != pending:
                            atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)
                        continue
                    if len(cards) > 1 and len(receipt['message_ids']) == 1:
                        # A larger explanation needs continuations. Preserve all source text,
                        # send the new group silently, and retain its normal revision binding.
                        def progress(value):
                            record['delivery_in_progress'] = value
                            atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)
                        receipt = await deliver_prompt(bot, record, receipt['chat_id'], receipt.get('thread_id'),
                                                       progress=progress, quiet=True)
                        record['prompt'] = receipt
                        record.setdefault('prompts', []).append(receipt)
                        record.pop('delivery_in_progress', None)
                    else:
                        from telegram.error import BadRequest
                        try:
                            await bot.edit_message_text(chat_id=receipt['chat_id'], message_id=int(receipt['message_ids'][-1]),
                                text=card['text'], entities=card['entities'], parse_mode=None, disable_web_page_preview=True,
                                reply_markup=review_buttons(record['id'], record.get('revision', 1)))
                        except BadRequest as exc:
                            if 'message is not modified' not in str(exc).lower():
                                raise
                        receipt['card'] = card
                        for old in record.get('prompts', []):
                            if old.get('message_ids') == receipt['message_ids'] and old.get('bot_id') == receipt['bot_id']:
                                old['card'] = card
                    atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)
                    changed += 1
        return changed

    if not wa.list_pending(wa.MEMORY):
        return 0
    try:
        with delivery_lock():
            return _run_async(refresh())
    except Exception:
        logger.warning('Could not refresh memory make-room plans; the existing cadence will retry', exc_info=True)
        return 0


def refresh_in_home(home):
    from plugins.platforms.telegram.memory_review import _home_scope
    with _home_scope(home):
        return refresh_space_reviews()
