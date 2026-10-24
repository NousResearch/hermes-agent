"""Freeze an agent's recommendation and deliver its native Telegram selection button."""
import copy
import logging
import time
import uuid

from tools import write_approval as wa
from tools.memory_prompt_send import _delivery_config, _make_bot, delivery_lock, record_lock
from tools.memory_prompt_text import ReviewMessage, split_for_platform, units
from utils import atomic_json_write

logger = logging.getLogger(__name__)


def register_draft(record, payload, context, explanation):
    if explanation is not None and not isinstance(explanation, str):
        return {'success': False, 'error': 'explanation must be a short string.'}
    record['discussion_draft'] = {'token': uuid.uuid4().hex[:12], 'revision': record.get('revision', 1),
        'payload': copy.deepcopy(payload), 'context': dict(context), 'explanation': (explanation or '').strip()}
    atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)
    return {'success': True, 'pending_id': record['id'], 'draft_token': record['discussion_draft']['token']}


def draft_cards(draft):
    card = ReviewMessage()
    if draft['explanation']:
        card.append(draft['explanation'] + '\n\n')
    payload = draft['payload']
    if payload['action'] == 'remove':
        card.append('Delete\n', 'bold').append(payload['matched_entry'], 'blockquote')
    else:
        card.append(payload['content'], 'blockquote')
    # Clip native entities without parsing the agent's text as markup or truncating it.
    result, offset = [], 0
    for part in split_for_platform([card.text], 3700):
        end = offset + units(part)
        entities = [{**e, 'offset': max(e['offset'], offset) - offset,
                     'length': min(e['offset'] + e['length'], end) - max(e['offset'], offset)}
                    for e in card.entities if e['offset'] < end and e['offset'] + e['length'] > offset]
        result.append(ReviewMessage(part, entities))
        offset = end
    return result


async def _deliver(record, draft, pconfig):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
    from tools.send_message_senders import _telegram_thread_kwargs

    d = record['discussion']
    receipt = {'revision': draft['revision'], 'chat_id': d['chat_id'], 'thread_id': d['thread_id'] or None,
               'message_ids': []}
    cards = draft_cards(draft)
    async with _make_bot(pconfig) as bot:
        receipt['bot_id'] = str(bot.id)
        for index, card in enumerate(cards):
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('Use this draft',
                callback_data=f"mem:use:{record['id']}:{draft['revision']}:{draft['token']}")]])
            message = await bot.send_message(chat_id=normalize_telegram_chat_id(d['chat_id']),
                text=card.text, entities=card.entities, parse_mode=None,
                disable_notification=index != len(cards) - 1, disable_web_page_preview=True,
                reply_to_message_id=int(d['initial_message_id']), allow_sending_without_reply=True,
                reply_markup=keyboard if index == len(cards) - 1 else None,
                **_telegram_thread_kwargs(d['thread_id'] or None))
            receipt['message_ids'].append(str(message.message_id))
        return receipt


def send_draft(result):
    """Transport runs after the tool releases its record lock; preserve lock ordering."""
    from model_tools import _run_async
    from tools.memory_review_discussion import _same_actor
    with delivery_lock(), record_lock(result['pending_id']):
        record = wa.get_pending(wa.MEMORY, result['pending_id'])
        draft = (record or {}).get('discussion_draft') or {}
        d = (record or {}).get('discussion') or {}
        context = draft.get('context') or {}
        revision = (record or {}).get('revision', 1)
        # A human can edit, snooze, or switch discussions after registration and
        # before transport acquires the locks. Never present that abandoned draft.
        if (not record or draft.get('token') != result['draft_token']
                or draft.get('revision') != revision or d.get('revision') != revision
                or d.get('history_checked_revision') != revision or not d.get('active')
                or not context.get('session_id') or d.get('session_id') != context['session_id']
                or not _same_actor(d, context) or record.get('snooze_until', 0) > time.time()):
            return {'success': False, 'error': 'The discussion changed before the draft could be presented.'}
        if not draft.get('receipt'):
            try:
                delivery = _delivery_config()
                if delivery is None:
                    raise ValueError('No Telegram transport')
                receipt = _run_async(_deliver(record, draft, delivery[0]))
            except Exception:
                logger.warning('Discussion draft delivery failed for %s', record['id'], exc_info=True)
                return {**result, 'delivered': False, 'message': 'Draft could not be presented. '
                    'Show the full recommendation in chat and explain that its button could not be delivered. '
                    'The pending review and memory are unchanged. Retry action=suggest to present the button.'}
            draft['receipt'] = receipt
            atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)
        return {**result, 'delivered': True, 'message': 'Your exact recommendation and draft were delivered '
            'with Use this draft. The pending review and memory are unchanged. End this turn with exactly '
            'NO_REPLY so the already visible recommendation is not repeated.'}
