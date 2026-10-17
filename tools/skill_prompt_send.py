"""Immediate skill-file prompts with the existing Telegram delivery and durable retry plumbing."""
import logging
import time

from hermes_constants import get_hermes_home
from tools import write_approval as wa
from tools.memory_prompt_send import _delivery_config, _make_bot, deliver_prompt, delivery_lock
from tools.skill_prompt_text import buttons, render_file
from tools.skill_review import prepare, review_lock, save

logger = logging.getLogger(__name__)


def send_prompt(record, *, delivery=None, due_before=None, nag_seconds=7200):
    from model_tools import _run_async
    with delivery_lock(), review_lock(record['id']):
        current = wa.get_pending(wa.SKILLS, record['id'])
        if not current or current.get('snooze_until', 0) > (time.time() if due_before is None else due_before):
            return False
        if (due_before is not None and not current.get('snooze_until')
                and current.get('last_prompt_at', 0) + nag_seconds > due_before):
            return False
        prepare(current)
        home = get_hermes_home()
        current['profile'] = home.name if home.parent.name == 'profiles' else 'default'
        try:
            delivery = delivery or _delivery_config()
            if delivery is None:
                return False
            review = current.get('skill_review') or {'files': [None], 'approved': []}
            revision = current.get('revision', 1)
            async def deliver():
                async with _make_bot(delivery[0]) as bot:
                    for index in range(len(review['files'])):
                        if index in review['approved']:
                            continue
                        key = str(index)
                        progress_map = current.setdefault('skill_delivery_in_progress', {})
                        # A completed group in an interrupted delivery is not sent twice.
                        previous = progress_map.get(key) or {}
                        if (previous.get('complete') and previous.get('revision') == revision
                                and previous.get('chat_id') == str(delivery[1].chat_id)
                                and previous.get('thread_id') == (str(delivery[1].thread_id) if delivery[1].thread_id is not None else None)
                                and previous.get('bot_id') == str(bot.id)):
                            continue
                        def progress(receipt):
                            progress_map[key] = receipt
                            save(current)
                        cards = render_file(current, index)
                        receipt = await deliver_prompt(bot,
                            {**current, 'delivery_in_progress': previous}, delivery[1].chat_id, delivery[1].thread_id,
                            progress=progress, cards=cards, buttons=buttons(current['id'], revision, index))
                        current.setdefault('skill_prompts', {}).setdefault(key, []).append(receipt)
                        progress_map[key] = {**receipt, 'complete': True}
                        save(current)
            _run_async(deliver())
        except Exception:
            logger.warning('Skill review delivery failed for %s; delivery will retry', current['id'], exc_info=True)
            return False
        current.pop('skill_delivery_in_progress', None)
        current.pop('snooze_until', None)
        current['last_prompt_at'] = time.time()
        save(current)
        return True


def retry_prompts(*, now, nag_seconds):
    """Retry new reviews only. Legacy records are prepared by explicit /skills opening."""
    sent = 0
    for record in wa.list_pending(wa.SKILLS):
        if 'revision' not in record:
            continue
        try:
            sent += bool(send_prompt(record, due_before=now, nag_seconds=nag_seconds))
        except Exception:
            logger.warning('Skill review retry failed for %s', record['id'], exc_info=True)
    return sent
