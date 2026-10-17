"""Recoverable conversion of outstanding memory batches; never apply or deduplicate.

The journal retains original records and fixed child IDs. Prepared children cannot be
approved. After children_written is durable, missing children are decisions, not a reason
to recreate them. Old aggregate IDs are refused independently of journal progress.
"""
from __future__ import annotations

import json
from hermes_constants import get_hermes_home
from hermes_cli.active_sessions import _FileLock
from tools import write_approval as wa
from tools.memory_review_records import independent_records
from tools.memory_prompt_send import record_lock
from utils import atomic_json_write


def journal_dir():
    return get_hermes_home() / 'pending' / '.memory-conversions'


def convert_pending_batches():
    """Idempotent local transition. Transport is handled by the existing delivery cadence."""
    directory = journal_dir()
    with _FileLock(directory / '.lock'):
        existing = {path.stem: json.loads(path.read_text(encoding="utf-8-sig")) for path in directory.glob('*.json')}
        for parent in wa.list_pending(wa.MEMORY):
            if parent.get('payload', {}).get('action') != 'batch' or parent['id'] in existing:
                continue
            with record_lock(parent['id']):
                parent = wa.get_pending(wa.MEMORY, parent['id'])
                if parent is None:
                    continue
                children = independent_records(parent['payload'], summary=parent.get('summary', ''),
                    origin=parent.get('origin'), parent_id=parent['id'], created_at=parent.get('created_at'))
                for child in children:
                    child['conversion_pending'] = parent['id']
                    if 'snooze_until' in parent:
                        child['snooze_until'] = parent['snooze_until']
                journal = {'parent_id': parent['id'], 'state': 'prepared', 'original': parent,
                           'children': children, 'retired_prompts': []}
                atomic_json_write(directory / f"{parent['id']}.json", journal)
                existing[parent['id']] = journal
        for pid, journal in existing.items():
            _resume_conversion(directory, pid, journal)
        return list(existing.values())


def _resume_conversion(directory, pid, journal):
    path = directory / f'{pid}.json'
    with record_lock(pid):
        if journal['state'] == 'prepared':
            for child in journal['children']:
                # A prepared child cannot have been decided through an approval surface.
                if wa.get_pending(wa.MEMORY, child['id']) is None:
                    atomic_json_write(wa._pending_path(wa.MEMORY, child['id']), child)
            journal['state'] = 'children_written'
            atomic_json_write(path, journal)
        if journal['state'] == 'children_written':
            wa.discard_pending(wa.MEMORY, pid)
            if wa._pending_path(wa.MEMORY, pid).exists():
                raise OSError('Could not retire the original combined proposal; recovery will retry.')
            for child in journal['children']:
                with record_lock(child['id']):
                    current = wa.get_pending(wa.MEMORY, child['id'])
                    if current and current.pop('conversion_pending', None):
                        atomic_json_write(wa._pending_path(wa.MEMORY, child['id']), current)
            journal['state'] = 'complete'
            atomic_json_write(path, journal)


async def _retire(bot, journal, progress):
    """Preserve the old comparison, disable its aggregate controls, add a replacement notice."""
    from telegram import InlineKeyboardMarkup
    prompts = journal['original'].get('prompts') or [journal['original'].get('prompt') or {}]
    retired = journal['retired_prompts']
    for prompt in prompts:
        ids = prompt.get('message_ids') or []
        if not ids:
            continue
        key = f"{prompt.get('chat_id')}:{ids[-1]}"
        if key in retired:
            continue
        # Even a maximum-length old message can retain all text: only replace its keyboard.
        from telegram.error import BadRequest
        try:
            await bot.edit_message_reply_markup(chat_id=prompt['chat_id'], message_id=int(ids[-1]),
                                               reply_markup=InlineKeyboardMarkup([]))
        except BadRequest as exc:
            if 'message is not modified' not in str(exc).lower():
                raise
        from tools.send_message_senders import _telegram_thread_kwargs
        await bot.send_message(chat_id=prompt['chat_id'],
            text='↳ This combined review has been replaced by separate memory cards below. '
                 'Each card now affects one memory only. No memory has changed.', parse_mode=None,
            reply_to_message_id=int(ids[-1]), disable_notification=True,
            **_telegram_thread_kwargs(prompt.get('thread_id')))
        retired.append(key)
        progress(journal)


def retire_combined_prompts():
    from tools.memory_prompt_send import _delivery_config, _make_bot
    from model_tools import _run_async
    directory = journal_dir()
    with _FileLock(directory / '.lock'):
        delivery = None
        for path in directory.glob('*.json'):
            journal = json.loads(path.read_text(encoding="utf-8-sig"))
            if journal['state'] != 'complete':
                continue
            prompts = journal['original'].get('prompts') or [journal['original'].get('prompt') or {}]
            expected = {f"{p.get('chat_id')}:{p['message_ids'][-1]}" for p in prompts if p.get('message_ids')}
            if expected <= set(journal['retired_prompts']):
                continue
            delivery = delivery or _delivery_config()
            if not delivery:
                return
            async def run():
                async with _make_bot(delivery[0]) as bot:
                    await _retire(bot, journal, lambda value: atomic_json_write(path, value))
            _run_async(run())
