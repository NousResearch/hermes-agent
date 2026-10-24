"""Native full-file comparison cards, using the memory review's lossless renderer."""
import json

from tools.memory_prompt_text import render_cards


def card_record(record, file_index):
    review = record['skill_review']
    file = review['files'][file_index]
    before, after = file['before'], file['after']
    if file.get('binary'):
        before = f"Binary file · {file['bytes']:,} bytes\nSHA-256: {file['base_hash']}"
    action = 'add' if before is None else 'remove' if after is None else 'replace'
    return {**record, 'target_path': file['path'], 'payload': {
        'action': action, 'matched_entry': before or '', 'content': after or ''},
        'review_issue': None, 'space_plan': None, 'subject': 'file'}


def render_file(record, file_index):
    if not record.get('skill_review'):
        from tools.memory_discussion_draft import draft_cards
        return draft_cards({'explanation': '⚠ ' + record.get('review_issue', 'This proposal cannot be previewed.'),
            'payload': {'action': 'add', 'content': json.dumps(record['payload'], ensure_ascii=False, indent=2)}})
    cards = render_cards(card_record(record, file_index), limit=3450)
    count = len(record['skill_review']['files'])
    if count > 1:
        cards[-1].append(f'\n\nFile {file_index + 1} of {count}. These files apply together after each is approved. '
                         'Disapprove cancels this whole proposal.', 'italic')
    return cards


def buttons(pending_id, revision, file_index):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    actions = (('✅ Approve', 'approve'), ('🗑 Disapprove', 'reject'), ('⏰ Snooze', 'snooze'), ('✏️ Edit', 'edit'))
    items = [InlineKeyboardButton(label, callback_data=f'skl:{action}:{pending_id}:{revision}:{file_index}')
             for label, action in actions]
    return InlineKeyboardMarkup([items[:2], items[2:]])
