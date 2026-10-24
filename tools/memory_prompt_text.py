"""Lossless native Telegram review cards, with explicit entities and a word diff.

Memory content is data: never parse Markdown/HTML or execute embedded commands.
Offsets below are UTF-16 units, as required by Telegram's Bot API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re


def units(text):
    return len(text.encode('utf-16-le')) // 2


@dataclass
class ReviewMessage:
    text: str = ''
    entities: list[dict] = field(default_factory=list)

    def append(self, text, kind=None, spans=()):
        offset = units(self.text)
        self.text += text
        if kind and text:
            self.entities.append(dict(type=kind, offset=offset, length=units(text)))
        for start, end, style in spans:
            if end > start:
                self.entities.append(dict(type=style, offset=offset + units(text[:start]),
                                          length=units(text[start:end])))
        return self


def entry_key(record):
    payload = record.get('payload') or {}
    value = str(payload.get('matched_entry') or payload.get('old_text') or payload.get('content') or 'New memory')
    first = value.splitlines()[0].strip()
    title = re.split(r':\s|\s[—–]\s|\s*\(', first, maxsplit=1)[0]
    title = re.split(r'(?<=[.!?])\s', title, maxsplit=1)[0]
    if len(title) > 64:
        title = title[:65].rsplit(' ', 1)[0].rstrip(' ,;') + '…'
    if not title or title.startswith(('/', '~/', '{', '[')):
        return 'Saved memory'
    return title


def word_diff(old, new):
    """Character spans of every changed token; whitespace is preserved exactly."""
    a, b = re.findall(r'\s+|\w+|[^\w\s]', old), re.findall(r'\s+|\w+|[^\w\s]', new)
    def boundaries(tokens):
        result = [0]
        for token in tokens:
            result.append(result[-1] + len(token))
        return result
    ap, bp = boundaries(a), boundaries(b)
    removed, added = [], []
    for kind, i, j, k, l in SequenceMatcher(None, a, b, autojunk=max(len(a), len(b)) > 3000).get_opcodes():
        if kind in {'delete', 'replace'}:
            removed.append((ap[i], ap[j], 'strikethrough'))
        if kind in {'insert', 'replace'}:
            added.append((bp[k], bp[l], 'bold'))
    return removed, added


def split_for_platform(chunks, limit):
    """Lossless semantic segmentation, with a safe oversized-token fallback."""
    if limit <= 0:
        raise ValueError('limit must be positive')
    text = ''.join(chunks)
    result = []
    while units(text) > limit:
        end, size = 0, 0
        for char in text:
            width = units(char)
            if size + width > limit:
                break
            size += width
            end += 1
        if end == 0:
            raise ValueError('limit cannot hold a Unicode character')
        prefix = text[:end]
        # Prefer paragraphs, then sentences, then words; don't waste most of a part.
        candidates = (list(re.finditer(r'\n\s*\n', prefix)),
                      list(re.finditer(r'[.!?](?:\s+|$)', prefix)),
                      list(re.finditer(r'\s+', prefix)))
        for matches in candidates:
            if matches and matches[-1].end() >= end // 2:
                end = matches[-1].end()
                break
        result.append(text[:end])
        text = text[end:]
    if text or not result:
        result.append(text)
    return result


def _quote(card, label, text, spans, empty, *, part=None):
    heading = label
    if part:
        heading += f' · {part[0]}/{part[1]}'
    card.append(heading + '\n', 'bold')
    if text:
        card.append(text, 'blockquote', spans)
    else:
        card.append(empty, 'italic')
    card.append('\n\n')


def _footer(card, status=None, issue=None, plan=None):
    if plan:
        card.append(f"This change would use {plan['count']:,} characters; the file’s limit is {plan['limit']:,} "
                    f"({plan['over']:,} over).\n\n")
        if plan['steps']:
            card.append('To make room, review these separate proposals:\n', 'bold')
            for step in plan['steps']:
                verb = 'Replace' if step['action'] == 'replace' else 'Delete'
                card.append(f"{verb} “{step['label']}” — frees {step['saved_chars']:,} characters.\n")
            card.append(f"\nAfter those changes, this proposal would fit at {plan['after']:,} / {plan['limit']:,} characters.\n\n")
        else:
            card.append('A space-saving proposal is still needed for this file.\n\n')
    elif issue:
        card.append('⚠ ' + issue + '\n\n', 'italic')
    if status:
        card.append(status, 'bold')
    return card


def render_cards(record, *, status=None, issue=None, limit=4096):
    """One independent review, optionally preceded by complete-text continuation parts."""
    payload = record.get('payload') or {}
    if issue is None:
        issue = record.get('review_issue')
    plan = record.get('space_plan')
    if payload.get('action') == 'batch':
        raise ValueError('Combined memory proposals must be converted before rendering.')
    action = payload.get('action')
    subject = record.get('subject', 'memory')
    old = str(payload.get('matched_entry') or payload.get('old_text') or '') if action != 'add' else ''
    new = str(payload.get('content') if payload.get('content') is not None else payload.get('new_text') or '') if action != 'remove' else ''
    removed, added = word_diff(old, new)
    profile = record.get('profile', 'default')
    target = record.get('target_path') or ('USER.md' if payload.get('target') == 'user' else 'MEMORY.md')
    context = f'{profile} wants to modify {target}'
    if action != 'replace':
        # On adds/deletes the entire entry changes: bolding every word adds no
        # information and makes the complete source harder to read.
        removed, added = [], []
    def head():
        return ReviewMessage().append(context + '\n\n', 'italic')
    full = head()
    if action != 'add':
        _quote(full, 'Delete' if action == 'remove' else 'Current', old, removed, '')
    if action != 'remove':
        _quote(full, f'New {subject}' if action == 'add' else 'Replacement', new, added, '')
    _footer(full, status, issue, plan)
    if units(full.text) <= limit:
        return [full]
    messages = []
    for label, source, spans in (('Delete' if action == 'remove' else 'Current', old, removed), (f'New {subject}' if action == 'add' else 'Replacement', new, added)):
        if not source:
            continue
        reserve = units(context) + 130
        parts = split_for_platform([source], limit - reserve)
        position = 0
        for index, part in enumerate(parts, 1):
            message = head()
            clipped = [(max(a, position) - position, min(b, position + len(part)) - position, style)
                       for a, b, style in spans if a < position + len(part) and b > position]
            _quote(message, label, part, clipped, '', part=(index, len(parts)))
            messages.append(message)
            position += len(part)
    compact = head()
    compact.append({'add': f'The new {subject} is in the messages directly above.',
                    'remove': f'The {subject} to delete is in the messages directly above.'}.get(
                        action, 'Current and Replacement are in the messages directly above.') + '\n\n')
    messages.append(_footer(compact, status, issue, plan))
    if any(units(m.text) > limit for m in messages):
        raise ValueError('Review metadata exceeds the Telegram message limit.')
    return messages


def render_prompt(record):
    """Compatibility for plain-text review consumers; transport uses render_cards."""
    return [message.text for message in render_cards(record)]
