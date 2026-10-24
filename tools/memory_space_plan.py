"""Explain capacity using real, independently reviewed changes in the same file."""
from __future__ import annotations

import time

from tools import write_approval as wa
from tools.memory_tool_store import ENTRY_DELIMITER


def space_plan(record, store, candidates=None, *, now=None):
    payload = record['payload']
    target = payload.get('target', 'memory')
    if payload.get('action') not in {'add', 'replace'}:
        return None
    # Validate syntax/pinning before considering capacity. The preview only reads.
    result = store.preview_operation(payload)
    if result.get('success') or result.get('capacity') is None:
        return None
    capacity = result['capacity']
    entries = list(store._entries_for(target))
    working = list(entries)
    now = time.time() if now is None else now
    eligible = []
    for candidate in wa.list_pending(wa.MEMORY) if candidates is None else candidates:
        op = candidate.get('payload', {})
        if (candidate['id'] == record['id'] or candidate.get('subsystem', wa.MEMORY) != wa.MEMORY
                or op.get('target', 'memory') != target
                or op.get('action') not in {'replace', 'remove'} or candidate.get('conversion_pending')
                or float(candidate.get('snooze_until') or 0) > now
                or not op.get('matched_entry')
                or op.get('matched_entry') == payload.get('matched_entry')):
            continue
        if not store.preview_operation(op).get('success'):
            continue
        simulated = list(entries)
        error, _, _, _ = store._apply_batch_op(simulated, op['action'], (op.get('content') or '').strip(),
            op.get('old_text', ''), 'Space-saving change', op['matched_entry'])
        saved = len(ENTRY_DELIMITER.join(entries)) - len(ENTRY_DELIMITER.join(simulated))
        if error is None and saved > 0:
            eligible.append((saved, candidate))
    # Fewer companion decisions; stable ordering makes unchanged cards quiet.
    eligible.sort(key=lambda item: (-item[0], item[1]['id']))
    steps = []
    for _, candidate in eligible:
        op = candidate['payload']
        simulated = list(working)
        error, _, _, _ = store._apply_batch_op(simulated, op['action'], (op.get('content') or '').strip(),
            op.get('old_text', ''), 'Space-saving change', op['matched_entry'])
        if error:
            continue # Two competing proposals for one entry cannot both free space.
        saved = len(ENTRY_DELIMITER.join(working)) - len(ENTRY_DELIMITER.join(simulated))
        if saved <= 0:
            continue
        final = list(simulated)
        error, _, _, _ = store._apply_batch_op(final, payload['action'], (payload.get('content') or '').strip(),
            payload.get('old_text', ''), 'Requested change', payload.get('matched_entry'))
        if error:
            continue
        from tools.memory_prompt_text import entry_key
        steps.append({'id': candidate['id'], 'revision': candidate.get('revision', 1),
                      'action': op['action'], 'label': entry_key(candidate), 'saved_chars': saved})
        working = simulated
        final_count = len(ENTRY_DELIMITER.join(final))
        if final_count <= capacity['limit']:
            return {**capacity, 'steps': steps, 'after': final_count}
    # Do not present a partial or conflicting set as a solution.
    return {**capacity, 'steps': [], 'after': None}


def producer_recovery(records, store):
    """Tell the existing memory-tool producer when it must propose make-room edits."""
    requests = []
    candidates = wa.list_pending(wa.MEMORY)
    for record in records:
        plan = space_plan(record, store, candidates)
        if not plan:
            continue
        target = record['payload'].get('target', 'memory')
        item = {'pending_id': record['id'], 'target': target,
                'target_path': str(store._path_for(target).absolute()),
                'characters_needed': plan['over'], 'limit': plan['limit'],
                'companion_ids': [s['id'] for s in plan['steps']]}
        if not plan['steps']:
            item['current_entries'] = list(store._entries_for(target))
            item['instruction'] = (
                'This requested change is already staged. Do not restage it. Propose concise replacements '
                'of OTHER existing entries in this file to free the required space, preserving their facts '
                'and instructions. Stage each companion for independent full-text approval; never apply '
                'or delete information silently. If no faithful consolidation is possible, explain that.')
        requests.append(item)
    return requests
