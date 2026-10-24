"""Memory-only approval normalization and non-mutating applicability checks."""
from __future__ import annotations

import copy
import time
import uuid

from tools import write_approval as wa


def reviewed_payload(payload):
    """Promote a pinned upstream span patch to the complete entry shown for approval."""
    op = copy.deepcopy(payload)
    if op.get('action') == 'patch':
        from tools.memory_tool_store import _patch_entry
        if not op.get('matched_entry'):
            raise ValueError('This patch predates entry pinning. Recreate it before approving.')
        after, error, _ = _patch_entry(op['matched_entry'], op.get('old_text') or '',
                                     op.get('content') or op.get('new_text') or '')
        if after is None:
            raise ValueError(error)
        op.update(action='replace', content=after, old_text=op['matched_entry'], source_action='patch')
    return op


def independent_records(payload, *, summary, origin, parent_id=None, created_at=None):
    """Preserve each operation and its sequential pin; never create an aggregate decision."""
    operations = payload.get('operations') if payload.get('action') == 'batch' else [payload]
    if not isinstance(operations, list) or not operations:
        raise ValueError('A memory proposal must contain at least one operation.')
    group = parent_id or uuid.uuid4().hex
    records, produced = [], {}
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get('action') not in {'add', 'replace', 'remove', 'patch'}:
            raise ValueError('Memory review supports add, patch, replace, or remove operations only.')
        op = reviewed_payload(operation)
        op['target'] = payload.get('target', 'memory')
        if op['target'] not in {'memory', 'user'}:
            raise ValueError('Invalid memory target.')
        if op['action'] in {'add', 'replace'} and not (op.get('content') or op.get('new_text') or '').strip():
            raise ValueError('A replacement or addition cannot be empty.')
        if op['action'] in {'replace', 'remove'} and not (op.get('old_text') or op.get('matched_entry')):
            raise ValueError('A replacement or deletion must identify its original entry.')
        if not op.get('old_text') and op.get('matched_entry'):
            op['old_text'] = op['matched_entry']
        if 'content' not in op and 'new_text' in op:
            op['content'] = op['new_text']
        pid = uuid.uuid4().hex[:8]
        record = dict(id=pid, subsystem=wa.MEMORY, action=op['action'], payload=op,
                      revision=1, summary=(summary or '').strip(), origin=origin or 'foreground',
                      created_at=created_at if created_at is not None else time.time(),
                      source_batch=group, source_index=index)
        pin = op.get('matched_entry')
        if pin in produced:
            prerequisite = produced[pin]
            record['depends_on'] = [prerequisite['id']]
            from tools.memory_prompt_text import entry_key
            record['prerequisite_title'] = entry_key(prerequisite)
        if op['action'] in {'add', 'replace'}:
            produced[(op.get('content') or op.get('new_text')).strip()] = record
        from tools.memory_review_context import current_source
        if provenance := current_source():
            record['provenance'] = provenance
        records.append(record)
    return records


def stage_memory(payload, *, summary, origin, force=False):
    from tools.memory_rejections import was_rejected
    # Retain exact historical batch rejection suppression before splitting.
    if not force and was_rejected(payload):
        return {'skipped': True, 'reason': 'previously disapproved', 'pending_ids': []}
    records = independent_records(payload, summary=summary, origin=origin)
    staged, suppressed, errors = [], [], []
    for record in records:
        if not force and was_rejected(record['payload']):
            suppressed.append({'source_index': record['source_index'], 'reason': 'previously disapproved'})
            continue
        try:
            wa.atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)
        except OSError as exc:
            errors.append({'source_index': record['source_index'], 'error': str(exc)})
            continue
        staged.append(record)
        wa._prompt_for_record(wa.MEMORY, record)
    if staged:
        from tools.memory_space_refresh import refresh_space_reviews
        refresh_space_reviews()
    if payload.get('action') != 'batch' and staged:
        return staged[0]
    return {'records': staged, 'pending_ids': [r['id'] for r in staged],
            'suppressed': suppressed, 'errors': errors, 'skipped': not staged and bool(suppressed) and not errors}


def stage_result(record, message, **extra):
    """Tool result: partial disk failures and suppression must not invent successful IDs."""
    ids = record.get('pending_ids', [record['id']] if 'id' in record else [])
    result = dict(success=not bool(record.get('errors')), pending_ids=ids, message=message, **extra)
    if ids:
        result['staged'] = True
        if len(ids) == 1 and 'id' in record:
            result['pending_id'] = ids[0]
    if record.get('suppressed'):
        result['suppressed_operations'] = record['suppressed']
    if record.get('errors'):
        result['errors'] = record['errors']
    if ids:
        from tools.memory_space_plan import producer_recovery
        from tools.memory_tool import load_on_disk_store
        records = [r for pid in ids if (r := wa.get_pending(wa.MEMORY, pid)) is not None]
        if recovery := producer_recovery(records, load_on_disk_store()):
            result['capacity_recovery'] = recovery
    return result


def review_issue(record, store):
    """Read the live store without writing; approval repeats these checks under its lock."""
    payload = record['payload']
    if record.get('conversion_pending'):
        return 'This proposal is being recovered for independent review. Please wait for its new card.'
    if payload.get('action') == 'batch':
        return 'This combined proposal must be reviewed individually after conversion.'
    result = store.preview_operation(payload)
    if result.get('success'):
        return None
    if record.get('depends_on') and payload.get('matched_entry') not in store._entries_for(payload.get('target', 'memory')):
        if any(wa.get_pending(wa.MEMORY, pid) for pid in record['depends_on']):
            title = record.get('prerequisite_title', 'the prerequisite memory')
            return f'Waiting for an earlier proposal: “{title}”. Approve it to create the exact original shown below. This action never applies it for you.'
        return 'The earlier proposal is no longer outstanding and this exact original does not exist. Recreate this change against the current memory; nothing has been applied.'
    if payload.get('action') in {'replace', 'remove'} and payload.get('matched_entry') and payload.get('matched_entry') not in store._entries_for(payload.get('target', 'memory')):
        return 'The saved original has changed since this proposal was staged. Nothing can be overwritten; recreate the proposal against the current entry.'
    error = result.get('error', 'This change cannot currently be applied.')
    return error if len(error) <= 350 else error[:350].rsplit(' ', 1)[0] + '…'
