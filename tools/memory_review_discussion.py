"""Read historical context and revise one pending memory during its owning conversation."""
import copy
import time

from tools import write_approval as wa
from tools.memory_prompt_send import record_lock, send_prompt
from tools.memory_tool_store import ENTRY_DELIMITER
from tools.memory_review_policy import DISCUSSION_GUIDANCE
from utils import atomic_json_write


def active_discussion(context):
    if not context or not context.get('session_id'):
        return None
    return next((r for r in wa.list_pending(wa.MEMORY)
                 if (d := r.get('discussion')) and d.get('active')
                 and d.get('revision') == r.get('revision', 1) and r.get('snooze_until', 0) <= time.time()
                 and (d.get('session_id') == context['session_id'] or
                      (not d.get('session_id') and d.get('initial_message_id') == context.get('message_id')))
                 and _same_actor(d, context)), None)


def _same_actor(discussion, context):
    return all(str(discussion.get(k) or '') == str(context.get(k) or '')
               for k in ('platform', 'user_id', 'chat_id', 'thread_id'))


def _budget(record, store):
    target = record['payload'].get('target', 'memory')
    entries = store._entries_for(target)
    old = record['payload'].get('matched_entry')
    other = list(entries)
    if old in other:
        other.remove(old)
    limit = store._char_limit(target)
    used = len(ENTRY_DELIMITER.join(entries))
    return {'limit': limit, 'current_characters': used,
            'maximum_draft_characters': max(0, limit - len(ENTRY_DELIMITER.join(other)) - (len(ENTRY_DELIMITER) if other else 0))}


def discussion_tool(action, pending_id, expected_revision, *, content=None, query=None,
                    remove_entry=False, explanation=None, context=None, session_db=None):
    if (not context or not context.get('session_id') or not context.get('user_id')
            or not isinstance(pending_id, str) or not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)):
        return {'success': False, 'error': 'Open Chat about it on the latest memory card first.'}
    try:
        with record_lock(pending_id):
            record = wa.get_pending(wa.MEMORY, pending_id)
            d = (record or {}).get('discussion') or {}
            if (not record or record.get('revision', 1) != expected_revision or not d.get('active')
                    or d.get('revision') != expected_revision or not _same_actor(d, context)
                    or (d.get('session_id') and d['session_id'] != context['session_id'])
                    or (not d.get('session_id') and d.get('initial_message_id') != context.get('message_id'))
                    or record.get('snooze_until', 0) > time.time()):
                return {'success': False, 'error': 'This discussion is stale or belongs to another conversation. Open the latest card.'}
            from tools.memory_tool import load_on_disk_store
            store = load_on_disk_store()
            # Refresh disk state before budget calculation. A changed original is a conflict.
            preview = store.preview_operation(record['payload'])
            if action in {'revise', 'suggest'} and not preview.get('success') and not preview.get('capacity'):
                return {'success': False, 'error': preview.get('error', 'The original memory changed.')}
            budget = _budget(record, store)
            if action == 'review':
                from tools.memory_review_history import retrieve_history
                history = retrieve_history(record, query=query, session_db=session_db,
                                           current_session_id=context['session_id'])
                d['session_id'] = context['session_id']
                d['history_checked_revision'] = expected_revision
                d['history_status'] = history['status']
                record['discussion'] = d
                atomic_json_write(wa._pending_path(wa.MEMORY, pending_id), record)
                return {'success': True, 'pending_id': pending_id, 'revision': expected_revision,
                        'target_path': str(store._path_for(record['payload'].get('target', 'memory')).absolute()),
                        'budget': budget, 'applicability_issue': preview.get('error'),
                        'memory_files': {target: {
                            'path': str(store._path_for(target).absolute()),
                            'entries': list(store._entries_for(target)),
                            'entry_delimiter': ENTRY_DELIMITER,
                            'current_characters': len(ENTRY_DELIMITER.join(store._entries_for(target))),
                            'limit': store._char_limit(target)}
                            for target in ('memory', 'user') if store.target_enabled(target)},
                        'proposal': record['payload'], 'provenance': record.get('provenance'), 'history': history,
                        'instruction': DISCUSSION_GUIDANCE}
            if d.get('history_checked_revision') != expected_revision:
                return {'success': False, 'error': 'Retrieve the original context and memory files with action=review before revising this proposal.'}
            if action == 'revise' and (not context.get('message_id') or context['message_id'] == d.get('initial_message_id')):
                return {'success': False, 'error': 'Chat about it requests a recommendation, not a revision. '
                        'Present your recommendation now; wait for the user to request a revised draft in a follow-up message.'}
            if not isinstance(remove_entry, bool):
                return {'success': False, 'error': 'remove_entry must be a boolean.'}
            payload = copy.deepcopy(record['payload'])
            if remove_entry:
                if payload['action'] == 'add' or not payload.get('matched_entry'):
                    return {'success': False, 'error': 'This is a new addition, not an existing entry. Recommend Disapprove instead.'}
                if content not in (None, ''):
                    return {'success': False, 'error': 'Provide remove_entry=true or replacement content, not both.'}
                payload['action'] = 'remove'
                payload.pop('content', None)
            elif not isinstance(content, str) or not content.strip():
                return {'success': False, 'error': 'Provide the complete proposed memory text.', 'budget': budget}
            else:
                payload['content'] = content.strip()
                if payload['action'] == 'remove':
                    payload['action'] = 'replace'
            result = store.preview_operation(payload)
            if not result.get('success'):
                return {**result, 'budget': budget, 'instruction': 'Keep the pending proposal unchanged. '
                        'Shorten the draft faithfully or discuss separate make-room changes, then retry.'}
            if action == 'suggest':
                from tools.memory_discussion_draft import register_draft
                return register_draft(record, payload, context, explanation)
            stage_revision(record, payload, context)
        return present_revision(record, budget)
    except ValueError as exc:
        return {'success': False, 'error': str(exc)}


def stage_revision(record, payload, context):
    """Pin a human-selected revision; caller holds the owning record lock."""
    record['payload'] = payload
    record['action'] = payload['action']
    record['revision'] = record.get('revision', 1) + 1
    record['draft_provenance'] = {k: v for k, v in context.items() if v}
    record['discussion'].update(revision=record['revision'], history_checked_revision=record['revision'])
    for field in ('prompt', 'last_prompt_at', 'delivery_in_progress', 'snooze_until', 'space_plan', 'discussion_draft'):
        record.pop(field, None)
    record['prompts'] = []
    atomic_json_write(wa._pending_path(wa.MEMORY, record['id']), record)


def present_revision(record, budget):
    delivered = send_prompt(wa.MEMORY, record)
    from tools.memory_space_refresh import refresh_space_reviews
    refresh_space_reviews()
    return {'success': True, 'staged': True, 'pending_id': record['id'], 'revision': record['revision'],
            'delivered': delivered, 'draft_characters': len(record['payload'].get('content', '')), 'budget': budget,
            'message': ('Revised review presented for approval. Confirm this in one short sentence. '
                if delivered else 'The revised draft is staged but Telegram delivery failed. Tell the user it could not be presented; '
                'do not claim the new review card is ready. Delivery will retry through the existing reminder mechanism. ') +
            'Memory has not changed. The card shows the full change; do not repeat the draft, budget arithmetic, '
            'approval boilerplate, internal IDs or revisions unless the user asks.'}
