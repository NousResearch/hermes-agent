"""Transport-independent skill review: freeze, inspect, revise and decide exact drafts."""
import copy
import time
from contextlib import contextmanager

from hermes_constants import get_hermes_home
from tools import write_approval as wa
from tools.skill_review_projection import check_sources, fingerprint, project, selected_payload, validate_text
from utils import atomic_json_write


@contextmanager
def review_lock(pending_id):
    import re
    from tools.skill_usage import skill_file_lock
    if not isinstance(pending_id, str) or not re.fullmatch('[a-f0-9]{8}', pending_id):
        raise ValueError('Invalid skill review id.')
    with skill_file_lock(get_hermes_home() / 'pending' / '.skill-review-locks' / f'{pending_id}.lock'):
        yield


def save(record):
    atomic_json_write(wa._pending_path(wa.SKILLS, record['id']), record)


def prepare(record):
    """Prepare only once; legacy proposals freeze when explicitly opened, never silently rebase."""
    if 'skill_review' not in record and 'review_issue' not in record:
        try:
            record['skill_review'] = project(record['payload'])
        except (ValueError, OSError, TypeError, KeyError) as exc:
            record['review_issue'] = str(exc)
        record.setdefault('revision', 1)
        save(record)
    return record


def inspect(pending_id):
    with review_lock(pending_id):
        record = wa.get_pending(wa.SKILLS, pending_id)
        return copy.deepcopy(prepare(record)) if record else None


def validate(record):
    review = record.get('skill_review')
    if not review:
        raise ValueError(record.get('review_issue') or 'Open this skill proposal for review first.')
    if review['payload_hash'] != fingerprint(record['payload']):
        raise ValueError('The proposal changed after its preview. Recreate this proposal.')
    check_sources(review)


def revise(pending_id, revision, file_index, text):
    """A human edit invalidates all file approvals and receipts, never writes skill content."""
    with review_lock(pending_id):
        record = wa.get_pending(wa.SKILLS, pending_id)
        if not record or record.get('revision', 1) != revision:
            raise ValueError('This proposal changed. Open the latest review.')
        validate(record)
        review = record['skill_review']
        if not 0 <= file_index < len(review['files']):
            raise ValueError('Invalid file review.')
        file = review['files'][file_index]
        if file.get('delete_package'):
            raise ValueError('This proposal deletes the whole skill. Disapprove it to keep the skill and request a replacement proposal.')
        validate_text(file, text, new_skill=file['name'] in review['creates'] and file['relative_path'] == 'SKILL.md')
        review['edits'][str(file_index)] = text
        file['after'] = text
        review['approved'] = []
        record['revision'] = revision + 1
        for key in ('skill_prompts', 'skill_delivery_in_progress', 'last_prompt_at', 'snooze_until'):
            record.pop(key, None)
        save(record)
        return copy.deepcopy(record)


def decide(pending_id, revision, file_index, action):
    """One file per human approval; dependent files apply together through the existing engine."""
    from tools import skill_manager_tool as sm
    with review_lock(pending_id):
        record = wa.get_pending(wa.SKILLS, pending_id)
        if not record or record.get('revision', 1) != revision:
            raise ValueError('This proposal changed or was already handled.')
        if record.get('snooze_until', 0) > time.time():
            raise ValueError('This proposal is snoozed. Open its latest review when due.')
        if action == 'reject':
            wa.discard_pending(wa.SKILLS, pending_id)
            return {'record': record, 'status': '🗑 Disapproved · Skill unchanged', 'finished': True}
        if action != 'approve':
            raise ValueError('Unknown skill review decision.')
        validate(record)
        review = record['skill_review']
        if not 0 <= file_index < len(review['files']):
            raise ValueError('Invalid file review.')
        if file_index in review['approved']:
            return {'record': record, 'status': '✅ Approved · Waiting for the other files', 'finished': False}
        with sm._skill_mutation_locks(review['roots']):
            validate(record)
            review['approved'].append(file_index)
            if len(review['approved']) < len(review['files']):
                save(record)
                return {'record': record, 'status': '✅ Approved · Waiting for the other files', 'finished': False}
            import json
            result = json.loads(sm.apply_skill_pending(selected_payload(record)))
            if not result.get('success'):
                review['approved'].remove(file_index)
                save(record)
                raise ValueError(result.get('error', 'The skill change could not be applied.'))
            wa.discard_pending(wa.SKILLS, pending_id)
            return {'record': record, 'status': '✅ Approved · Skill changes saved', 'finished': True}


def apply_reviewed(record):
    """CLI approval explicitly covers every file; share validation and the decision lock."""
    from tools import skill_manager_tool as sm
    review = record.get('skill_review')
    if not review:
        if record.get('review_issue'):
            raise ValueError(record['review_issue'])
        return sm.apply_skill_pending(record['payload'])
    with sm._skill_mutation_locks(review['roots']):
        validate(record)
        return sm.apply_skill_pending(selected_payload(record))
