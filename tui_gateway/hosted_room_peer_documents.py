"""Source-authorized document identity and lazy byte fulfillment for peer turns."""
import base64
import hashlib
import json

from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_documents import manifest
from gateway.hosted_room_driver import validate_bound_task_manifest
from gateway import hosted_rooms


def task_documents(db_path, binding, task):
    attachments = task['payload'].get('attachments')
    if not attachments:
        return None
    bound = validate_bound_task_manifest(attachments)
    member = str(task['payload'].get('target_member_id') or task['payload']['target_profile'])
    identity = [binding.room_id, binding.gateway_id, binding.authority_epoch, member,
                task['identity'].task_id, task['execution_generation']]
    key = 'group.document-input.v1.' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old is not None:
            saved = manifest(json.loads(old[0]), member_id=member)
            if [{k: v for k, v in item.items() if k not in {'sha256', 'recipient_member_id'}} for item in saved] != bound:
                raise ValueError('peer input identity changed')
            return saved
    # Reject the complete format/size envelope before any source read.
    manifest([{**item, 'sha256': '0' * 64, 'recipient_member_id': member} for item in bound], member_id=member)
    store = HostedRoomAttachmentStore(db_path)
    result = []
    for item in bound:
        described = store.describe(room_id=binding.room_id, attachment_id=item['attachment_id'],
                                   event_id=item['event_id'], recipient_member_id=member)
        if any(described[k] != item[k] for k in ('kind', 'name', 'mime', 'size')):
            raise ValueError('peer document source identity changed')
        result.append({**item, 'sha256': described['sha256'], 'recipient_member_id': member})
    result = manifest(result, member_id=member)
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'))
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (key, encoded))
        if conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0] != encoded:
            raise ValueError('peer input identity changed')
    return result


def transfer_documents(db_path, dispatch):
    """Only called after the receiver proves this attempt requires input bytes."""
    if db_path is None:
        raise ValueError('peer document source storage unavailable')
    inputs = manifest(dispatch.document_inputs, member_id=dispatch.member_id)
    store = HostedRoomAttachmentStore(db_path)
    result = []
    for item in inputs:
        saved = store.read(room_id=dispatch.room_id, event_id=item['event_id'],
                          attachment_id=item['attachment_id'], recipient_member_id=dispatch.member_id)
        if (any(saved.attachment[k] != item[k] for k in ('kind', 'name', 'mime', 'size'))
                or hashlib.sha256(saved.data).hexdigest() != item['sha256']):
            raise ValueError('peer document source identity changed')
        result.append(base64.b64encode(saved.data).decode('ascii'))
    return result
