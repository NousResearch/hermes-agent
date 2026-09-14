"""Process-bound worker ledger routing through the existing durable outbox.

Binding is private runtime wiring, not model tool surface or a storage fallback.
Fresh executor/helper threads retain the registered interpreter's owner boundary.
"""
import os
import threading

from agent.runtime_session_store import WorkerPersistenceError, is_worker_process

_store = None
_lock = threading.Lock()


def bind_worker_delegation_store(store):
    global _store
    if not is_worker_process() or store.scope['pid'] != os.getpid():
        raise WorkerPersistenceError('permission_denied')
    with _lock:
        if _store is not None and _store is not store:
            raise WorkerPersistenceError('worker_delegation_already_bound')
        _store = store


def worker_ledger():
    if not is_worker_process():
        return None
    with _lock:
        if _store is None or _store.scope['pid'] != os.getpid():
            raise WorkerPersistenceError('worker_delegation_ledger_unbound')
        return WorkerDelegationLedger(_store)


def require_ledger_owner():
    if is_worker_process():
        raise WorkerPersistenceError('worker_delegation_owner_only')


class WorkerDelegationLedger:
    def __init__(self, store):
        self.store = store

    def dispatch(self, record):
        if record.get('parent_session_id') != self.store.scope['session_id']:
            raise WorkerPersistenceError('permission_denied')
        from tools.async_delegation import _ROUTING_KEYS
        if any(record.get(key) for key in _ROUTING_KEYS):
            raise WorkerPersistenceError('unsupported_producer')
        return self.store._apply('delegation.dispatch', {
            key: record.get(key, '') for key in ('delegation_id', 'parent_session_id', 'session_key',
                'origin_ui_session_id', 'origin_session_id', 'dispatched_at')} | {'task': {
            key: record[key] for key in ('goal', 'goals', 'context', 'toolsets', 'role', 'model', 'is_batch', 'task_indexes')
            if key in record}})['value']

    def complete(self, event, result):
        return self.store._apply('delegation.complete', {'event': event, 'result': result})['value']

    def child(self, delegation_id, entry):
        return self.store._apply('delegation.child', {'delegation_id': delegation_id, 'entry': entry})['value']

    def read(self, delegation_id):
        return self.store._apply('delegation.read', {'delegation_id': delegation_id})['value']

    def claim(self, delegation_id, claim_id):
        return self.store._apply('delegation.claim', {'delegation_id': delegation_id, 'claim_id': claim_id})['value']

    def ack(self, delegation_id, claim_id):
        return self.store._apply('delegation.ack', {'delegation_id': delegation_id, 'claim_id': claim_id})['value']

    def defer(self, delegation_id, claim_id):
        return self.store._apply('delegation.defer', {'delegation_id': delegation_id, 'claim_id': claim_id})['value']

    def release(self, delegation_id, claim_id):
        return self.store._apply('delegation.release', {'delegation_id': delegation_id, 'claim_id': claim_id})['value']

    def drop(self, delegation_id, claim_id):
        return self.store._apply('delegation.drop', {'delegation_id': delegation_id, 'claim_id': claim_id})['value']

    def unscheduled(self, delegation_id):
        return self.store._apply('delegation.unscheduled', {'delegation_id': delegation_id})['value']
