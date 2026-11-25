"""Closed ledger authority and token-guarded retry contracts."""
import json
import os
import time

import pytest
import psutil

from hermes_cli.process_identity import worker_birth
from gateway.session_worker_delegation import _probe
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.workers import _gateway
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session


def worker_delegation_handlers(execution_id, worker_pid, birth):
    """The backend's ledger handlers for a worker whose process the gateway probed first."""
    return _gateway.worker_delegation_handlers(execution_id, worker_pid, birth, *_probe(worker_pid, birth))


def test_worker_dispatch_is_bound_to_execution_and_cannot_replace_foreign_rows(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, 'parent', 'cli')
    seed_session(db, 'other', 'cli')
    holder = raw_holder(db)
    handlers = worker_delegation_handlers('worker-a', os.getpid(), worker_birth(psutil.Process()))
    payload = {'delegation_id': 'deleg_owned', 'task': {'goal': 'inert'}, 'dispatched_at': time.time(),
               'parent_session_id': 'parent', 'session_key': '', 'origin_ui_session_id': '', 'origin_session_id': ''}
    def apply(name, payload, sid='parent', table=handlers):
        return holder._execute_write(lambda conn: table[name](holder, conn, sid, payload))
    try:
        apply('delegation.dispatch', payload)
        foreign = worker_delegation_handlers('worker-b', os.getpid(), worker_birth(psutil.Process()))
        for sid, table in [('other', handlers), ('parent', foreign)]:
            with pytest.raises(AuthorityRefusal, match='permission_denied'):
                apply('delegation.read', {'delegation_id': 'deleg_owned'}, sid, table)
            with pytest.raises(AuthorityRefusal, match='permission_denied'):
                apply('delegation.dispatch', dict(payload, parent_session_id=sid), sid, table)
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            apply('delegation.dispatch', dict(payload, delegation_id='bad', session_key='foreign-route'))
        with pytest.raises(AuthorityRefusal, match='invalid_params'):
            apply('delegation.read', {'delegation_id': 'deleg_owned', 'sql': 'DELETE FROM sessions'})
        with holder._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0] == 1
    finally:
        db.close()


def test_worker_completion_claims_refund_and_preserve_terminal_result(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, 'parent', 'cli')
    holder = raw_holder(db)
    handlers = worker_delegation_handlers('worker-a', os.getpid(), worker_birth(psutil.Process()))
    def apply(name, **payload):
        return holder._execute_write(lambda conn: handlers['delegation.' + name](holder, conn, 'parent', payload))
    try:
        apply('dispatch', delegation_id='deleg_owned', task={'goal': 'inert'}, dispatched_at=time.time(),
              parent_session_id='parent', session_key='', origin_ui_session_id='', origin_session_id='')
        event = {'type': 'async_delegation', 'delegation_id': 'deleg_owned', 'parent_session_id': 'parent',
                 'session_key': '', 'origin_ui_session_id': '', 'origin_session_id': '', 'status': 'completed',
                 'completed_at': time.time(), 'summary': 'durable'}
        apply('complete', event=event, result={'summary': 'durable'})
        for index in range(10):
            token = f'claim-{index}'
            assert apply('claim', delegation_id='deleg_owned', claim_id=token)['value']
            assert not apply('ack', delegation_id='deleg_owned', claim_id='foreign')['value']
            assert apply('defer', delegation_id='deleg_owned', claim_id=token)['value']
        row = apply('read', delegation_id='deleg_owned')['value']
        assert row['delivery_attempts'] == 0 and row['delivery_state'] == 'pending'
        assert apply('claim', delegation_id='deleg_owned', claim_id='accepted')['value']
        assert apply('ack', delegation_id='deleg_owned', claim_id='accepted')['value']
        assert not apply('claim', delegation_id='deleg_owned', claim_id='late')['value']
        with pytest.raises(AuthorityRefusal, match='admission_conflict'):
            apply('complete', event=dict(event, summary='overwrite'), result={'summary': 'overwrite'})
        with holder._read_ctx() as conn:
            row = conn.execute('SELECT result_json,delivery_state FROM async_delegations').fetchone()
            assert json.loads(row[0]) == {'summary': 'durable'} and row[1] == 'delivered'
    finally:
        db.close()
