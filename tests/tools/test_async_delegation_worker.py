"""Closed ledger authority and token-guarded retry contracts."""
import json
import os
import time

import pytest
import psutil

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tools import async_delegation as ledger


def test_worker_dispatch_is_bound_to_execution_and_cannot_replace_foreign_rows(tmp_path):
    from gateway.session_worker_delegation import worker_delegation_handlers
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    db.create_session('other', source='cli')
    handlers = worker_delegation_handlers('worker-a', os.getpid(), psutil.Process().create_time())
    payload = {'delegation_id': 'deleg_owned', 'task': {'goal': 'inert'}, 'dispatched_at': time.time(),
               'parent_session_id': 'parent', 'session_key': '', 'origin_ui_session_id': '', 'origin_session_id': ''}
    def apply(name, payload, sid='parent', table=handlers):
        return db._execute_write(lambda conn: table[name](db, conn, sid, payload))
    try:
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
            ledger._initialize_schema(conn)
        apply('delegation.dispatch', payload)
        foreign = worker_delegation_handlers('worker-b', os.getpid(), psutil.Process().create_time())
        for sid, table in [('other', handlers), ('parent', foreign)]:
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                apply('delegation.read', {'delegation_id': 'deleg_owned'}, sid, table)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                apply('delegation.dispatch', dict(payload, parent_session_id=sid), sid, table)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            apply('delegation.dispatch', dict(payload, delegation_id='bad', session_key='foreign-route'))
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            apply('delegation.read', {'delegation_id': 'deleg_owned', 'sql': 'DELETE FROM sessions'})
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0] == 1
    finally:
        db.close()


def test_worker_completion_claims_refund_and_preserve_terminal_result(tmp_path):
    from gateway.session_worker_delegation import worker_delegation_handlers
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    handlers = worker_delegation_handlers('worker-a', os.getpid(), psutil.Process().create_time())
    def apply(name, **payload):
        return db._execute_write(lambda conn: handlers['delegation.' + name](db, conn, 'parent', payload))
    try:
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
            ledger._initialize_schema(conn)
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
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            apply('complete', event=dict(event, summary='overwrite'), result={'summary': 'overwrite'})
        with db._read_ctx() as conn:
            row = conn.execute('SELECT result_json,delivery_state FROM async_delegations').fetchone()
            assert json.loads(row[0]) == {'summary': 'durable'} and row[1] == 'delivered'
    finally:
        db.close()
