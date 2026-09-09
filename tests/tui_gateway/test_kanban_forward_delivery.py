"""Real SQLite/native ownership tests; no provider or worker execution."""
import contextlib
import threading
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from hermes_cli import kanban_db_delivery as delivery
from tui_gateway import server
from tui_gateway.kanban_delivery import reconcile


@pytest.fixture
def native(tmp_path, monkeypatch):
    home = tmp_path / 'profile'; home.mkdir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    db = SessionDB(home / 'state.db')
    db.create_session('origin', source='tui')
    conn = kbc.connect()
    tid = kb.create_task(conn, title='untrusted task', assignee='original-worker')
    kbn.add_notify_sub(conn, task_id=tid, platform='tui', chat_id='origin')
    kb.complete_task(conn, tid, summary='result data')
    session = dict(session_key='origin', history_lock=threading.Lock(), running=False,
                   agent=object(), history=[], profile_home=str(home), source='tui')
    @contextlib.contextmanager
    def owner_db(_):
        yield db
    session['_native_session_db'] = server._session_db
    monkeypatch.setattr(server, '_session_db', owner_db)
    monkeypatch.setattr(server, '_load_cfg', lambda: {})
    yield db, conn, tid, session
    if session.get('active_session_lease'):
        server._release_active_session_slot(session)
    conn.close(); db.close()


def cursor(conn):
    return kbn.list_notify_subs(conn)[0]['last_event_id']


def test_poll_durable_cache_repeat_and_empty_no_lease(native):
    db, conn, tid, session = native
    before = cursor(conn)
    server._notif_poll_kanban('live', session)
    assert cursor(conn) > before
    assert len(db.get_messages('origin')) == 1
    assert len(session['history']) == 1
    assert not session.get('active_session_lease')
    server._notif_poll_kanban('live', session)
    assert len(db.get_messages('origin')) == 1
    assert not session.get('active_session_lease')
    assert kb.get_task(conn, tid).assignee == 'original-worker'


@pytest.mark.parametrize('gate', ['running', 'queued_prompt', 'queued_prompts', '_auto_continue_scheduled'])
def test_pending_human_and_running_defer(native, gate):
    db, conn, tid, session = native
    before = cursor(conn)
    session[gate] = True
    reconcile(server, 'live', session)
    assert cursor(conn) == before and db.get_messages('origin') == []
    assert not session.get('active_session_lease')


@pytest.mark.parametrize('expired', [False, True])
def test_other_turn_owner_never_stolen_or_released(native, expired):
    db, conn, tid, session = native
    db.try_acquire_session_turn_lease('origin', 'unknown-other-owner')
    if expired:
        db._execute_write(lambda c: c.execute('UPDATE session_turn_leases SET expires_at=1'))
    with db._read_ctx() as c:
        before = [tuple(r) for r in c.execute('SELECT * FROM session_turn_leases')]
    reconcile(server, 'live', session)
    with db._read_ctx() as c:
        assert before == [tuple(r) for r in c.execute('SELECT * FROM session_turn_leases')]
    assert db.get_messages('origin') == []


def test_partial_batch_stops_at_failed_eligible_event(native):
    db, conn, tid, session = native
    # complete_task is a transition, not a repeat-event emitter. Reopen the
    # synthetic task as in the existing notification regression fixture.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.complete_task(conn, tid, summary='second result')
    eligible = [e.id for e in kb.list_events(conn, tid) if e.kind == 'completed']
    assert len(eligible) == 2
    print('PARTIAL_BATCH_EVENTS', [(e.id, e.kind, e.payload) for e in kb.list_events(conn, tid)])
    # Native trigger aborts the second message, not the admission method.
    db._execute_write(lambda c: c.execute("CREATE TRIGGER fail_second BEFORE INSERT ON messages "
        "WHEN (SELECT COUNT(*) FROM messages)>0 BEGIN SELECT RAISE(ABORT,'second fails'); END"))
    with pytest.raises(Exception, match='second fails'):
        reconcile(server, 'live', session)
    assert len(db.get_messages('origin')) == 1
    assert cursor(conn) == min(eligible)
    assert cursor(conn) < max(eligible)
    db._execute_write(lambda c: c.execute('DROP TRIGGER fail_second'))
    reconcile(server, 'live', session)
    assert len(db.get_messages('origin')) == 2


def test_frozen_legacy_cutover_repeat_and_subgeneration(native):
    db, conn, tid, session = native
    # Freeze the exact old claim state; both lost and delivered histories are ambiguous.
    with kb.write_txn(conn):
        conn.execute('UPDATE kanban_notify_subs SET delivery_version=0, subscription_generation=NULL, cutover_event_id=NULL')
    sub = kbn.list_notify_subs(conn)[0]
    key = {k: sub[k] for k in ('task_id', 'platform', 'chat_id', 'thread_id')}
    _, old_cursor, events = kbn.claim_unseen_events_for_sub(conn, **key)
    frozen = [tuple(r) for r in conn.execute('SELECT * FROM task_events')]
    sub, pending = delivery.read_pending(conn, sub)
    assert pending == [] and sub['cutover_event_id'] == old_cursor
    for delivered in (False, True):
        if delivered:
            db.append_message('origin', 'system', content='old delivered text')
        with pytest.raises(ValueError, match='Ambiguous legacy'):
            delivery.refuse_legacy_replay(sub, old_cursor)
        assert cursor(conn) == old_cursor
    generation = sub['subscription_generation']
    delivery.migrate(conn); conn.commit()
    again, _ = delivery.read_pending(conn, sub)
    assert again['subscription_generation'] == generation
    kbn.remove_notify_sub(conn, **key); kbn.add_notify_sub(conn, **key)
    replacement, _ = delivery.read_pending(conn, sub)
    assert replacement['subscription_generation'] != generation
    assert not delivery.acknowledge(conn, sub, old_cursor, old_cursor + 1)
    assert frozen == [tuple(r) for r in conn.execute('SELECT * FROM task_events')]


def test_thread_separation_and_alias_dedup(native, monkeypatch):
    db, conn, tid, session = native
    kbn.add_notify_sub(conn, task_id=tid, platform='tui', chat_id='origin', thread_id='foreign-thread')
    monkeypatch.setattr(kb, 'list_boards', lambda **_: [{'slug': 'default'}, {'slug': 'alias'}])
    reconcile(server, 'live', session)
    assert len(db.get_messages('origin')) == 1
    subs = kbn.list_notify_subs(conn)
    assert next(s for s in subs if s['thread_id'])['last_event_id'] == max(e.id for e in kb.list_events(conn, tid))


@pytest.mark.parametrize('gate', ['competitor', 'corrupt', 'read-error', 'stale-slot'])
def test_host_registry_refuses_unproven_owner(native, monkeypatch, gate):
    from hermes_cli import active_sessions as active
    db, conn, tid, session = native
    before = cursor(conn)
    lease = None
    if gate == 'competitor':
        lease, error = active.try_acquire_active_session(
            session_id='origin', surface='tui', config={},
            registry_home=session['profile_home'], metadata={'live_session_id': 'competitor'})
        assert error is None
        saved = lease.state_path.read_bytes()
    elif gate == 'corrupt':
        path = Path(session['profile_home']) / 'runtime' / 'active_sessions.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{not-json')
    elif gate == 'read-error':
        def unavailable(*args, **kwargs):
            raise active.ActiveSessionRegistryError('fixture read unavailable')
        monkeypatch.setattr(active, '_read_entries', unavailable)
    else:
        assert server._ensure_active_session_slot('live', session) is None
        session['active_session_lease'].release()
    try:
        server._notif_poll_kanban('live', session)
        assert cursor(conn) == before
        assert db.get_messages('origin') == []
        if lease:
            assert lease.state_path.read_bytes() == saved
    finally:
        if lease:
            lease.release()


def test_cache_failure_retry_preserves_receipt_and_cursor(native, monkeypatch):
    db, conn, tid, session = native
    before = cursor(conn)
    load = server._load_resume_transcript
    def fail(*args):
        raise OSError('cache unavailable')
    monkeypatch.setattr(server, '_load_resume_transcript', fail)
    with pytest.raises(OSError, match='cache unavailable'):
        reconcile(server, 'live', session)
    assert cursor(conn) == before
    assert len(db.get_messages('origin')) == 1
    monkeypatch.setattr(server, '_load_resume_transcript', load)
    reconcile(server, 'live', session)
    assert cursor(conn) > before
    assert len(db.get_messages('origin')) == len(session['history']) == 1


def test_profile_bound_host_uses_native_store_not_launcher(native, monkeypatch, tmp_path):
    db, conn, tid, session = native
    # Remove only the fixture DB override; exercise the actual profile resolver.
    monkeypatch.setattr(server, '_session_db', session['_native_session_db'])
    other = tmp_path / 'other-profile'; other.mkdir()
    other_db = SessionDB(other / 'state.db')
    try:
        other_db.create_session('foreign', source='tui')
    finally:
        other_db.close()
    session['profile_home'] = str(other)
    session['session_key'] = 'foreign'
    before = cursor(conn)
    reconcile(server, 'foreign-live', session)
    assert cursor(conn) == before
    assert db.get_messages('origin') == []
    session['profile_home'] = str(Path(db.db_path).parent)
    session['session_key'] = 'origin'
    reconcile(server, 'origin-live', session)
    assert len(db.get_messages('origin')) == 1


def test_board_replacement_changes_identity_and_rejects_old_ack(native, tmp_path):
    db, conn, tid, session = native
    old, _ = delivery.read_pending(conn, kbn.list_notify_subs(conn)[0])
    replacement_path = tmp_path / 'replacement.db'
    replacement = kbc.connect(replacement_path)
    try:
        # A different physical board must never accept the old envelope.
        new_tid = kb.create_task(replacement, title='replacement', assignee='worker')
        kbn.add_notify_sub(replacement, task_id=new_tid, platform='tui', chat_id='origin')
        new, _ = delivery.read_pending(replacement, kbn.list_notify_subs(replacement)[0])
        assert new['board_generation'] != old['board_generation']
        assert not delivery.acknowledge(replacement, old, old['last_event_id'], old['last_event_id'] + 1)
    finally:
        replacement.close()


def test_native_repeat_board_migration_retains_forward_receipts(native):
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    before = dict(kbn.list_notify_subs(conn)[0])
    path = Path(conn.execute('PRAGMA database_list').fetchone()[2])
    for _ in range(2):
        kbc.init_db(path)
        assert dict(kbn.list_notify_subs(conn)[0]) == before
    reconcile(server, 'live', session)
    assert len(db.get_messages('origin')) == 1
