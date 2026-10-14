import sqlite3

from tools.browser_tab_ownership import OwnershipRegistry


def ledger(home, status):
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    path = home / 'cron' / 'executions.db'
    path.parent.mkdir()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE executions(id TEXT, job_id TEXT, status TEXT, handoff_pending INTEGER, effect_in_flight INTEGER)')
        db.execute('INSERT INTO executions VALUES (?,?,?,?,?)', ('execution', 'job', status, 0, 0))


def test_cron_exact_and_unknown(tmp_path):
    from tools.browser_tab_lifecycle import Authority
    ledger(tmp_path, 'completed')
    authority = Authority(tmp_path)
    owner = authority.owner('cron:job:execution', None)
    assert authority.evidence(*owner)
    assert authority.evidence(*authority.owner('cron:other:execution', None)) is None
    assert authority.evidence('unknown:execution', 'unresolved') is None
    with sqlite3.connect(tmp_path / 'cron/executions.db') as db:
        db.execute("UPDATE executions SET status='running'")
    assert authority.evidence(*owner) is None


def test_retirement_requires_immutable_evidence(tmp_path):
    r = OwnershipRegistry(tmp_path / 'tabs.db')
    token = r.admit('a', 'g', 'ws', 'daemon')
    assert not r.retire('a', 'g', evidence='canonical:completed')
    r.finish(token)
    assert not r.retire('a', 'g')
    assert r.retire('a', 'g', evidence='canonical:completed')
    assert r.retire('a', 'g', evidence='different')
    with sqlite3.connect(r.path) as db:
        assert db.execute('SELECT evidence FROM retirements').fetchone()[0] == 'canonical:completed'


def test_missing_source_does_not_create_database(tmp_path):
    from tools.browser_tab_lifecycle import Authority
    a = Authority(tmp_path)
    assert a.evidence(*a.owner('cron:job:execution', None)) is None
    assert not (tmp_path / 'cron').exists()


def test_sessions_only_explicit_resource_retirement(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tools.browser_tab_lifecycle import Authority
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    db = SessionDB(tmp_path / 'state.db')
    try:
        for reason in ('agent_close', 'idle', 'daily', 'startup_orphan_reap', 'new_session', 'session_reset', 'suspended'):
            db.create_session(reason, 'cli')
            db.end_session(reason, reason)
            a = Authority(tmp_path)
            owner = a.owner('arbitrary', reason)
            assert owner == ('session:' + reason, reason)
            assert bool(a.evidence(*owner)) == (reason in ('new_session', 'session_reset'))
        db.create_session('root', 'cli')
        db.end_session('root', 'compression')
        db.create_session('tip', 'cli', parent_session_id='root')
        assert a.owner('arbitrary', 'tip') == ('session:root', 'root')
        assert a.evidence('session:root', 'root') is None
        db.end_session('tip', 'new_session')
        assert a.evidence('session:root', 'root')
    finally:
        db.close()


def test_sweep_keeps_unknown_and_exact_hold(tmp_path):
    from tools.browser_tab_lifecycle import sweep
    ledger(tmp_path, 'completed')
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    ws = 'ws://fixture/devtools/browser/generation'
    for owner, gen, target in [('cron:job','execution','owned'), ('unknown:x','unresolved','unknown')]:
        token = r.admit(owner, gen, ws, target)
        r.record_created(token, target)
        r.finish(token)
    r.hold(ws, 'owned', 'child:unverified')
    live = {'owned', 'unknown', 'external'}
    class Browser:
        def __init__(self, address):
            assert address == ws
            self.socket = self
        def close(self): pass
        def targets(self): return live.copy()
        def close_target(self, target): live.remove(target)
    sweep(tmp_path, connect=Browser)
    assert live == {'owned', 'unknown', 'external'}
    r.release(ws, 'owned', 'child:unverified')
    sweep(tmp_path, connect=Browser)
    assert live == {'unknown', 'external'}


def test_quarantine_needs_daemon_and_client_proof(tmp_path, monkeypatch):
    from tools import browser_tab_lifecycle as lifecycle
    ledger(tmp_path, 'completed')
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    token = r.admit('cron:job', 'execution', 'ws', 'd')
    r.bind_daemon(token, 111, 'daemon-start')
    with r._db() as db:
        db.execute('UPDATE calls SET client_pid=222, client_start=?', ('client-start',))
    r.quarantine(token)
    monkeypatch.setattr(lifecycle, 'process_dead', lambda pid, start: pid == 111)
    lifecycle.sweep(tmp_path)
    assert r.call(token)['state'] == 'quarantined'
    monkeypatch.setattr(lifecycle, 'process_dead', lambda pid, start: bool(pid and start))
    lifecycle.sweep(tmp_path)
    assert r.call(token)['state'] == 'drained'


def test_kanban_exact_run_and_child_mapping(tmp_path, monkeypatch):
    import json
    from hermes_state import SessionDB
    from tools.browser_tab_lifecycle import Authority
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    db.close()
    board = tmp_path / 'board.db'
    with sqlite3.connect(board) as db:
        db.executescript('CREATE TABLE tasks(id TEXT, status TEXT); CREATE TABLE task_runs(id INTEGER, task_id TEXT, status TEXT, outcome TEXT, ended_at INTEGER, worker_pid INTEGER, worker_started_at TEXT);')
        db.execute("INSERT INTO tasks VALUES ('card', 'done')")
        db.execute("INSERT INTO task_runs VALUES (12,'card','done','completed',1,NULL,NULL)")
    monkeypatch.setenv('HERMES_KANBAN_DB', str(board))
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'card')
    monkeypatch.setenv('HERMES_KANBAN_RUN_ID', '12')
    a = Authority(tmp_path)
    owner = a.owner('card', None)
    assert owner[0].startswith('kanban:')
    assert a.evidence(*owner)
    assert a.evidence(owner[0], '13') is None
    with delegated_child_context():
        assert a.owner('card', None) == ('child-task:card', 'card')
    with non_dispatcher_owned_context():
        assert a.owner('card', None)[0].startswith('unknown:')
    with sqlite3.connect(board) as db:
        db.execute("UPDATE tasks SET status='review'")
    assert a.evidence(*owner) is None
    with sqlite3.connect(tmp_path / 'state.db') as db:
        db.execute('INSERT INTO async_delegations(delegation_id,origin_session,state,dispatched_at,updated_at,task_json) VALUES (?,?,?,?,?,?)',
                   ('unit','parent','completed',1,1,json.dumps({'child_task_ids': {'0':'sa_child'}})))
    assert a.owner('sa_child', None) == ('child:unit','sa_child')
    assert a.evidence('child:unit','sa_child') is None


def test_disable_stops_existing_sweep(tmp_path):
    from tools.browser_tab_lifecycle import sweep
    ledger(tmp_path, 'completed')
    (tmp_path / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: false\n')
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('cron:job', 'execution', 'ws', 'd')
    r.record_created(t, 'a')
    r.finish(t)
    connected = []
    sweep(tmp_path, connect=lambda ws: connected.append(ws))
    assert connected == []
    assert not r.pending('ws')


def test_native_child_finish_releases_exact_hold_only(tmp_path):
    from tools.browser_tab_lifecycle import Authority, native_child_finished
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    for child in ('one', 'two'):
        t = r.admit('child:unit', child, 'ws', child)
        r.record_created(t, child)
        r.finish(t)
    a = Authority(tmp_path)
    assert a.evidence('child:unit', 'one') is None
    native_child_finished('one', tmp_path)
    assert a.evidence('child:unit', 'one')
    assert a.evidence('child:unit', 'two') is None


def test_native_child_completion_waits_for_inflight_request(tmp_path):
    from tools.browser_tab_lifecycle import native_child_finished
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('child:unit', 'one', 'ws', 'd')
    r.request_started(t)
    native_child_finished('one', tmp_path)
    assert not r.pending('ws')
    assert r.call(t)['state'] == 'active'


def test_child_hold_waits_for_quarantined_browser_call(tmp_path):
    import json
    from tools.browser_tab_lifecycle import Authority, native_child_finished
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('child:unit', 'child', 'ws', 'd')
    r.request_started(t)
    r.quarantine(t)
    native_child_finished('child', tmp_path)
    a = Authority(tmp_path)
    assert not a.actor_finished(json.dumps(['child:unit', 'child']))
    with r._db() as db:
        db.execute("UPDATE calls SET state='drained' WHERE token=?", (t,))
    assert a.actor_finished(json.dumps(['child:unit', 'child']))


def test_borrow_only_child_gets_completion_receipt(tmp_path):
    import json
    from tools.browser_tab_lifecycle import Authority, native_child_finished
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('session:parent', 'parent', 'ws', 'd')
    r.record_created(t, 'tab')
    r.finish(t)
    actor = json.dumps(['child-task:borrower', 'borrower'])
    r.hold('ws', 'tab', actor)
    native_child_finished('borrower', tmp_path)
    assert Authority(tmp_path).actor_finished(actor)


def test_unverified_fingerprint_never_proves_dead():
    import os
    from tools.browser_tab_lifecycle import process_dead
    assert not process_dead(os.getpid(), 'unverified')


def test_restart_recovers_active_calls_only_with_both_dead(tmp_path, monkeypatch):
    from tools import browser_tab_lifecycle as life
    ledger(tmp_path, 'completed')
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('cron:job', 'execution', 'ws', 'd')
    r.bind_daemon(t, 111, '1')
    with r._db() as db:
        db.execute('UPDATE calls SET client_pid=222, client_start=?', ('2',))
    monkeypatch.setattr(life, 'process_dead', lambda pid, start: pid == 111)
    life.sweep(tmp_path)
    assert r.call(t)['state'] == 'active'
    monkeypatch.setattr(life, 'process_dead', lambda pid, start: bool(pid and start))
    life.sweep(tmp_path)
    assert r.call(t)['state'] == 'drained'


def test_bounded_batches_do_not_starve_behind_unknown_rows(tmp_path):
    from tools.browser_tab_lifecycle import _batch
    r = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    for index in range(19):
        r.admit('unknown:' + str(index), 'g', 'ws', 'd' + str(index))
    seen = set()
    for _ in range(4):
        with r._db() as db:
            rows = _batch(db, 2, 'SELECT rowid AS position,* FROM calls', 8)
        assert len(rows) <= 8
        seen.update(row['owner'] for row in rows)
    assert len(seen) == 19


def test_kanban_archive_waits_for_native_worker(tmp_path, monkeypatch):
    import os
    from tools.browser_tab_lifecycle import Authority
    board = tmp_path / 'board.db'
    with sqlite3.connect(board) as db:
        db.executescript('CREATE TABLE tasks(id TEXT,status TEXT); CREATE TABLE task_runs(id INTEGER,task_id TEXT,status TEXT,outcome TEXT,ended_at INTEGER,worker_pid INTEGER,worker_started_at TEXT);')
        db.execute("INSERT INTO tasks VALUES ('card','archived')")
        db.execute("INSERT INTO task_runs VALUES (1,'card','reclaimed','reclaimed',1,?,'unverified')", (os.getpid(),))
    owner = 'kanban:' + __import__('json').dumps([str(board), 'card'])
    a = Authority(tmp_path)
    assert a.evidence(owner, '1') is None
    with sqlite3.connect(board) as db:
        db.execute('UPDATE task_runs SET worker_pid=NULL')
    assert a.evidence(owner, '1')
