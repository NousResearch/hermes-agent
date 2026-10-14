import json
import os
from types import SimpleNamespace
from concurrent.futures import Future

from tools.browser_tab_lifecycle import Authority, sweep
from tools.browser_tab_ownership import OwnershipRegistry


def test_real_kanban_done_worker_still_running(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _process_fingerprint
    from hermes_state import SessionDB
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    sdb = SessionDB(tmp_path / 'state.db')
    sdb.close()
    from hermes_cli.kanban_db_connect import connect
    conn = connect(tmp_path / 'board.db')
    try:
        task = kb.create_task(conn, title='isolated browser owner', assignee='default')
        kb.recompute_ready(conn)
        run = kb.claim_task(conn, task)
        assert run
        run_id = run.current_run_id
        with conn:
            conn.execute('UPDATE task_runs SET worker_pid=?,worker_started_at=? WHERE id=?',
                (os.getpid(), _process_fingerprint(os.getpid()), run_id))
        monkeypatch.setenv('HERMES_KANBAN_TASK', task)
        monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(run_id))
        a = Authority(tmp_path)
        owner = a.owner(task, None)
        assert owner[0].startswith('kanban:')
        assert kb.complete_task(conn, task, summary='fixture finished card, native worker lives',
            expected_run_id=run_id, fire_lifecycle_hook=False)
        assert a.evidence(*owner) is None
        with conn:
            conn.execute('UPDATE task_runs SET worker_pid=NULL,worker_started_at=NULL WHERE id=?', (run_id,))
        assert a.evidence(*owner)
    finally:
        conn.close()


def test_timeout_future_keeps_child_then_records_native_finish(tmp_path, monkeypatch):
    from tools import delegate_tool_child_run as runner
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('child:unit', 'child-id', 'ws', 'd')
    r.record_created(t, 'child-tab')
    r.finish(t)
    child = SimpleNamespace(_browser_tab_home=str(tmp_path), _browser_tab_task='child-id')
    future = Future()
    closed = []
    monkeypatch.setattr(runner, '_close_child', lambda child, message: closed.append(child))
    runner._defer_close_after_timeout(child, future)
    a = Authority(tmp_path)
    assert a.evidence('child:unit', 'child-id') is None
    future.set_result(None)
    assert closed == [child]
    assert a.evidence('child:unit', 'child-id')


def test_registered_process_hold_needs_canonical_exit(tmp_path):
    import subprocess
    import sys
    from tools.browser_tab_lifecycle import register_process_hold
    from tools.process_registry import ProcessSession
    from tools.process_registry_results import save_completed_result
    from hermes_constants import get_hermes_home
    from unittest.mock import patch
    r = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    t = r.admit('session:a', 'g', 'ws', 'd')
    r.record_created(t, 'owned')
    r.finish(t)
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        process = ProcessSession(id='proc_testhold', command='isolated test', task_id='a',
            owner_task_id='a', session_key='', cwd=str(tmp_path), pid=p.pid)
        actor = register_process_hold(tmp_path, 'ws', 'owned', process)
        a = Authority(tmp_path)
        assert not a.actor_finished(actor)
        p.terminate()
        p.wait(timeout=5)
        process.exited = True
        process.exit_code = p.returncode
        with patch('tools.process_registry_results.get_hermes_home', lambda: tmp_path):
            save_completed_result(process)
        assert a.actor_finished(actor)
        r.retire('session:a', 'g', evidence='fixture:terminal')
        r.release('ws', 'owned', actor)
        assert r.pending('ws')[0]['target'] == 'owned'
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)


def test_native_startup_recovers_only_enabled_profile(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_constants import get_hermes_home
    from tools import browser_tab_lifecycle as life
    from agent import periodic_scheduler
    homes = [tmp_path / 'enabled', tmp_path / 'disabled']
    for index, home in enumerate(homes):
        OwnershipRegistry(home / 'browser_tabs.sqlite')
        (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: ' + ('true' if index == 0 else 'false') + '\n')
    captured = []
    monkeypatch.setattr(life, '_handles', {})
    monkeypatch.setattr(periodic_scheduler, 'schedule',
        lambda callback, interval: captured.append((str(get_hermes_home()), interval, callback)))
    for home in homes:
        token = set_hermes_home_override(home)
        try:
            life.startup(home)
        finally:
            reset_hermes_home_override(token)
    assert [(home, interval) for home, interval, _ in captured] == [(str(homes[0]), 60)]
