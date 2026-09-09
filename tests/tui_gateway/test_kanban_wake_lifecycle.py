"""Provider-free: real admission/receipt/ownership; execution instrumented after admission."""
import threading
import pytest
from tests.tui_gateway.test_kanban_forward_delivery import native
from tui_gateway import server
from tui_gateway.kanban_delivery import reconcile


@pytest.fixture(autouse=True)
def executable_agent(native):
    native[3]['agent'] = type('Agent', (), {'session_id': 'origin',
        'run_conversation': lambda *a, **k: None})()


def rows(db):
    with db._read_ctx() as c:
        return [dict(r) for r in c.execute('SELECT * FROM notification_receipts')]


def test_subscription_is_forward_at_creation(native):
    from hermes_cli import kanban_db_notify as kbn
    db, conn, tid, session = native
    sub = kbn.list_notify_subs(conn)[0]
    assert sub['delivery_version'] == 1
    assert sub['subscription_generation']


@pytest.mark.parametrize('state', ['admitted', 'started'])
def test_recover_native_boundary_without_duplicate(native, monkeypatch, state):
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    # Fault injection AFTER real admission, before the native thread is started.
    real_thread = server.threading.Thread
    class FailStart(real_thread):
        def start(self):
            assert rows(db)[0]['continuation_state'] == 'admitted'
            raise RuntimeError('after admission before thread')
    monkeypatch.setattr(server.threading, 'Thread', FailStart)
    server._notif_poll_kanban('live', session)
    assert rows(db)[0]['continuation_state'] == 'admitted'
    monkeypatch.setattr(server.threading, 'Thread', real_thread)
    if state == 'started':
        from tui_gateway.kanban_delivery import _transition
        _transition(db, rows(db)[0]['identity_json'], 'admitted', 'started', 'dead-host',
                    expected_owner=rows(db)[0]['continuation_owner'])
    done = threading.Event()
    monkeypatch.setattr(server, '_prepare_turn_input', lambda *a: ('p', 'p', 80, None))
    def invoke(*args):
        done.set()
        raise RuntimeError('model stopped')
    monkeypatch.setattr(server, '_invoke_agent', invoke)
    session['agent'] = type('Agent', (), {'session_id': 'origin', 'run_conversation': lambda *a, **k: None})()
    server._notif_poll_kanban('live', session)
    if state == 'admitted':
        assert done.wait(3)
        session['_run_thread'].join(3)
    else:
        assert not done.is_set()
    assert rows(db)[0]['continuation_state'] == 'blocked'
    server._notif_poll_kanban('live', session)
    assert len(db.get_messages('origin')) == 1


def test_native_admission_retry_and_uncertain_started_blocks(native, monkeypatch):
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    assert rows(db)[0]['continuation_state'] == 'pending'
    # Refusal at the real host registry boundary must leave continuation pending.
    ensure = server._ensure_active_session_slot
    monkeypatch.setattr(server, '_ensure_active_session_slot', lambda *a: 'not admitted')
    server._notif_poll_kanban('live', session)
    assert rows(db)[0]['continuation_state'] == 'pending'
    monkeypatch.setattr(server, '_ensure_active_session_slot', ensure)
    seen = []
    done = threading.Event()
    monkeypatch.setattr(server, '_emit', lambda e, s, p=None: seen.append((e, s, p)))
    # Native admission and thread admission are NOT mocked. Stop model execution
    # only once the actual runner crosses its durable started boundary.
    def prepared(*args):
        return ('prompt', 'prompt', 80, None)
    def invoke(*args):
        assert rows(db)[0]['continuation_state'] == 'started'
        done.set()
        raise RuntimeError('provider-free execution stopped')
    monkeypatch.setattr(server, '_prepare_turn_input', prepared)
    monkeypatch.setattr(server, '_invoke_agent', invoke)
    session['agent'] = type('Agent', (), {'session_id': 'origin', 'run_conversation': lambda *a, **k: None})()
    server._notif_poll_kanban('live', session)
    assert done.wait(3)
    session['_run_thread'].join(3)
    assert rows(db)[0]['continuation_state'] == 'blocked'
    assert len(db.get_messages('origin')) == 1
    server._notif_poll_kanban('live', session)
    assert any(e == 'status.update' and s == 'live' and 'blocked' in p['text'].lower()
               for e, s, p in seen if p)
    assert len(db.get_messages('origin')) == 1
    assert sum(e == 'message.start' for e, _, _ in seen) == 1


def test_owner_token_fences_stale_aba(native):
    from tui_gateway.kanban_delivery import _transition
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    identity = rows(db)[0]['identity_json']
    _transition(db, identity, 'pending', 'admitted', 'old')
    _transition(db, identity, 'admitted', 'pending', None, expected_owner='old')
    _transition(db, identity, 'pending', 'admitted', 'new')
    with pytest.raises(RuntimeError, match='state changed'):
        _transition(db, identity, 'admitted', 'started', 'old', expected_owner='old')
    assert rows(db)[0]['continuation_owner'] == 'new'
    assert rows(db)[0]['continuation_state'] == 'admitted'


def test_blocked_prefix_does_not_starve_and_reminder_is_durable(native, monkeypatch):
    from hermes_cli import kanban_db as kb
    from tui_gateway.kanban_delivery import continue_pending
    db, conn, tid, session = native
    for i in range(33):
        reconcile(server, 'live', session)
        db._execute_write(lambda c: c.execute(
            "UPDATE notification_receipts SET continuation_state='blocked', continuation_error='uncertain'"))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.complete_task(conn, tid, summary='next')
    reconcile(server, 'live', session)
    calls, notices = [], []
    monkeypatch.setattr(server, '_run_prompt_submit', lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(server, '_emit', lambda e, s, p=None: notices.append((e, p)))
    continue_pending(server, 'live', session)
    assert len(calls) == 1
    blocked = [r for r in rows(db) if r['continuation_state'] == 'blocked']
    assert any(r['continuation_notice_after'] for r in blocked)
    session['running'] = False
    # A fresh UI record cannot erase the durable cooldown or blocked outcome.
    session.pop('_kanban_blocked_displayed', None)
    notices.clear()
    continue_pending(server, 'live', session)
    assert sum(e == 'status.update' for e, _ in notices) <= 1
    assert all(r['continuation_state'] == 'blocked' for r in rows(db)[:33])


@pytest.mark.parametrize('change', ['compression', 'generation', 'message-replacement'])
def test_continuation_identity_at_dispatch(native, monkeypatch, change):
    from tui_gateway.kanban_delivery import continue_pending
    from hermes_state_errors import _STATE_DB_GENERATION_KEY
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    if change == 'compression':
        db.end_session('origin', end_reason='compression')
        db.create_session('tip', source='tui', parent_session_id='origin')
        session['session_key'] = 'tip'
        session['agent'].session_id = 'tip'
    elif change == 'generation':
        db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?', ('replacement', _STATE_DB_GENERATION_KEY)))
    else:
        db._execute_write(lambda c: c.execute("UPDATE messages SET content='replacement'"))
    calls = []
    monkeypatch.setattr(server, '_run_prompt_submit', lambda *a, **k: calls.append(a) or True)
    continue_pending(server, 'live', session)
    assert len(calls) == (1 if change == 'compression' else 0)
    if change != 'compression':
        assert rows(db)[0]['continuation_state'] == 'blocked'
        assert 'identity' in rows(db)[0]['continuation_error'].lower()


def test_human_lock_in_refuses_notification_reservation(native, monkeypatch):
    from tui_gateway.kanban_delivery import continue_pending
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    with session['history_lock']:
        assert not session['running']
    calls = []
    monkeypatch.setattr(server, '_emit', lambda *a, **k: None)
    monkeypatch.setattr(server, '_run_prompt_submit', lambda *a, **k: calls.append(a) or True)
    continue_pending(server, 'live', session)
    before = dict(session)
    err, fields = server._lock_in_submit_turn(
        'human', 'live', session, 'human', {}, False, None, None)
    assert err is not None and err['error']['code'] == 4091
    assert fields == {} and session == before and len(calls) == 1


def test_real_native_preparation_excludes_losing_human(native, monkeypatch):
    from tui_gateway.kanban_delivery import continue_pending
    db, conn, tid, session = native
    reconcile(server, 'live', session)
    entered, first, both, release = [], threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(server, '_emit', lambda *a, **k: None)
    def prepared(*args):
        entered.append(threading.current_thread())
        first.set()
        if len(entered) == 2:
            both.set()
        assert release.wait(4)
        raise RuntimeError('provider-free stop before model')
    monkeypatch.setattr(server, '_prepare_turn_input', prepared)
    with session['history_lock']:
        assert not session['running']
    try:
        continue_pending(server, 'live', session)
        assert first.wait(2)
        inflight = dict(session['inflight_turn'])
        runner = session['_run_thread']
        err, _ = server._lock_in_submit_turn('human', 'live', session, 'human', {}, False, None, None)
        if err is None:
            assert server._run_prompt_submit('human', 'live', session, 'human', image_paths=[])
        overlap = both.wait(1)
        assert not overlap and len(entered) == 1
        assert err['error']['code'] == 4091
        assert session['_run_thread'] is runner and session['inflight_turn'] == inflight
    finally:
        release.set()
        for thread in entered:
            thread.join(3)
        assert all(not thread.is_alive() for thread in entered)


NATIVE_PROCESS = r'''
import os, sys, threading, json
from pathlib import Path
from types import SimpleNamespace

def guard(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system', 'os.posix_spawn'):
        raise RuntimeError('provider-free probe denied ' + event)
sys.addaudithook(guard)
from hermes_state import SessionDB
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tui_gateway import server
from tui_gateway.kanban_delivery import reconcile, continue_pending
home = Path(os.environ['HERMES_HOME'])
from hermes_cli.profiles import get_profile_dir
assert get_profile_dir('isolated-profile').resolve() == home.resolve()
assert home.is_dir()
from hermes_constants import get_default_hermes_root
from hermes_state_guard import _has_pytest_ancestor, _in_test_context, _real_platform_state_root
import hermes_state
assert get_default_hermes_root().resolve() == home.parent.parent.resolve()
assert not home.resolve().is_relative_to(_real_platform_state_root())
assert _has_pytest_ancestor() and _in_test_context()
assert not hermes_state._STATE_DB_GUARD_BYPASS
assert not os.environ.get('HERMES_STATE_DB_GUARD_BYPASS')
print('GUARD_PROOF', json.dumps({'profile': str(home), 'pytest_ancestry': True,
    'test_context': True, 'bypass': False}), flush=True)
thread_errors = []
threading.excepthook = lambda args: thread_errors.append(str(args.exc_value))
mode = sys.argv[1]
db = SessionDB(home / 'state.db')
if mode != 'recover':
    db.create_session('origin', source='tui')
    conn = kbc.connect()
    tid = kb.create_task(conn, title='untrusted lifecycle data', assignee='worker')
    kbn.add_notify_sub(conn, task_id=tid, platform='tui', chat_id='origin')
    kb.complete_task(conn, tid, summary='completed data')
    conn.close()
def row():
    with db._read_ctx() as c:
        return dict(c.execute('SELECT * FROM notification_receipts').fetchone())
seen = []
server._emit = lambda e, s, p=None: seen.append((e, p))
agent = SimpleNamespace(session_id='origin', run_conversation=lambda *a, **k: None)
session = dict(session_key='origin', source='tui', profile='isolated-profile', profile_home=str(home),
    agent=agent, history_lock=threading.Lock(), running=False, history=[], history_version=0)
server._sessions['live'] = session
# Only prepare/model boundary is instrumented, after actual host and prompt admission.
def prepared(sid, session, st, text, images):
    ordinary = text == 'human normal'
    assert row()['continuation_state'] == ('completed' if ordinary else 'admitted')
    st.history = list(session['history'])
    st.history_version = session['history_version']
    st.prompt_text = text
    def model(message, **kwargs):
        assert row()['continuation_state'] == ('completed' if ordinary else 'started')
        assert server.read_turn_marker(home, 'origin')['auto_continue'] is ordinary
        if mode == 'crash':
            (home / 'started-proof.json').write_text(json.dumps(row()))
            os._exit(73)
        if mode == 'recover':
            raise AssertionError('uncertain started turn executed again')
        db.append_message('origin', 'user', content=message)
        db.append_message('origin', 'assistant', content='native absorbed result')
        return {'final_response': 'native absorbed result', 'messages': db.get_messages('origin')}
    agent.run_conversation = model
    return text, text, 80, None
server._prepare_turn_input = prepared
if mode != 'recover':
    reconcile(server, 'live', session)
continue_pending(server, 'live', session)
if session.get('_run_thread'):
    session['_run_thread'].join(5)
    assert not session['_run_thread'].is_alive()
if mode == 'recover':
    assert row()['continuation_state'] == 'blocked'
    continue_pending(server, 'live', session)
    assert any(e == 'status.update' and 'blocked' in p['text'] for e,p in seen if p)
    assert not any(e == 'message.start' for e,p in seen)
    assert len(db.get_messages('origin')) == 1
else:
    assert row()['continuation_state'] == 'completed', (row(), seen)
    assert sum(e == 'message.start' for e,p in seen) == 1
    assert sum(e == 'message.complete' for e,p in seen) == 1
    assert any(e == 'message.complete' and p['text'] == 'native absorbed result' for e,p in seen if p)
    assert sum(e == 'session.info' for e,p in seen) == 1 and seen[-1][0] == 'session.info'
    assert not thread_errors, thread_errors
    assert any(m['content'] == 'native absorbed result' for m in session['history'])
    continue_pending(server, 'live', session)
    assert sum(e == 'message.start' for e,p in seen) == 1
print(json.dumps({'mode': mode, 'state': row()['continuation_state'], 'events': seen}), flush=True)
if mode == 'success':
    seen.clear()
    err, fields = server._lock_in_submit_turn('human', 'live', session, 'human normal', {}, False, None, None)
    assert err is None and fields == {}, err
    assert server._run_prompt_submit('human', 'live', session, 'human normal', image_paths=[])
    session['_run_thread'].join(5)
    assert not session['_run_thread'].is_alive() and not thread_errors, thread_errors
    assert [e for e,p in seen] == ['message.start', 'message.complete', 'session.info'], seen
    assert seen[1][1]['text'] == 'native absorbed result'
    print(json.dumps({'ordinary': 'completed', 'events': seen}), flush=True)
'''


@pytest.mark.parametrize('mode', ['success', 'crash'])
def test_actual_native_success_and_process_death(tmp_path, mode):
    import os, sys, subprocess, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    tmp_path = tmp_path / 'native-process'; tmp_path.mkdir()
    # Native custom-root convention: keep the profile outside HOME/.hermes,
    # which the intact state-db guard reserves even when HOME is temporary.
    home = tmp_path / 'hermes-root' / 'profiles' / 'isolated-profile'; home.mkdir(parents=True)
    env = {'HOME': str(tmp_path), 'HERMES_HOME': str(home),
           'HERMES_KANBAN_DB': str(tmp_path / 'board.db'), 'PYTHONPATH': str(repo),
           'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'PYTHONNOUSERSITE': '1'}
    env.update({key: value for key, value in os.environ.items()
                if key.startswith('PYTEST_') or key == 'HERMES_TEST_ISOLATION'})
    for phase in ([mode, 'recover'] if mode == 'crash' else [mode]):
        run = subprocess.run([sys.executable, '-c', NATIVE_PROCESS, phase], cwd=repo,
                             env=env, capture_output=True, text=True, timeout=20)
        (tmp_path / (phase + '.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == (73 if phase == 'crash' else 0), run.stdout + run.stderr
        if phase == 'crash':
            assert json.loads((home / 'started-proof.json').read_text())['continuation_state'] == 'started'
        print('NATIVE_PROCESS', phase, run.stdout, flush=True)


def test_busy_display_pending_human_and_cache_repair(native, monkeypatch):
    db, conn, tid, session = native
    emits = []
    monkeypatch.setattr(server, '_emit', lambda e, s, p=None: emits.append((e, s, p)))
    session['running'] = True
    server._notif_poll_kanban('live', session)
    assert session['_kanban_pending']
    assert any(e == 'status.update' and tid in p['text'] for e, s, p in emits if p)
    assert db.get_messages('origin') == []
    session['running'] = False
    session['queued_prompt'] = 'human first'
    server._notif_poll_kanban('live', session)
    assert db.get_messages('origin') == []
    session.pop('queued_prompt')
    load = server._load_resume_transcript
    monkeypatch.setattr(server, '_load_resume_transcript', lambda *a: (_ for _ in ()).throw(OSError('cache')))
    with pytest.raises(OSError):
        reconcile(server, 'live', session)
    assert session['_kanban_cache_dirty']
    monkeypatch.setattr(server, '_load_resume_transcript', load)
    # Exercise actual native admission, not a replacement admission function.
    assert server._admit_prompt_turn('live', session, 'human', [], None)
    assert len(session['history']) == 1
    assert not session.get('_kanban_cache_dirty')
