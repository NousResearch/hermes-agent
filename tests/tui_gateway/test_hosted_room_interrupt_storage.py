"""Real serve handlers fence Stop without waiting on a submit's persistence."""
from concurrent.futures import ThreadPoolExecutor
import errno
import threading
from types import SimpleNamespace

import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway import hosted_room_driver as tasks
from hermes_state import SessionDB
from tests.tui_gateway.test_hosted_room_stop_server import _defer, _turn
from tests.tui_gateway.test_interrupt_dead_compute_host import _DeadComputeHost
from tui_gateway import server
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError
from tui_gateway.turn_marker import read_turn_marker


@pytest.fixture
def live_session(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / 'state.db')
    monkeypatch.setattr(server, '_get_db', lambda: db)
    for name in ('_schedule_agent_build', '_schedule_session_cap_enforcement', '_register_session_cwd',
                 '_start_agent_build', '_tts_stream_stop'):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, '_restart_completed_failed_agent_build', lambda *a: True)
    monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *a: None)
    created = server.handle_request({'id': 'create', 'method': 'session.create', 'params': {
        'source': 'bot_room', 'title': 'Group: room', 'hidden': True, 'room_plumbing': True}})
    sid = created['result']['session_id']
    session = server._sessions[sid]
    session['profile_home'] = str(tmp_path)
    interrupted = threading.Event()
    session['agent'] = SimpleNamespace(interrupt=interrupted.set)
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join(timeout=5)
    assert not dead.is_alive()
    session['_run_thread'] = dead  # A prior turn is not evidence about the pending submit.
    try:
        yield HostedRoomServerRPC(server), sid, session, interrupted
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_stop_does_not_wait_for_submit_storage_or_release_its_turn(live_session, monkeypatch):
    rpc, sid, session, interrupted = live_session
    entered, release = threading.Event(), threading.Event()
    ran = []

    def persist(*args):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(server, '_persist_session_row_for_submit', persist)
    monkeypatch.setattr(server, '_run_prompt_submit', lambda *a, **k: ran.append(True))
    args = dict(profile='default', session_id=sid, source='bot_room', prompt='hello',
                task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
                member_id='one', on_terminal=lambda receipt: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = pool.submit(rpc.submit, **args)
        try:
            assert entered.wait(5)
            stopped = pool.submit(rpc.interrupt, profile='default', session_id=sid,
                source='bot_room', expected_task_id='task', execution_generation=1)
            assert stopped.result(timeout=3)['status'] == 'interrupted'
            assert interrupted.is_set() and session['_turn_cancel_requested']
            # The submit still owns the turn, even before its run thread exists.
            assert session['running']
            with pytest.raises(HostedRoomSessionError, match='busy'):
                rpc.submit(**{**args, 'execution_generation': 2})
        finally:
            release.set()
            submitted.result(timeout=5)
    session['_run_thread'].join(timeout=5)
    assert not session['_run_thread'].is_alive() and not session['running']
    assert not ran


def test_stop_releases_a_dead_hosted_thread_for_retry_and_disband(live_session, tmp_path, monkeypatch):
    rpc, sid, session, interrupted = live_session
    service, adapter_server, _record, task = _turn(tmp_path, monkeypatch)
    _defer(service, task)
    session.update(running=True,
        _hosted_room_task={'task_id': task['identity'].task_id, 'execution_generation': 1})
    adapter_server._sessions['runtime'] = session
    result = rpc.interrupt(profile='default', session_id=sid, source='bot_room',
        expected_task_id=task['identity'].task_id, execution_generation=1)
    assert result['status'] == 'interrupted' and interrupted.is_set()
    assert not session['running']
    assert not rpc.info(profile='default', session_id=sid, source='bot_room')['active']
    assert service.retry_room_task('room', task_id=task['identity'].task_id)['status'] == 'queued'
    assert service.stop_room('room', cancel_id='disband', require_acknowledged=True) == 1
    assert tasks.get_task(service.db_path, task['identity'])['status'] == 'cancelled'


@pytest.mark.parametrize(('failure', 'code'), [
    ('unavailable', 5072), ('disk_full', 5070), ('write_error', 5071), ('startup', None)])
def test_failed_hosted_submit_releases_its_reservation(live_session, monkeypatch, failure, code):
    rpc, sid, session, _interrupted = live_session

    def fail(*args):
        if failure == 'unavailable':
            return False
        if failure == 'disk_full':
            raise OSError(errno.ENOSPC, 'disk full')
        raise RuntimeError('submit failed')

    seam = '_restart_completed_failed_agent_build' if failure == 'startup' else '_ensure_session_db_row'
    with monkeypatch.context() as failing:
        failing.setattr(server, seam, fail)
        with pytest.raises(HostedRoomSessionError if code else RuntimeError) as caught:
            rpc.submit(profile='default', session_id=sid, source='bot_room', prompt='hello',
                task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
                member_id='one', on_terminal=lambda receipt: None)
    if code:
        assert caught.value.code == code
    assert not session.get('_hosted_submit_active')
    # A startup failure has no producer to release running; normal Stop must recover it.
    server.handle_request({'id': 'stop', 'method': 'session.interrupt', 'params': {'session_id': sid}})
    assert not session['running'] and session.get('inflight_turn') is None


@pytest.mark.parametrize('generation', [None, 1, 2])
def test_interrupt_checks_generation_in_the_real_handler(live_session, generation):
    rpc, sid, session, interrupted = live_session
    session.update(running=True, _hosted_room_task={'task_id': 'task', 'execution_generation': 2})
    result = rpc.interrupt(profile='default', session_id=sid, source='bot_room',
                           expected_task_id='task', execution_generation=generation)
    assert interrupted.is_set() is (generation != 1)
    assert result['status'] == ('not_interrupted' if generation == 1 else 'interrupted')


def test_hosted_stop_recovers_an_unreachable_compute_host(live_session, monkeypatch):
    rpc, sid, session, _interrupted = live_session
    session.update(running=True, inflight_turn={'user': 'hello', 'assistant': '', 'streaming': True},
        _hosted_room_task={'task_id': 'task', 'execution_generation': 1})
    monkeypatch.setattr(server, '_session_uses_compute_host', lambda session: True)
    monkeypatch.setattr(server, '_get_compute_host_supervisor', lambda: _DeadComputeHost())
    result = rpc.interrupt(profile='default', session_id=sid, source='bot_room',
        expected_task_id='task', execution_generation=1)
    assert result == {'status': 'interrupted', 'turn_isolation': True}
    assert not session['running'] and session.get('inflight_turn') is None
    assert session['_turn_cancel_requested'] and session['_turn_user_input']


def test_hosted_stop_retires_marker_before_the_interrupted_turn_exits(live_session, monkeypatch):
    rpc, sid, session, interrupted = live_session
    entered, release = threading.Event(), threading.Event()
    hook_lock_available = []

    def run(message, **kwargs):
        entered.set()
        assert release.wait(10)
        return {'final_response': 'stopped'}

    def hook(name, **kwargs):
        if name == 'agent_loop_stopped':
            acquired = session['history_lock'].acquire(timeout=2)
            hook_lock_available.append(acquired)
            if acquired:
                session['history_lock'].release()

    for name in ('_wire_callbacks', '_sync_agent_model_with_config', '_tts_stream_begin',
                 '_sync_session_key_after_compress'):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, '_get_usage', lambda agent: {})
    monkeypatch.setattr('hermes_cli.plugins.invoke_hook', hook)
    session['agent'].run_conversation = run
    session['agent'].clear_interrupt = lambda: None
    session['agent'].session_id = session['session_key']
    rpc.submit(profile='default', session_id=sid, source='bot_room', prompt='hello',
        task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
        member_id='one', on_terminal=lambda receipt: None)
    try:
        assert entered.wait(5)
        home, key = session['profile_home'], session['_active_turn_marker_key']
        assert read_turn_marker(home, key) is not None
        result = rpc.interrupt(profile='default', session_id=sid, source='bot_room',
            expected_task_id='task', execution_generation=1)
        assert result['status'] == 'interrupted' and interrupted.is_set()
        assert session['running'] and session['_run_thread'].is_alive()
        assert read_turn_marker(home, key) is None
        assert '_active_turn_marker_key' not in session
        assert session['_turn_user_input'] and hook_lock_available == [True]
    finally:
        release.set()
        session['_run_thread'].join(timeout=5)
    assert not session['_run_thread'].is_alive() and not session['running']


def test_stop_never_releases_a_prompt_that_followed_the_turn_it_saw_alive(live_session, monkeypatch):
    _, sid, session, _ = live_session
    finish = threading.Event()
    turn = threading.Thread(target=finish.wait, args=(10,))
    turn.start()
    session.update(running=True, _run_thread=turn)

    def next_prompt_during_the_stop_hook(*args, **kwargs):
        finish.set()
        turn.join(timeout=5)
        session.pop('_hosted_room_task', None)

    monkeypatch.setattr('hermes_cli.plugins.invoke_hook', next_prompt_during_the_stop_hook)
    server._interrupt_session_turn(sid, session)
    assert not turn.is_alive() and session['running'] is True


def test_a_hosted_submit_whose_thread_cannot_start_releases_its_slot(live_session, monkeypatch):
    rpc, sid, session, _ = live_session

    def refuse(*args):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(server, '_restart_completed_failed_agent_build', refuse)
    with pytest.raises((RuntimeError, HostedRoomSessionError)):
        rpc.submit(profile='default', session_id=sid, source='bot_room', prompt='hello',
                   task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
                   member_id='one', on_terminal=lambda receipt: None)
    assert session['running'] is False and '_hosted_submit_active' not in session
    assert rpc.info(profile='default', session_id=sid, source='bot_room')['active'] is False
