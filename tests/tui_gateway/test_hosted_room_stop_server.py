"""The standalone group chat adapter distinguishes interrupt delivery from completion."""
import threading
import time

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from tests.tui_gateway.test_hosted_room_server_rpc import _server
from tui_gateway.hosted_room_service import HostedRoomService


def _turn(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    server, _calls = _server()
    service = HostedRoomService(server, db_path=tmp_path / 'state.db')
    service.local_profiles = lambda: ('default', 'other')
    service.create_room(room_id='room', name='Room', members=[
        {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
        {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
    service.send(room_id='room', event_id='first', payload={'text': '@one first', 'thread_id': 'thread'})
    task, = tasks.list_tasks(service.db_path, room_id='room', status='queued')
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    tasks.start_task(service.db_path, task['identity'], lease, expected_cancel_generation=0, clock=time.time)
    record = server._sessions['runtime'] = {
        'history_lock': threading.Lock(), 'running': True,
        '_hosted_room_task': {'task_id': task['identity'].task_id, 'execution_generation': 1}}
    return service, server, record, task


@pytest.mark.parametrize('ack', [{'status': 'interrupted'}, {'interrupted': True}])
def test_standalone_stop_waits_for_the_session_to_finish(tmp_path, monkeypatch, ack):
    service, server, record, task = _turn(tmp_path, monkeypatch)
    server._methods['session.interrupt'] = lambda rid, params: {'id': rid, 'result': ack}
    assert service.stop_room('room', cancel_id='stop') == 1
    assert record['running']
    assert tasks.get_task(service.db_path, task['identity'])['status'] == 'stopping'
    with pytest.raises(rooms.RoomStopPendingError):
        service.stop_room('room', cancel_id='end', require_acknowledged=True)
    record['running'] = False  # the producer finally leaves its run thread
    service.stop_room('room', cancel_id='end', require_acknowledged=True)
    assert tasks.get_task(service.db_path, task['identity'])['status'] == 'cancelled'


def _defer(service, task, *, deadline=False):
    runtime, binding = service.runtime, service.bindings()[0]
    lease = runtime._ensure_lease(binding)
    task = tasks.get_task(service.db_path, task['identity'])
    stopping = tasks.begin_task_cancel(service.db_path, task['identity'],
        cancel_id=f"deadline:{task['execution_generation']}" if deadline else 'stop',
        expected_cancel_generation=task['cancel_generation'], clock=runtime.clock)
    deferred = tasks.defer_unconfirmed_stop(service.db_path, task['identity'], lease,
        expected_execution_generation=task['execution_generation'], expected_cancel_generation=stopping['cancel_generation'],
        reason='member_unavailable', clock=runtime.clock)
    return runtime._publish(binding, deferred)


@pytest.mark.parametrize('outcome', ['active', 'settled', 'idle'])
def test_standalone_retry_confirms_the_deferred_attempt(tmp_path, monkeypatch, outcome):
    service, server, record, task = _turn(tmp_path, monkeypatch)
    _defer(service, task)
    if outcome != 'active':
        record['running'] = False
    if outcome == 'settled':
        server._methods['session.history'] = lambda rid, params: {'result': {'messages': [{
            'role': 'assistant', 'status': 'settled', 'task_id': task['identity'].task_id,
            'execution_generation': 1, 'content': 'kept answer', 'settlement_id': 'answer'}]}}
    retry = lambda: service.retry_room_task('room', task_id=task['identity'].task_id,
                                            member_id='one', execution_generation=1)
    if outcome == 'active':
        with pytest.raises(tasks.InvalidTaskTransitionError, match='active'):
            retry()
        assert tasks.get_task(service.db_path, task['identity'])['status'] == 'deferred'
        with pytest.raises(rooms.RoomStopPendingError):
            service.stop_room('room', cancel_id='disband', require_acknowledged=True)
        service.send(room_id='room', event_id='next', payload={'text': '@two next', 'thread_id': 'next'})
        assert len(tasks.list_tasks(service.db_path, room_id='room', status='queued')) == 1
    else:
        result = retry()
        assert result['status'] == ('settled' if outcome == 'settled' else 'queued')
        if outcome == 'settled':
            assert result['result']['text'] == 'kept answer'


@pytest.mark.parametrize('outcome', ['current', 'stale', 'storage_error'])
def test_cancelled_callback_finishes_a_deferred_deadline_without_reclaiming_queue(
        tmp_path, monkeypatch, caplog, outcome):
    service, _server, record, task = _turn(tmp_path, monkeypatch)
    runtime, binding = service.runtime, service.bindings()[0]
    lease = runtime._ensure_lease(binding)
    attempt = tasks.TaskAttempt(task['identity'], lease, 1, 0)
    _defer(service, task, deadline=True)
    if outcome == 'stale':
        record['running'] = False
        queued = service.retry_room_task('room', task_id=task['identity'].task_id)
        tasks.start_task(service.db_path, task['identity'], lease,
            expected_cancel_generation=queued['cancel_generation'], clock=runtime.clock)
        _defer(service, task, deadline=True)
    service.send(room_id='room', event_id='next', payload={'text': '@two next', 'thread_id': 'next'})
    next_task, = tasks.list_tasks(service.db_path, room_id='room', status='queued')
    tasks.start_task(service.db_path, next_task['identity'], lease, expected_cancel_generation=0, clock=runtime.clock)
    record['running'] = False
    if outcome == 'storage_error':
        def fail(*args):
            raise RuntimeError('terminal storage failed')
        monkeypatch.setattr(runtime, '_complete_acknowledged_stop', fail)
    runtime._wake.clear()
    runtime._on_terminal(binding, attempt, {'status': 'cancelled'})
    assert runtime._wake.is_set()
    saved = tasks.get_task(service.db_path, task['identity'])
    assert saved['status'] == ('failed' if outcome == 'current' else 'deferred')
    if outcome == 'current':
        assert saved['result']['reason_code'] == 'turn_deadline_exceeded'
    assert tasks.get_task(service.db_path, next_task['identity'])['status'] == 'running'
    if outcome == 'storage_error':
        assert any(record.exc_info and 'terminal reconciliation failed' in record.message
                   and 'terminal storage failed' in str(record.exc_info[1]) for record in caplog.records)
