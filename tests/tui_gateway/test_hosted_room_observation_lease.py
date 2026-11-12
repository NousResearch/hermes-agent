"""Keep watching admitted turns through transient observation failures."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_agent.gateway import hosted_room_driver as state, hosted_rooms
from hermes_agent.tui_gateway import hosted_room_driver as runtime_module
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, PROFILE, FakeSessionRPC, RecordingTurnLocks, _admit, _identity, _runtime,
    db as db_fixture,
)

db = db_fixture


@pytest.fixture
def observed(db, monkeypatch):
    identity, now, published = _identity(), [100.0], []
    hosted_rooms.append_event(db, room_id=BINDING.room_id, event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'hello'},
        authority_gateway_id=BINDING.gateway_id, authority_epoch=1)
    _admit(db, identity)
    locks = RecordingTurnLocks()
    rpc = FakeSessionRPC(auto_complete=False, required_lock=locks)
    runtime = _runtime(db, rpc, locks, lease_ttl_seconds=30, indeterminate_defer_seconds=5,
                       clock=lambda: now[0], publish_terminal=lambda _, task: published.append(task))
    monkeypatch.setattr(runtime_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    attempt = state.TaskAttempt(identity, runtime._ensure_lease(BINDING), 1, 0)
    o = SimpleNamespace(db=db, identity=identity, now=now, rpc=rpc, runtime=runtime,
                        attempt=attempt, published=published,
                        fail=lambda: now[0] >= 132, action=lambda: None)

    def tick(_timeout):
        assert now[0] < 200, 'polling did not finish'
        assert locks.held_for(PROFILE)
        assert runtime.status()['current_task'] == identity
        now[0] += 1
        o.action()

    monkeypatch.setattr(runtime._wake, 'wait', tick)
    return o


def fail_observation(o, monkeypatch, operation='history'):
    owner = o.runtime if operation == '_report_pending_action' else o.rpc
    original = getattr(owner, operation)

    def observe(*args, **kwargs):
        assert not o.runtime._status_lock.locked()
        if o.fail():
            raise OSError('temporary observation failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, operation, observe)


@pytest.mark.parametrize('operation', ['history', 'info', '_report_pending_action'])
@pytest.mark.parametrize('first_failure', [100, 132])
def test_continuous_failures_retire_exact_attempt(observed, monkeypatch, operation, first_failure):
    o = observed
    o.fail = lambda: o.now[0] >= first_failure
    resolutions = []

    def resolve(*_):
        assert not resolutions, 'an admitted attempt must keep its transport'
        resolutions.append(True)
        return o.rpc

    o.runtime.transport_resolver = resolve
    fail_observation(o, monkeypatch, operation)
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert task['status'] == 'indeterminate'
    assert task['indeterminate_at'] == first_failure + 30
    assert task['execution_generation'] == 1
    assert o.runtime._leases[BINDING.room_id].lease_generation == o.attempt.lease.lease_generation
    assert 'observation failed after submit: temporary observation failure' in o.runtime._last_error
    assert not o.published
    from tests.gateway.test_room_work_transition_capture import snapshot
    assert snapshot(o.db)['tasks'][0]['phase'] == 'indeterminate'


@pytest.mark.parametrize('operation', ['history', 'info', '_report_pending_action'])
def test_flapping_reads_reset_grace_and_settle_once(observed, monkeypatch, operation):
    o = observed
    o.fail = lambda: 132 <= o.now[0] < 140 or 145 <= o.now[0] < 170
    fail_observation(o, monkeypatch, operation)

    def complete():
        assert state.get_task(o.db, o.identity)['status'] == 'running'
        if o.now[0] == 170:
            o.rpc.complete(o.identity.task_id, content='Recovered once')

    o.action = complete
    o.runtime._process_room(BINDING)
    assert o.now[0] == 170
    assert state.get_task(o.db, o.identity)['result']['text'] == 'Recovered once'
    assert [t['status'] for t in o.published] == ['settled']
    assert len([c for c in o.rpc.calls if c[0] == 'submit']) == 1


@pytest.mark.parametrize('finish', [140, 147, 150, 152, 161, 163])
def test_local_callback_at_review_boundaries_settles_once(observed, monkeypatch, finish):
    o = observed
    fail_observation(o, monkeypatch)
    o.action = lambda: o.rpc.complete(o.identity.task_id, content='Exact reply') if o.now[0] == finish else None
    o.runtime._process_room(BINDING)
    if finish > 162:
        assert state.get_task(o.db, o.identity)['status'] == 'indeterminate'
        o.now[0] = finish
        o.rpc.complete(o.identity.task_id, content='Exact reply')
    assert state.get_task(o.db, o.identity)['result']['text'] == 'Exact reply'
    assert [t['status'] for t in o.published] == ['settled']


@pytest.mark.parametrize('control', ['stop', 'deadline'])
def test_stop_and_deadline_still_reach_live_member(observed, monkeypatch, control):
    o = observed
    # Read faults stop when Stop begins; the same member remains live throughout.
    o.fail = lambda: o.now[0] >= 132 and state.get_task(o.db, o.identity)['status'] == 'running'
    fail_observation(o, monkeypatch)
    if control == 'deadline':
        o.runtime.turn_timeout_seconds = 52
    else:
        def stop():
            if o.now[0] == 152:
                state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                        expected_cancel_generation=0, clock=o.runtime.clock)
        o.action = stop
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert task['status'] == ('failed' if control == 'deadline' else 'cancelled')
    assert o.now[0] == 152
    assert [kw['expected_task_id'] for name, kw in o.rpc.calls if name == 'interrupt'] == [o.identity.task_id]
    if control == 'deadline':
        assert task['result']['reason_code'] == 'turn_deadline_exceeded'
        assert [t['status'] for t in o.published] == ['failed']


@pytest.fixture
def retired(observed, monkeypatch):
    fail_observation(observed, monkeypatch)
    observed.runtime._process_room(BINDING)
    assert state.get_task(observed.db, observed.identity)['status'] == 'indeterminate'
    return observed


@pytest.mark.parametrize('fence', ['execution', 'cancel', 'identity'])
def test_retired_callback_checks_exact_attempt(retired, fence):
    o = retired
    other = _identity('other')
    _admit(o.db, other)
    wrong = {'execution': replace(o.attempt, execution_generation=2),
             'cancel': replace(o.attempt, cancel_generation=1),
             'identity': replace(o.attempt, identity=other)}[fence]
    o.runtime._on_terminal(BINDING, wrong, {'status': 'settled', 'text': 'Wrong'})
    assert state.get_task(o.db, o.identity)['status'] == 'indeterminate'
    assert state.get_task(o.db, other)['status'] == 'queued'
    assert not o.published
    o.rpc.complete(o.identity.task_id, content='Exact attempt')
    assert state.get_task(o.db, o.identity)['result']['text'] == 'Exact attempt'
    assert [t['status'] for t in o.published] == ['settled']


@pytest.mark.parametrize('race', ['stop', 'disband', 'error'])
@pytest.mark.parametrize('where', ['_ensure_lease', '_resolve_indeterminate'])
def test_retired_callback_race_always_wakes(retired, monkeypatch, race, where):
    o = retired
    original, calls = getattr(o.runtime, where), []

    def once(*args, **kwargs):
        calls.append(where)
        monkeypatch.setattr(o.runtime, where, original)
        if race == 'error':
            raise RuntimeError('publication unavailable')
        task = state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                      expected_cancel_generation=0, clock=o.runtime.clock)
        state.complete_task_cancel(o.db, o.identity, cancel_id='stop',
                                   expected_cancel_generation=task['cancel_generation'], clock=o.runtime.clock)
        if race == 'disband':
            hosted_rooms.disband_room(o.db, room_id=BINDING.room_id,
                expected_gateway_id=BINDING.gateway_id, expected_epoch=1)
        o.runtime._wake.clear()
        return original(*args, **kwargs)

    monkeypatch.setattr(o.runtime, where, once)
    o.runtime._wake.clear()
    o.runtime._on_terminal(BINDING, o.attempt, {'status': 'settled', 'text': 'Late'})
    assert calls == [where]
    assert o.runtime._wake.is_set()
    assert not o.published


def test_member_recovers_just_after_retirement_with_same_run(retired):
    o = retired
    o.fail = lambda: False
    # Another profile's transport recovers a durable receipt without a local callback.
    o.runtime.rpc = FakeSessionRPC()
    o.runtime.transport_resolver = lambda *_: o.rpc
    for session in o.rpc.states.values():
        session['on_terminal'] = None
    o.now[0] += 1
    o.rpc.complete(o.identity.task_id, content='Recovered history')
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert task['status'] == 'settled'
    assert task['result']['text'] == 'Recovered history'
    assert task['execution_generation'] == 1
    assert len([c for c in o.rpc.calls if c[0] == 'submit']) == 1


@pytest.mark.parametrize('error', [state.StaleLeaseError, state.StaleTaskError,
                                   state.InvalidTaskTransitionError, state.RoomUnavailableError,
                                   'not_admitted', 'refused_before_submit'])
def test_observation_retry_preserves_state_and_admission_errors(observed, monkeypatch, error):
    o, reads = observed, []
    history = o.rpc.history

    def observe(**kwargs):
        reads.append(o.now[0])
        if o.now[0] < 132:
            return history(**kwargs)
        if o.now[0] == 132:
            raise OSError('temporary observation failure')
        exc = RuntimeError('admission proof') if isinstance(error, str) else error('state fence')
        if isinstance(error, str):
            setattr(exc, error, True)
        raise exc

    monkeypatch.setattr(o.rpc, 'history', observe)
    o.runtime._process_room(BINDING)
    assert reads[-1] == 133
    expected = {'not_admitted': 'queued', 'refused_before_submit': 'failed'}.get(error, 'running')
    assert state.get_task(o.db, o.identity)['status'] == expected
    if error in (state.StaleLeaseError, state.StaleTaskError):
        assert 'fenced: state fence' in o.runtime._last_error


def test_grace_holds_worker_and_profile_lock_until_stop(db, monkeypatch):
    import threading
    from contextlib import contextmanager

    identity = _identity()
    _admit(db, identity)
    profile_lock, watching, release = threading.Lock(), threading.Event(), threading.Event()

    @contextmanager
    def turn_lock(_profile):
        with profile_lock:
            yield

    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, turn_lock, lease_ttl_seconds=30)
    history, failures = rpc.history, []

    def observe(**kwargs):
        if state.get_task(db, identity)['status'] == 'running':
            failures.append(True)
            if len(failures) == 2:
                watching.set()
                assert release.wait(10), 'Stop handshake did not arrive'
            raise OSError('temporary read failure')
        return history(**kwargs)

    monkeypatch.setattr(rpc, 'history', observe)
    runtime.start()
    try:
        assert watching.wait(10), 'the room worker abandoned observation'
        assert runtime._room_threads[BINDING.room_id].is_alive()
        assert profile_lock.locked()
        assert runtime.status()['current_task'] == identity
        state.begin_task_cancel(db, identity, cancel_id='stop', expected_cancel_generation=0, clock=runtime.clock)
        release.set()
        from tests.waiting import wait_until
        wait_until(lambda: state.get_task(db, identity)['status'] == 'cancelled', timeout=10)
        assert [kw['expected_task_id'] for name, kw in rpc.calls if name == 'interrupt'] == [identity.task_id]
    finally:
        release.set()
        assert runtime.stop(timeout=10)
    assert not profile_lock.locked()


def test_stop_reaches_member_whose_history_keeps_failing(observed, monkeypatch):
    o = observed
    fail_observation(o, monkeypatch)

    def stop():
        if o.now[0] == 152:
            state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                    expected_cancel_generation=0, clock=o.runtime.clock)

    o.action = stop
    o.runtime._process_room(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'cancelled'
    assert [kw['expected_task_id'] for name, kw in o.rpc.calls if name == 'interrupt'] == [o.identity.task_id]


def test_expired_lease_during_stop_frees_the_worker(observed, monkeypatch):
    o = observed
    monkeypatch.setattr(o.rpc, 'interrupt', lambda **_: {'interrupted': False})
    kept_polling = []

    def wait(_timeout):
        if o.now[0] >= 500:  # a member call blocked past the lease: this worker must not poll on
            kept_polling.append(True)
            o.runtime._stop.set()
            return
        o.now[0] += 1
        if o.now[0] == 110:
            state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                    expected_cancel_generation=0, clock=o.runtime.clock)
        if o.now[0] == 112:
            o.now[0] = 500

    monkeypatch.setattr(o.runtime._wake, 'wait', wait)
    o.runtime._process_room(BINDING)
    assert not kept_polling
    assert state.get_task(o.db, o.identity)['status'] == 'stopping'
    assert o.runtime.status()['current_task'] is None


def test_every_probe_timing_out_still_defers(db, monkeypatch):
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, None, lease_ttl_seconds=30, indeterminate_defer_seconds=60,
                       clock=lambda: now[0], transport_resolver=lambda *_: rpc)
    monkeypatch.setattr(runtime_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(runtime._wake, 'wait', lambda _: now.__setitem__(0, now[0] + .25))

    def timeout(**_):
        now[0] += 30  # the full transport budget, the same as the lease
        raise TimeoutError('timed out')

    rpc.history = rpc.resolve_exact = timeout
    runtime._run_room_once(BINDING)
    while state.get_task(db, identity)['status'] != 'deferred':
        assert now[0] < 400, 'recovery never deferred the turn'
        now[0] += 5
        runtime._run_room_once(BINDING)


@pytest.mark.parametrize('limit, status, published', [(1830.0, 'settled', ['settled']), (60.0, 'failed', ['failed'])])
def test_turn_active_at_recovery_deadline(retired, monkeypatch, limit, status, published):
    o = retired
    o.rpc.required_lock = None  # a room scan drives Stop outside the turn lock
    o.runtime.turn_timeout_seconds = limit
    monkeypatch.setattr(o.runtime, '_inspect_local_recovery_session',
                        lambda _: runtime_module._RecoveryInspection(terminal=None, active=True, status='working'))
    for _ in range(3):
        o.now[0] += 5
        o.runtime._process_room(BINDING)
    if status == 'settled':
        assert state.get_task(o.db, o.identity)['status'] == 'indeterminate'
        o.runtime._on_terminal(BINDING, o.attempt, {'status': 'settled', 'content': 'done'})
    assert state.get_task(o.db, o.identity)['status'] == status
    assert [t['status'] for t in o.published] == published
    assert len([c for c in o.rpc.calls if c[0] == 'interrupt']) == (status == 'failed')


def test_lost_transport_still_reaches_deferral(retired, monkeypatch):
    o = retired

    def gone(*_):
        raise PermissionError('permission_denied')

    o.runtime.transport_resolver = gone
    o.now[0] += 6
    o.runtime._process_room(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'deferred'


def test_late_reply_after_the_retired_attempts_lease_expired(retired):
    o = retired
    o.now[0] += 500  # the recovery probe waited on the profile lock past the lease
    o.runtime._on_terminal(BINDING, o.attempt, {'status': 'settled', 'content': 'done'})
    assert state.get_task(o.db, o.identity)['status'] == 'settled'
    assert [t['status'] for t in o.published] == ['settled']


def test_stop_of_a_lost_member_lets_go_of_the_worker(observed, monkeypatch):
    o = observed
    fail_observation(o, monkeypatch)

    def refuse(**_):
        raise OSError('owner socket is gone')

    monkeypatch.setattr(o.rpc, 'interrupt', refuse)

    def stop():
        if o.now[0] == 140:
            state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                    expected_cancel_generation=0, clock=o.runtime.clock)

    o.action = stop
    o.runtime._process_room(BINDING)
    assert o.now[0] < 180  # the fixture's tick fails the test if this worker never lets go
    assert o.runtime.status()['current_task'] is None
    assert state.get_task(o.db, o.identity)['status'] == 'stopping'


def test_stop_of_a_member_on_another_daemon_defers(observed, monkeypatch):
    o = observed
    o.rpc.served_elsewhere = True
    o.runtime.transport_resolver = lambda *_: o.rpc
    fail_observation(o, monkeypatch)

    def refuse(**_):
        raise OSError('owner socket is gone')

    monkeypatch.setattr(o.rpc, 'interrupt', refuse)

    def stop():
        if o.now[0] == 140:
            state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                    expected_cancel_generation=0, clock=o.runtime.clock)

    o.action = stop
    o.runtime._process_room(BINDING)
    o.now[0] += 10
    o.runtime._process_room(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'deferred'


def test_stop_probe_outlasting_the_lease_still_defers(observed, monkeypatch):
    o = observed
    o.rpc.served_elsewhere = True
    o.runtime.transport_resolver = lambda *_: o.rpc
    fail_observation(o, monkeypatch)

    def timeout(**_):
        o.now[0] += 30  # the full transport budget, the same as the lease
        raise TimeoutError('timed out')

    def stop():
        if o.now[0] == 140:
            state.begin_task_cancel(o.db, o.identity, cancel_id='stop',
                                    expected_cancel_generation=0, clock=o.runtime.clock)
            monkeypatch.setattr(o.rpc, 'resolve_exact', timeout)

    o.action = stop
    o.runtime._run_room_once(BINDING)
    for _ in range(12):
        if state.get_task(o.db, o.identity)['status'] == 'deferred':
            break
        o.now[0] += 5
        o.runtime._run_room_once(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'deferred'
