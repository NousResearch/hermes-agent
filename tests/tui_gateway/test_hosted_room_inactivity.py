"""A group chat turn's limit counts from the bot's last progress, timed by the home's own clock."""

import json
import math
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as state
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tui_gateway.hosted_room_service import _hosted_room_turn_timeout_seconds
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, _admit, _identity, _runtime,
)
from tests.tui_gateway.test_hosted_room_observation_lease import (
    observed as observed_fixture, retired as retired_fixture, db as db_fixture, fail_observation,
)
from tests.tui_gateway.test_hosted_room_slow_reads import _member_runtime, _restarted_mid_turn

observed, retired, db = observed_fixture, retired_fixture, db_fixture


@pytest.mark.parametrize('value', ['0', '-1', 'nan'])
def test_agent_timeout_zero_disables_the_group_chat_limit_as_documented(monkeypatch, value):
    monkeypatch.setenv('HERMES_AGENT_TIMEOUT', value)
    assert math.isinf(_hosted_room_turn_timeout_seconds())


def test_agent_timeout_counts_inactivity_not_wall_clock(observed):
    o = observed
    o.runtime.turn_timeout_seconds = 60
    original = o.rpc.info
    o.rpc.info = lambda **kwargs: {**original(**kwargs), 'progress_at': o.now[0],
                                   'execution_generation': 1, 'tool': 'terminal'}
    o.action = lambda: o.rpc.complete(o.identity.task_id, content='Long but active turn done.') if o.now[0] == 190 else None
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert (task['status'], (task['result'] or {}).get('text')) == ('settled', 'Long but active turn done.')


@pytest.mark.parametrize('last_change', [None, 100, 110])
def test_stalled_turn_expires_from_its_last_progress_or_start_whatever_the_bots_clock(observed, last_change):
    o = observed
    o.rpc.required_lock = None
    o.runtime.turn_timeout_seconds = 60
    original = o.rpc.info
    # The bot's clock runs three hours ahead of the home's, and it reports progress until last_change.
    o.rpc.info = lambda **kwargs: {**original(**kwargs), 'execution_generation': 1, **(
        {} if last_change is None else {'progress_at': 3 * 3600 + min(o.now[0], last_change)})}
    o.runtime._process_room(BINDING)  # the fixture's tick fails once 100 polls pass without an outcome
    task = state.get_task(o.db, o.identity)
    assert task['status'] == 'failed'
    assert task['result']['reason_code'] == 'turn_deadline_exceeded'
    assert o.now[0] == (last_change or 100) + 60
    assert [kwargs['expected_task_id'] for name, kwargs in o.rpc.calls if name == 'interrupt'] == [o.identity.task_id]

def _with_limit(o, monkeypatch, limit, *, member_transport):
    """The observed fixture's runtime, built again with this limit on the chosen route."""
    o.runtime._release_idle_leases()
    if member_transport:
        o.rpc = FakeSessionRPC(auto_complete=False)
    o.runtime = HostedRoomRuntime(
        db_path=o.db, rooms=[BINDING], rpc=None if member_transport else o.rpc,
        transport_resolver=(lambda *_: o.rpc) if member_transport else None, turn_lock=o.runtime.turn_lock,
        clock=o.runtime.clock, lease_ttl_seconds=30, indeterminate_defer_seconds=5,
        turn_timeout_seconds=limit)

    def tick(_timeout):
        assert o.now[0] < 200, 'polling did not finish'
        o.now[0] += 1

    monkeypatch.setattr(o.runtime._wake, 'wait', tick)
    return o


@pytest.mark.parametrize('member_transport', [True, False])
@pytest.mark.parametrize('limit', [60, 0, -1])
def test_a_watched_turn_that_went_uncertain_keeps_its_recent_progress_until_it_finishes(
        observed, monkeypatch, member_transport, limit):
    o = _with_limit(observed, monkeypatch, limit, member_transport=member_transport)
    original = o.rpc.info
    # The bot keeps streaming (its frame times run on its own clock) while its reads fail.
    o.rpc.info = lambda **kwargs: {**original(**kwargs), 'progress_at': o.now[0] - 5000,
                                   'execution_generation': 1}
    fail_observation(o, monkeypatch)
    o.runtime._process_room(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'indeterminate'
    o.rpc.required_lock = None
    o.now[0] += 10
    o.runtime._process_room(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'indeterminate'
    o.rpc.complete(o.identity.task_id, content='Recovered active turn.')
    assert state.get_task(o.db, o.identity)['status'] == 'settled'


@pytest.mark.parametrize('member_transport', [True, False])
def test_recovery_without_progress_preserves_the_start_based_limit(retired, member_transport):
    o = retired
    o.rpc.required_lock = None
    o.runtime.turn_timeout_seconds = 60
    if member_transport:
        o.runtime.rpc = None
        o.runtime.transport_resolver = lambda *_: o.rpc
    o.now[0] += 10
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert task['status'] == 'failed'
    assert task['result']['reason_code'] == 'turn_deadline_exceeded'


def test_progress_for_another_attempt_does_not_extend_the_turn(observed):
    o = observed
    o.runtime.turn_timeout_seconds = 10
    o.rpc.info = lambda **kwargs: {'active': True, 'task_id': o.identity.task_id,
                                   'execution_generation': 2, 'progress_at': o.now[0]}
    o.runtime._process_room(BINDING)
    assert state.get_task(o.db, o.identity)['status'] == 'failed'
    assert o.now[0] == 110


def test_dashboard_frame_progress_reaches_the_inactivity_watchdog(observed, monkeypatch):
    import threading
    from collections import OrderedDict
    from tui_gateway import hosted_room_member_activity as activity
    from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC

    o = observed
    o.runtime.turn_timeout_seconds = 60
    monkeypatch.setattr(activity, '_progress', OrderedDict())
    monkeypatch.setattr(activity, 'time', SimpleNamespace(time=lambda: o.now[0]))
    record = {'history_lock': threading.Lock(), 'running': True, '_hosted_room_task': {
        'room_id': BINDING.room_id, 'task_id': o.identity.task_id, 'execution_generation': 1}}
    server = SimpleNamespace(_sessions={'session-1': record}, _sessions_lock=threading.Lock())
    rpc = HostedRoomServerRPC(server)
    monkeypatch.setattr(o.rpc, 'info', rpc.info)

    def stream_frame():
        activity.project_room_member_activity({'method': 'event', 'params': {
            'session_id': 'session-1', 'type': 'message.delta', 'payload': {'text': 'working'}}},
            server._sessions)
        if o.now[0] == 190:
            o.rpc.complete(o.identity.task_id, content='Active dashboard turn done.')

    o.action = stream_frame
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert (task['status'], (task['result'] or {}).get('text')) == ('settled', 'Active dashboard turn done.')
    record['_hosted_room_task']['execution_generation'] = 2
    assert 'progress_at' not in rpc.info(profile='ops', session_id='session-1', source='bot_room')


def _ticking(runtime, monkeypatch, now, step, until):
    """Move the home's clock on every poll; end the watch once it reaches ``until``."""
    def tick(_timeout):
        now[0] += step
        if now[0] >= until:
            runtime._stop.set()

    monkeypatch.setattr(runtime._wake, 'wait', tick)


@pytest.mark.parametrize('uncertain', [True, False])
def test_a_busy_turn_seen_first_after_a_restart_is_not_stopped_at_once(db, monkeypatch, uncertain):
    """A long, healthy turn: the first probe of the new process, 41 minutes in, reports it active
    and streaming a moment ago, whether the turn was uncertain by then or not."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member, runtime = _restarted_mid_turn(db, monkeypatch, lambda: now[0], turn_timeout_seconds=1830)
    now[0] += 2500
    if uncertain:
        state.recover_room(db, runtime._ensure_lease(BINDING), clock=runtime.clock)
    info = member.info
    member.info = lambda **kwargs: {**info(**kwargs), 'execution_generation': 1, 'progress_at': now[0] - 3}
    runtime._run_room_once(BINDING)
    assert state.get_task(db, identity)['status'] == ('indeterminate' if uncertain else 'running')
    assert not [call for call, _ in member.calls if call == 'interrupt']


def test_the_first_frame_of_a_watched_turn_counts_as_activity(db, monkeypatch):
    """Polls saw no frame for 95 s, so a frame at 95 s is activity, not a baseline."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member, runtime = _member_runtime(db, lambda: now[0], member_transport=True, turn_timeout_seconds=100)
    info = member.info
    member.info = lambda **kwargs: {**info(**kwargs), 'execution_generation': 1,
                                    **({'progress_at': 7777.0} if now[0] >= 195 else {})}
    _ticking(runtime, monkeypatch, now, 1, 250)
    runtime._run_room_once(BINDING)
    assert state.get_task(db, identity)['status'] == 'running', runtime._last_error


def test_waiting_for_the_profile_lock_does_not_spend_the_turn_limit(db, monkeypatch):
    """Two group chats share a bot: the limit starts at the submit, not when the row started."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member = FakeSessionRPC(auto_complete=False)

    @contextmanager
    def busy_profile(_profile):
        now[0] += 90  # the other group chat's turn holds the bot's lock (bot_mode.turn_wait_seconds)
        yield

    runtime = HostedRoomRuntime(db_path=db, rooms=[BINDING], rpc=member, turn_lock=busy_profile,
                                clock=lambda: now[0], lease_ttl_seconds=1000, turn_timeout_seconds=100)

    def tick(_timeout):
        now[0] += 1
        if now[0] == 240:  # 50 s after its submit
            member.complete(identity.task_id, content='Done inside its own limit.')

    monkeypatch.setattr(runtime._wake, 'wait', tick)
    runtime._process_room(BINDING)
    assert state.get_task(db, identity)['status'] == 'settled', state.get_task(db, identity)['result']


def _watched_with_reports(db, monkeypatch, now, report, *, limit=0):
    """Watch one turn while the bot gives ``report(identity)`` as its status, 5 s per poll, up to 500 s."""
    identity = _identity()
    _admit(db, identity)
    member, runtime = _member_runtime(db, lambda: now[0], member_transport=True, lease_ttl_seconds=30,
                                      turn_timeout_seconds=limit)
    member.info = lambda **_kwargs: report(identity)
    _ticking(runtime, monkeypatch, now, 5, now[0] + 500)
    runtime._run_room_once(BINDING)
    return state.get_task(db, identity)['status'], runtime


@pytest.mark.parametrize('report', [
    {'active': False, 'status': 'unknown', 'execution_generation': 1},
    {'active': False, 'status': 'idle', 'task_id': None, 'execution_generation': None},
    {'active': False, 'task_id': None},  # the dashboard's answer for a session that is gone
])
@pytest.mark.parametrize('limit', [0, 3600])
def test_a_turn_whose_bot_lost_it_goes_uncertain_after_one_lease(db, monkeypatch, report, limit):
    """Recovery takes it from there. With the limit off nothing else could ever end this watch."""
    now = [100.0]
    status, _runtime = _watched_with_reports(
        db, monkeypatch, now, lambda identity: {'task_id': identity.task_id, **report}, limit=limit)
    assert status == 'indeterminate'
    assert now[0] == 130


@pytest.mark.parametrize('report', [
    {'active': True, 'status': 'queued', 'task_id': 'an-earlier-turn', 'execution_generation': 3},  # gateway FIFO
    {'active': True, 'status': 'queued', 'task_id': 'task-1', 'execution_generation': 1},
    {'active': True, 'status': 'waiting_for_approval', 'pending_approval': {'request_id': 'r', 'choices': ['once']}},
    {'running': True, 'task_id': None},  # a dashboard turn before its task is named
])
def test_a_queued_or_waiting_turn_is_not_taken_for_lost(db, monkeypatch, report):
    status, runtime = _watched_with_reports(db, monkeypatch, [100.0], lambda _identity: report)
    assert status == 'running', runtime._last_error


def test_a_bot_slow_to_report_its_turn_active_is_not_taken_for_lost(db, monkeypatch):
    now = [100.0]
    status, runtime = _watched_with_reports(db, monkeypatch, now, lambda identity: {
        'active': now[0] % 50 >= 25, 'task_id': identity.task_id})
    assert status == 'running', runtime._last_error


@pytest.mark.parametrize('progress', [None, 7777.0])
def test_a_restart_does_not_exempt_a_quiet_active_turn_from_the_limit(db, monkeypatch, progress):
    """The home restarted while another daemon or a peer still reports the turn active."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member, runtime = _restarted_mid_turn(db, monkeypatch, lambda: now[0], turn_timeout_seconds=100)
    info = member.info
    member.info = lambda **kwargs: {**info(**kwargs), 'execution_generation': 1,
                                    **({} if progress is None else {'progress_at': progress})}
    runtime._run_room_once(BINDING)  # first seen at the start: nothing to stop yet
    assert state.get_task(db, identity)['status'] == 'running'
    now[0] += 200  # never a new frame, and the old process's lease has lapsed
    runtime._stop.clear()
    runtime._run_room_once(BINDING)
    task = state.get_task(db, identity)
    assert (task['status'], (task['result'] or {}).get('reason_code')) == ('failed', 'turn_deadline_exceeded')
    assert [kwargs['expected_task_id'] for name, kwargs in member.calls if name == 'interrupt'] == [identity.task_id]


def test_a_restart_leaves_a_turn_the_old_process_still_holds(db, monkeypatch):
    """Only the lease holder may stop it: this process may not act while that lease is live."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member, runtime = _restarted_mid_turn(db, monkeypatch, lambda: now[0], turn_timeout_seconds=10)
    now[0] += 20
    runtime._run_room_once(BINDING)
    assert state.get_task(db, identity)['status'] == 'running'
    assert not [call for call, _ in member.calls if call == 'interrupt']


@pytest.mark.parametrize('uncertain', [False, True])
def test_an_older_peers_active_turn_survives_a_home_restart(db, monkeypatch, uncertain):
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member, runtime = _restarted_mid_turn(db, monkeypatch, lambda: now[0], turn_timeout_seconds=100)
    runtime.remote_member = lambda *_: True
    now[0] += 2500
    if uncertain:
        state.recover_room(db, runtime._ensure_lease(BINDING), clock=runtime.clock)
    # Older peers report that the run is active, but cannot report frame timestamps.
    for _ in range(3):
        runtime._stop.clear()
        runtime._run_room_once(BINDING)
        now[0] += 200
    assert state.get_task(db, identity)['status'] == ('indeterminate' if uncertain else 'running')
    assert not [call for call, _ in member.calls if call == 'interrupt']


def test_a_room_no_longer_listed_loses_its_progress_memory(db, monkeypatch):
    identity, now, listed = _identity(), [100.0], [BINDING]
    _admit(db, identity)
    _member, runtime = _member_runtime(db, lambda: now[0], member_transport=True, rooms=lambda: tuple(listed))
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    runtime._run_room_once(BINDING)
    assert runtime._attempt_progress
    listed.clear()
    runtime._stop.clear()
    runtime._run_cycle()
    assert not runtime._attempt_progress


def test_a_bad_row_cannot_keep_stop_from_recovering(db, monkeypatch):
    identity, bad, now = _identity(), _identity('task-2'), [100.0]
    _admit(db, identity)
    _admit(db, bad)
    member, runtime = _member_runtime(db, lambda: now[0], member_transport=True)
    info = member.info
    member.info = lambda **kwargs: {**info(**kwargs), 'progress_at': 7777.0}  # so the runtime remembers the attempt
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    runtime._run_room_once(BINDING)
    runtime._stop.clear()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_driver_tasks SET payload_json='not json' WHERE task_id=?", (bad.task_id,))
    runtime.cancel(identity, cancel_id='stop', acknowledge=False)  # the worker's pass acts on it
    assert state.get_task(db, identity)['status'] == 'stopping'
    runtime._run_room_once(BINDING)
    assert state.get_task(db, identity)['status'] == 'cancelled', runtime._last_error
    assert next(iter(member.states.values()))['active'] is False


def test_deadline_stop_finished_with_the_limit_turned_off_stores_strict_json(db):
    identity, clock = _identity(), lambda: 100.0
    _admit(db, identity)
    # A process with a finite limit began this deadline Stop, then the gateway restarted with
    # HERMES_AGENT_TIMEOUT=0 before the bot acknowledged it.
    lease = state.acquire_lease(db, room_id=BINDING.room_id, gateway_id=BINDING.gateway_id,
                                authority_epoch=1, process_generation='old', ttl_seconds=30, clock=clock)
    state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    stopping = state.begin_task_cancel(db, identity, cancel_id='deadline:1', expected_cancel_generation=0,
                                       clock=clock)
    state.release_lease(db, lease, clock=clock)
    runtime = _runtime(db, FakeSessionRPC(auto_complete=False), clock=clock, turn_timeout_seconds=0)
    runtime._complete_acknowledged_stop(BINDING, stopping, runtime._ensure_lease(BINDING))
    with sqlite3.connect(db) as conn:
        raw, = conn.execute('SELECT result_json FROM hosted_room_driver_tasks WHERE task_id=?',
                            (identity.task_id,)).fetchone()
    assert json.loads(raw, parse_constant=lambda name: pytest.fail(f'{name} is not JSON'))[
        'reason_code'] == 'turn_deadline_exceeded'
