"""Group chat recovery must ask a bot for live status even when history is unreadable."""

import logging

import pytest

from gateway import hosted_room_driver as state
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, RecordingTurnLocks, _admit, _identity,
)
from tests.tui_gateway.test_hosted_room_observation_lease import (
    observed as observed_fixture, db as db_fixture, fail_observation,
)

observed, db = observed_fixture, db_fixture


def test_member_that_reports_itself_running_is_not_deferred_as_unavailable(observed, monkeypatch):
    o = observed
    # The gateway uses a separate transport per member; the dashboard uses runtime.rpc.
    o.runtime.rpc = None
    o.runtime.transport_resolver = lambda *_: o.rpc
    fail_observation(o, monkeypatch)
    o.runtime._process_room(BINDING)
    task = state.get_task(o.db, o.identity)
    assert task['status'] == 'indeterminate'
    o.now[0] = task['indeterminate_at'] + 61
    o.runtime._process_room(BINDING)
    assert next(iter(o.rpc.states.values()))['active'] is True
    assert state.get_task(o.db, o.identity)['status'] != 'deferred'


@pytest.mark.parametrize('member_transport', [True, False])
def test_retry_waits_until_history_shows_whether_the_bot_already_answered(db, monkeypatch, member_transport):
    """Recovery may ask for live status without history, but Retry runs the turn again: it must
    first see in history that the uncertain attempt left no reply."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member = FakeSessionRPC(auto_complete=False)
    runtime = HostedRoomRuntime(
        db_path=db, rooms=[BINDING], rpc=None if member_transport else member,
        transport_resolver=(lambda *_: member) if member_transport else None,
        turn_lock=RecordingTurnLocks(), clock=lambda: now[0])
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    runtime._process_room(BINDING)
    runtime._stop.clear()
    lease = runtime._leases[BINDING.room_id]
    state.mark_attempt_indeterminate(db, state.TaskAttempt(identity, lease, 1, 0), lease, clock=runtime.clock)
    next(iter(member.states.values()))['on_terminal'] = None  # after a restart only history has the reply
    member.complete(identity.task_id, content='Adjustment applied once.')
    member.history_failures = 1
    with pytest.raises(RuntimeError, match='history read failed'):
        runtime.retry_indeterminate(identity)
    assert state.get_task(db, identity)['status'] == 'indeterminate'
    assert runtime.retry_indeterminate(identity)['result']['text'] == 'Adjustment applied once.'
    assert len([call for call, _ in member.calls if call == 'submit']) == 1


def test_recovery_logs_a_failed_read_by_type_only(db, monkeypatch, caplog):
    """Exception text can hold prompt text, and a peer repeats this every cycle."""
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member = FakeSessionRPC(auto_complete=False)
    runtime = HostedRoomRuntime(
        db_path=db, rooms=[BINDING], rpc=member, turn_lock=RecordingTurnLocks(), clock=lambda: now[0])
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    runtime._process_room(BINDING)
    session_id, = member.states

    def leaking(**_kwargs):
        raise OSError('PRIVATE_PROMPT_TEXT')

    member.history = leaking
    with caplog.at_level(logging.DEBUG, logger='tui_gateway.hosted_room_driver'):
        runtime._inspect_session(member, state.get_task(db, identity), session_id, read_history=True)
    assert 'OSError' in caplog.text and 'PRIVATE_PROMPT_TEXT' not in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
