"""The lease keeper keeps a watched turn leased while its worker waits on I/O."""

import pytest

from gateway import hosted_room_driver as state
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, RecordingTurnLocks, _admit, _identity, db as db_fixture,
)
from tests.waiting import wait_until

db = db_fixture


def _member_runtime(db, clock, *, member_transport, auto_complete=False, rooms=(BINDING,), member=None, **kwargs):
    member = member or FakeSessionRPC(auto_complete=auto_complete)
    return member, HostedRoomRuntime(
        db_path=db, rooms=rooms, rpc=None if member_transport else member,
        transport_resolver=(lambda _binding, _task: member) if member_transport else None,
        turn_lock=RecordingTurnLocks(), clock=clock, max_concurrent_rooms=1, **kwargs)


def _restarted_mid_turn(db, monkeypatch, clock, **kwargs):
    """The queued turn runs under a process that dies, taking its callback along. Returns the bot and
    the next process's runtime on the gateway route, which leaves a watch after one poll."""
    member, first = _member_runtime(db, clock, member_transport=True)
    monkeypatch.setattr(first._wake, 'wait', lambda _timeout: first._stop.set())
    first._run_room_once(BINDING)
    next(iter(member.states.values()))['on_terminal'] = None
    _member, runtime = _member_runtime(db, clock, member_transport=True, member=member, **kwargs)
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    return member, runtime


@pytest.mark.parametrize('member_transport', [True, False])
@pytest.mark.parametrize('stop_during_read', [False, True])
def test_one_timed_out_read_does_not_mark_a_live_turn_indeterminate(
        db, monkeypatch, member_transport, stop_during_read):
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    member, runtime = _member_runtime(db, lambda: now[0], member_transport=member_transport, lease_ttl_seconds=0.6)
    reads = []

    def slow_first(read):
        def observe(**kwargs):
            reads.append(kwargs)
            if len(reads) == 1:
                for _ in range(2):
                    expiry = runtime._leases[BINDING.room_id].expires_at
                    now[0] += 0.3
                    wait_until(lambda: runtime._leases[BINDING.room_id].expires_at > expiry,
                               timeout=5, what='the keeper to renew during the blocked read')
                if stop_during_read:
                    runtime.cancel(identity, cancel_id='stop-during-read')
                raise TimeoutError('transport call timed out after 30 seconds')
            return read(**kwargs)
        return observe

    # Whichever read the route observes with first waits out its whole transport timeout.
    member.history = slow_first(member.history)
    # Yield after the failed read to check the next worker cycle, without waiting in real time.
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    runtime._process_room(BINDING)
    runtime._stop.clear()
    runtime._process_room(BINDING)

    task = state.get_task(db, identity)
    assert task['status'] == ('cancelled' if stop_during_read else 'running')
    if not stop_during_read:
        assert next(iter(member.states.values()))['active'] is True
        member.complete(identity.task_id, content='Adjustment applied once.')
        assert state.get_task(db, identity)['status'] == 'settled'
    assert len([call for call, _ in member.calls if call == 'submit']) == 1
