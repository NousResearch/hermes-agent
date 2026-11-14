"""Exact late receipts survive deferral and lease expiry without reviving replaced work."""

from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, RecordingTurnLocks, _admit, _identity, db as db_fixture,
)

db = db_fixture


def _running(db, monkeypatch, *, rooms=(BINDING,)):
    identity, now, published = _identity(), [100.0], []
    _admit(db, identity)
    member = FakeSessionRPC(auto_complete=False)
    runtime = HostedRoomRuntime(
        db_path=db, rooms=rooms, rpc=member,
        turn_lock=RecordingTurnLocks(), clock=lambda: now[0],
        publish_terminal=lambda _binding, task: published.append(task))
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())
    runtime._process_room(BINDING)
    runtime._stop.clear()
    task = lambda: state.get_task(db, identity)
    lease = runtime._leases.get(BINDING.room_id)
    return SimpleNamespace(db=db, runtime=runtime, identity=identity, now=now, member=member,
                           published=published, task=task, lease=lease)


def _defer(room):
    attempt = state.TaskAttempt(room.identity, room.lease, 1, 0)
    state.mark_attempt_indeterminate(room.db, attempt, room.lease, clock=room.runtime.clock)
    state.defer_indeterminate_task(
        room.db, room.identity, room.lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason='member_unavailable', clock=room.runtime.clock)


def _assert_reply(room, text):
    assert (room.task()['status'], (room.task()['result'] or {}).get('text')) == ('settled', text)
    assert [task['result']['text'] for task in room.published] == [text]
    with pytest.raises(state.InvalidTaskTransitionError):
        room.runtime.retry_indeterminate(room.identity)
    assert room.task()['execution_generation'] == 1


def test_reply_that_lands_after_deferral_is_published(db, monkeypatch):
    room = _running(db, monkeypatch)
    _defer(room)
    room.member.complete(room.identity.task_id, content='Adjustment 8 applied.')
    _assert_reply(room, 'Adjustment 8 applied.')


def test_dashboard_keeps_receipt_until_it_can_commit_under_a_current_lease(db, monkeypatch):
    room = _running(db, monkeypatch)
    room.now[0] += 31
    state.acquire_lease(
        db, room_id=BINDING.room_id, gateway_id=BINDING.gateway_id, authority_epoch=1,
        process_generation='another-process', ttl_seconds=30, clock=room.runtime.clock)
    room.member.complete(room.identity.task_id, content='Saved during takeover.')
    assert room.task()['status'] == 'running'
    assert room.published == []
    room.member.history = lambda **_kwargs: []
    room.now[0] += 31
    room.runtime._process_room(BINDING)
    _assert_reply(room, 'Saved during takeover.')


@pytest.mark.parametrize('winner', ['retry', 'stop', 'discard', 'authority'])
def test_late_reply_cannot_replace_a_new_attempt_or_a_settled_control(db, monkeypatch, winner):
    room = _running(db, monkeypatch)
    _defer(room)
    if winner == 'retry':
        room.runtime.retry_indeterminate(room.identity)
        state.start_task(db, room.identity, room.lease, expected_cancel_generation=0,
                         clock=room.runtime.clock)
    elif winner == 'stop':
        state.cancel_task(db, room.identity, cancel_id='stop', expected_cancel_generation=0,
                          clock=room.runtime.clock)
    elif winner == 'discard':
        state.reopen_deferred_task(db, room.identity, room.lease, expected_execution_generation=1,
                                  expected_cancel_generation=0, clock=room.runtime.clock)
        state.resolve_indeterminate_cancellation(
            db, room.identity, room.lease, expected_execution_generation=1,
            expected_cancel_generation=0, cancel_id='discard', clock=room.runtime.clock)
    else:
        hosted_rooms.claim_authority(
            db, room_id=BINDING.room_id, expected_gateway_id=BINDING.gateway_id, expected_epoch=1,
            new_gateway_id='gateway-b', event_id='transfer', now=room.now[0])
    before = room.task()
    room.member.complete(room.identity.task_id, content='Old reply must not win.')
    assert room.task() == before
    assert room.published == []


@pytest.mark.parametrize('winner', ['retry', 'stop', 'discard'])
def test_control_winning_between_receipt_read_and_commit_keeps_its_fence(db, monkeypatch, winner):
    room = _running(db, monkeypatch)
    _defer(room)
    ensure_lease = room.runtime._ensure_lease
    won = []

    def control_wins(binding):
        lease = ensure_lease(binding)
        if not won:
            if winner == 'retry':
                state.requeue_deferred_task(
                    db, room.identity, lease, expected_execution_generation=1,
                    expected_cancel_generation=0, clock=room.runtime.clock)
                state.start_task(db, room.identity, lease, expected_cancel_generation=0,
                                 clock=room.runtime.clock)
            elif winner == 'stop':
                state.cancel_task(db, room.identity, cancel_id='stop', expected_cancel_generation=0,
                                  clock=room.runtime.clock)
            else:
                state.reopen_deferred_task(
                    db, room.identity, lease, expected_execution_generation=1,
                    expected_cancel_generation=0, clock=room.runtime.clock)
                state.resolve_indeterminate_cancellation(
                    db, room.identity, lease, expected_execution_generation=1,
                    expected_cancel_generation=0, cancel_id='discard', clock=room.runtime.clock)
            won.append(room.task())
        return lease

    monkeypatch.setattr(room.runtime, '_ensure_lease', control_wins)
    room.member.complete(room.identity.task_id, content='A receipt from the previous attempt.')
    assert won and room.task() == won[0]
    assert room.published == []


def _held_by_another_process(room):
    room.now[0] += 31
    state.acquire_lease(
        room.db, room_id=BINDING.room_id, gateway_id=room.lease.gateway_id, authority_epoch=1,
        process_generation='another-process', ttl_seconds=30, clock=room.runtime.clock)


def test_retry_that_finds_a_kept_reply_publishes_it(db, monkeypatch):
    room = _running(db, monkeypatch)
    _defer(room)
    _held_by_another_process(room)
    room.member.complete(room.identity.task_id, content='Finished while the lease was held.')
    assert room.task()['status'] == 'deferred'
    room.now[0] += 31
    result = room.runtime.retry_indeterminate(room.identity, execution_generation=1)
    assert result['status'] == 'settled'
    _assert_reply(room, 'Finished while the lease was held.')


@pytest.mark.parametrize('lost', ['authority', 'listing'])
def test_kept_reply_is_dropped_once_this_gateway_no_longer_runs_the_group_chat(db, monkeypatch, lost):
    listed = [BINDING]
    room = _running(db, monkeypatch, rooms=lambda: tuple(listed))
    _defer(room)
    _held_by_another_process(room)
    if lost == 'authority':
        hosted_rooms.claim_authority(
            db, room_id=BINDING.room_id, expected_gateway_id=BINDING.gateway_id, expected_epoch=1,
            new_gateway_id='gateway-b', event_id='transfer', now=room.now[0])
    room.member.complete(room.identity.task_id, content='Finished after the move.')
    if lost == 'listing':
        listed.clear()
    room.now[0] += 31
    for _ in range(2):
        room.runtime._run_cycle()
    assert not room.runtime._terminal_receipts
    assert (room.task()['status'], room.published) == ('deferred', [])


def test_reply_after_an_unconfirmed_stop_deferred_the_turn_is_published(db, monkeypatch):
    """Stop never reached the bot, so its run finished: the reply wins, as it does while stopping."""
    room = _running(db, monkeypatch)
    state.begin_task_cancel(db, room.identity, cancel_id='stop', expected_cancel_generation=0,
                            clock=room.runtime.clock)
    state.defer_unconfirmed_stop(db, room.identity, room.lease, expected_execution_generation=1,
                                 expected_cancel_generation=1, reason='member_unavailable',
                                 clock=room.runtime.clock)
    room.member.complete(room.identity.task_id, content='Finished before the Stop arrived.')
    _assert_reply(room, 'Finished before the Stop arrived.')


def test_a_late_reply_does_not_hide_what_other_bots_said_meanwhile(tmp_path, monkeypatch):
    from tests.gateway.test_hosted_room_lead import MEMBERS, PROFILES, ROOM_ID
    from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server
    from tui_gateway.hosted_room_service import HostedRoomService

    class HeldRPC(_FakeRPC):
        """Other bots answer at once; ada's reply is held until the test releases it."""
        def __init__(self):
            super().__init__()
            self.held, self.prompts = {}, []

        def submit(self, *, profile, prompt, on_terminal, **kwargs):
            self.prompts.append((profile, prompt))
            if profile in self.held:
                self.held[profile] = on_terminal
                return {'accepted': True}
            return super().submit(profile=profile, prompt=prompt, on_terminal=on_terminal, **kwargs)

    db = tmp_path / 'state.db'
    service = HostedRoomService(_server(), db_path=db)
    rpc = service.rpc = service.runtime.rpc = HeldRPC()
    rpc.held['ada'] = None
    runtime = service.runtime
    service.local_profiles = lambda: PROFILES
    service.create_room(room_id=ROOM_ID, name='Kernel', members=MEMBERS)
    service.send(room_id=ROOM_ID, event_id='u1', payload={'text': 'Why is the build slow?', 'thread_id': 't1'})
    binding = service.bindings()[0]
    monkeypatch.setattr(runtime._wake, 'wait', lambda _timeout: runtime._stop.set())

    def process():
        for _ in range(12):
            runtime._stop.clear()
            runtime._process_room(binding)

    runtime._stop.clear()
    runtime._process_room(binding)
    ada, = state.list_tasks(db, room_id=ROOM_ID, status='running')
    assert ada['payload']['target_profile'] == 'ada'
    lease = runtime._leases[ROOM_ID]
    state.mark_attempt_indeterminate(db, state.TaskAttempt(ada['identity'], lease, 1, 0), lease, clock=runtime.clock)
    state.defer_indeterminate_task(db, ada['identity'], lease, expected_execution_generation=1,
                                   expected_cancel_generation=0, reason='member_unavailable', clock=runtime.clock)
    process()  # the others answer while ada's turn is deferred
    events = hosted_rooms.read_events(db, room_id=ROOM_ID, limit=hosted_rooms.MAX_LOG_LIMIT)['events']
    said = next(e['payload']['text'] for e in events if e['kind'] == 'message.member')
    rpc.held.pop('ada')({'status': 'settled', 'text': 'Ada, late: the cache was cold.'})
    process()
    rpc.prompts.clear()
    service.send(room_id=ROOM_ID, event_id='u2', payload={'text': 'Next step?', 'thread_id': 't1'})
    process()
    ada_prompt = next(prompt for profile, prompt in rpc.prompts if profile == 'ada')
    assert said in ada_prompt
    assert 'Next step?' in ada_prompt
    assert 'the cache was cold' not in ada_prompt  # its own reply is in its own session
    assert 'Why is the build slow?' not in ada_prompt  # it answered that one


def test_invalid_callback_keeps_the_existing_failure_outcome(db, monkeypatch):
    room = _running(db, monkeypatch)
    callback = next(iter(room.member.states.values()))['on_terminal']
    callback({'status': 'settled', 'settlement_id': 'x' * 129, 'text': 'Malformed receipt.'})
    assert room.task()['status'] == 'failed'
