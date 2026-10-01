"""A message sent while the previous discussion settles still gets its turn."""
import time

from gateway import hosted_room_driver as driver, hosted_rooms
from tests.tui_gateway.test_hosted_room_service import _server
from tui_gateway.hosted_room_service import HostedRoomService


def test_a_message_sent_as_the_previous_discussion_settles_is_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    service = HostedRoomService(_server(), db_path=tmp_path / 'state.db')
    service.local_profiles = lambda: ('default', 'helper')
    service.create_room(room_id='room', name='Room', members=[
        {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
        {'member_id': 'helper', 'profile': 'helper', 'handle': 'helper'}])
    binding = service.bindings()[0]
    service.send(room_id='room', event_id='first', payload={'text': '@helper first', 'thread_id': 'thread'})
    (first,) = driver.list_tasks(service.db_path, room_id='room', status='queued')
    lease = driver.acquire_lease(service.db_path, room_id='room', gateway_id=binding.gateway_id,
                                 authority_epoch=binding.authority_epoch, process_generation='test',
                                 ttl_seconds=60, clock=time.time)
    attempt = driver.start_task(service.db_path, first['identity'], lease, expected_cancel_generation=0,
                                clock=time.time)
    driver.settle_task(service.db_path, attempt, settlement_id='failure:first', status='failed',
                       result={'error': 'the member could not answer'}, clock=time.time)
    settle = service._append_room_status

    def settle_after_a_new_message(room, decision):
        # The user's next message lands between planning the settle and logging it.
        hosted_rooms.append_event(
            service.db_path, room_id='room', event_id='user:second', kind='message.user',
            actor={'kind': 'user', 'id': 'desktop'}, payload={'text': '@helper second', 'thread_id': 'thread'},
            authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch)
        settle(room, decision)
    monkeypatch.setattr(service, '_append_room_status', settle_after_a_new_message)
    service.prepare_room(binding)
    kinds = [e['kind'] for e in service._events('room')]
    assert kinds[-3:] == ['turn.failed', 'message.user', 'room.activity'], kinds
    monkeypatch.setattr(service, '_append_room_status', settle)
    service.prepare_room(binding)
    (second,) = driver.list_tasks(service.db_path, room_id='room', status='queued')
    assert second['payload']['source_event_seq'] == kinds.index('message.user', 1) + 1
