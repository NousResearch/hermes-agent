"""Service share and raced publication preserve the original durable task."""
import time
import sqlite3
from contextlib import closing
from types import SimpleNamespace

from gateway import hosted_rooms, hosted_room_driver as driver
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from tui_gateway.hosted_room_service import HostedRoomService


def service_at(db):
    service = HostedRoomService(SimpleNamespace(), db_path=db)
    service.local_profiles = lambda: ('default', 'ops')
    return service


def test_service_share_retains_bytes_and_reuses_frozen_task(tmp_path):
    db = tmp_path / 'state.db'
    service = service_at(db)
    service.create_room(room_id='room', name='Room', members=[
        {'member_id': 'default', 'profile': 'default', 'handle': 'hermes'},
        {'member_id': 'ops', 'profile': 'ops', 'handle': 'ops'}])
    store = HostedRoomAttachmentStore(db)
    item = store.put(room_id='room', upload_id='upload', kind='file', name='proof.txt',
                     mime='application/octet-stream', data=b'\x00exact shared bytes\xff')
    manifest = [{k: item[k] for k in ('attachment_id', 'kind', 'name', 'mime', 'size')}]
    event = service.send(room_id='room', event_id='share', payload={
        'text': '@hermes inspect', 'thread_id': 'thread', 'attachments': manifest})
    before = driver.list_tasks(db, room_id='room')[0]
    assert before['payload']['attachments'][0]['event_id'] == event['event_id']
    with closing(sqlite3.connect(db)) as conn, conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'hosted_room_policy_%'").fetchall()
        for (table,) in tables:
            conn.execute(f'DROP TABLE "{table}"')
    restored = service_at(db)
    restored.prepare_room(restored.bindings()[0])
    assert driver.list_tasks(db, room_id='room')[0]['payload'] == before['payload']
    assert HostedRoomAttachmentStore(db, clock=lambda: time.time() + 999999).read_viewer(
        room_id='room', attachment_id=item['attachment_id'], event_id='share',
        authority_gateway_id=service.bindings()[0].gateway_id, authority_epoch=1).data == b'\x00exact shared bytes\xff'


def test_publication_cursor_race_retries_without_new_task(tmp_path, monkeypatch):
    db = tmp_path / 'state.db'
    service = service_at(db)
    service.create_room(room_id='room', name='Room', members=[
        {'member_id': 'default', 'profile': 'default', 'handle': 'hermes'},
        {'member_id': 'ops', 'profile': 'ops', 'handle': 'ops'}])
    service.send(room_id='room', event_id='share', payload={'text': '@hermes inspect', 'thread_id': 'thread'})
    binding = service.bindings()[0]
    task = driver.list_tasks(db, room_id='room')[0]
    lease = driver.acquire_lease(db, room_id='room', gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation='test', ttl_seconds=30, clock=time.time)
    attempt = driver.start_task(db, task['identity'], lease, expected_cancel_generation=0, clock=time.time)
    driver.settle_task(db, attempt, settlement_id='result', status='settled', result={'text':'done'}, clock=time.time)
    append = hosted_rooms.append_event
    raced = []
    def interleave(path, **kwargs):
        if kwargs['kind'] == 'message.member' and not raced:
            raced.append(True)
            append(path, room_id='room', event_id='race', kind='room.activity',
                actor={'kind':'gateway','id':binding.gateway_id}, payload={'status':'idle'},
                authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch)
        return append(path, **kwargs)
    monkeypatch.setattr(hosted_rooms, 'append_event', interleave)
    service.prepare_room(binding)
    assert not any(e['kind'] == 'message.member' for e in service._events('room'))
    service.prepare_room(binding)
    assert sum(e['kind'] == 'message.member' for e in service._events('room')) == 1
    assert driver.get_task_for_turn(db, task['identity'])['execution_generation'] == attempt.execution_generation
