"""The standalone viewer can list and save the same published version."""
import base64
import sqlite3

import tui_gateway.server as srv
from tests.tui_gateway.test_groups_forward_parity import MEMBERS, home as home_fixture, _gateway

home = home_fixture


def test_standalone_catalog_uses_the_downloads_group_scope(home, monkeypatch):
    _gateway(monkeypatch, 'absent')
    for name in ('files', 'empty'):
        assert 'result' in srv._methods['groups.create'](1, dict(room_id=name, name=name, members=MEMBERS))
    uploaded = srv._methods['groups.attachment.upload'](2, dict(
        room_id='files', upload_id='version-one', name='report.txt', kind='file', mime='text/plain',
        data_base64=base64.b64encode(b'version one\n').decode()))['result']
    manifest = {key: uploaded[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}
    assert 'result' in srv._methods['groups.send'](3, dict(
        room_id='files', event_id='share', payload=dict(thread_id='files', text='Review', attachments=[manifest])))
    page = srv._methods['groups.attachment.list'](4, dict(room_id='files'))['result']
    item, = page['items']
    assert item['attachment_id'] == manifest['attachment_id'] and item.get('available', True)
    saved = srv._methods['groups.attachment.download'](5, dict(
        room_id='files', event_id=item['event_id'], attachment_id=item['attachment_id']))['result']
    assert base64.b64decode(saved['data_base64']) == b'version one\n'
    assert srv._methods['groups.attachment.list'](6, dict(room_id='empty'))['result']['items'] == []
    from gateway.hosted_rooms import default_db_path
    with sqlite3.connect(default_db_path()) as conn:
        conn.execute('DELETE FROM hosted_room_attachments')
        conn.execute('DELETE FROM hosted_room_attachment_blobs')
    missing, = srv._methods['groups.attachment.list'](7, dict(room_id='files'))['result']['items']
    assert missing['available'] is False
    refused = srv._methods['groups.attachment.download'](8, dict(
        room_id='files', event_id=item['event_id'], attachment_id=item['attachment_id']))
    assert refused['error']['data']['reason'] == 'attachment_unavailable'
