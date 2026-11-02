"""Preview forwarding executes the actual owner, never a local unowned substitute."""
import asyncio
from types import SimpleNamespace

from gateway.hosted_rooms import create_room, local_authority_gateway_id
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway import methods_groups_forward, owner_forward, transport
from tui_gateway import server


def test_forward_executes_owner_and_denies_untrusted_or_unowned(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('model:\n  default: fixture\n')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), instance_id='owner', events={},
                                    epoch=begin_runtime_epoch(db, instance_id='test'))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        identity = {'user_id': 'owner', 'provider': 'basic'}
        connection = AuthorityConnection(authority, object(), identity)
        service.authorize_room(connection.actor.subject, 'room', create=True)
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=local_authority_gateway_id(),
                    members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                             {'member_id': 'two', 'profile': 'two', 'handle': 'two'}])
        endpoint, calls = object(), []
        monkeypatch.setattr(methods_groups_forward, 'owner_endpoint', lambda home: endpoint)

        def forward(home, selected, method, params):
            calls.append((home, selected, method, params))
            reply = asyncio.run(connection.dispatch({'id': 1, 'method': method, 'params': params}))
            return reply['result']
        monkeypatch.setattr(owner_forward, 'forward_rpc', forward)
        with db._read_ctx() as conn:
            before = tuple(conn.iterdump())
        token = transport.bind_transport(SimpleNamespace(auth_identity=identity))
        try:
            result = server._methods['groups.preview'](1, {'room_id': 'room', 'text': '@two'})
            assert [m['member_id'] for m in result['result']['members']] == ['two']
            assert calls[0][2] == 'groups.preview'
            monkeypatch.setattr(methods_groups_forward, 'owner_endpoint', lambda home: None)
            assert server._methods['groups.preview'](2, {'room_id': 'room', 'text': '@two'})['error']
        finally:
            transport.reset_transport(token)
        token = transport.bind_transport(SimpleNamespace(auth_identity=None))
        try:
            assert server._methods['groups.preview'](3, {'room_id': 'room', 'text': '@two'})['error']
        finally:
            transport.reset_transport(token)
        assert len(calls) == 1
        with db._read_ctx() as conn:
            assert tuple(conn.iterdump()) == before


def test_forward_selected_and_handed_owners_with_refusals(tmp_path, monkeypatch, caplog):
    import sqlite3
    from gateway.hosted_rooms import default_db_path
    from hermes_cli.gateway_client import GatewayRPCError
    root = tmp_path / 'state'
    secondary = root / 'profiles' / 'helper'
    secondary.mkdir(parents=True)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(root))
    (root / 'config.yaml').write_text('model:\n  default: fixture\n')
    (secondary / 'config.yaml').write_text('model:\n  default: fixture\n')
    monkeypatch.setattr(server, '_current_profile_name', lambda: 'default')
    monkeypatch.setattr(server, '_profile_home', lambda profile: secondary if profile == 'helper' else None)
    gateway_id = local_authority_gateway_id()
    members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
               {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]
    identity = {'user_id': 'owner', 'provider': 'basic'}
    with SessionDB(root / 'state.db') as root_db, SessionDB(secondary / 'state.db') as other_db:
        owners = {}
        for home, db, name in [(root, root_db, 'handed'), (secondary, other_db, 'selected')]:
            authority = SimpleNamespace(db=db, profile_id=str(home), instance_id='owner', events={},
                                        epoch=begin_runtime_epoch(db, instance_id='test'))
            service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
            monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
            owner = AuthorityConnection(authority, object(), identity)
            service.authorize_room(owner.actor.subject, name, create=True)
            create_room(db.db_path, room_id=name, name=name, authority_gateway_id=gateway_id, members=members)
            if name == 'selected':
                create_room(db.db_path, room_id='denied', name='denied', authority_gateway_id=gateway_id, members=members)
            owners[home] = owner
        store = default_db_path()
        create_room(store, room_id='handed', name='handed', authority_gateway_id=gateway_id, members=members)
        create_room(store, room_id='local', name='Local', authority_gateway_id=gateway_id, members=members)
        with sqlite3.connect(store) as conn:
            conn.execute('UPDATE hosted_rooms SET authority_gateway_id=? WHERE room_id=?', (gateway_id + ':gateway-store', 'handed'))
        with sqlite3.connect(store) as conn:
            standalone_before = tuple(conn.iterdump())
        snapshots = {}
        for home, db in [(root, root_db), (secondary, other_db)]:
            with db._read_ctx() as conn: snapshots[home] = tuple(conn.iterdump())
        endpoint = object()
        calls = []
        monkeypatch.setattr(methods_groups_forward, 'owner_endpoint', lambda home: endpoint)
        def forward(home, selected, method, params):
            calls.append((home, selected, method, dict(params)))
            reply = asyncio.run(owners[home].dispatch({'id': 1, 'method': method, 'params': params}))
            if 'error' in reply:
                raise GatewayRPCError(reply['error']['message'], code=reply['error']['code'])
            return reply['result']
        monkeypatch.setattr(owner_forward, 'forward_rpc', forward)
        token = transport.bind_transport(SimpleNamespace(auth_identity=identity))
        try:
            text = 'private-sentinel @two'
            result = server._methods['groups.preview'](1, {'profile': 'helper', 'room_id': 'selected', 'text': text})
            assert [m['member_id'] for m in result['result']['members']] == ['two']
            assert calls[-1] == (secondary, endpoint, 'groups.preview', {'profile': 'helper', 'room_id': 'selected', 'text': text})
            refused = server._methods['groups.preview'](2, {'profile': 'helper', 'room_id': 'denied', 'text': text})
            assert 'error' in refused and text not in str(refused)
            result = server._methods['groups.preview'](3, {'profile': 'helper', 'room_id': 'handed', 'text': text})
            assert [m['member_id'] for m in result['result']['members']] == ['two']
            assert calls[-1] == (root, endpoint, 'groups.preview', {'room_id': 'handed', 'text': text})
            count = len(calls)
            assert 'error' in server._methods['groups.preview'](4, {'profile': 'helper', 'room_id': 'local', 'text': text})
            assert len(calls) == count
            monkeypatch.setattr(methods_groups_forward, 'owner_endpoint', lambda home: None)
            assert 'error' in server._methods['groups.preview'](5, {'profile': 'helper', 'room_id': 'handed', 'text': text})
            assert len(calls) == count
        finally:
            transport.reset_transport(token)
        token = transport.bind_transport(SimpleNamespace(auth_identity={'user_id': 'uid:999999', 'provider': 'local'}))
        monkeypatch.setattr(methods_groups_forward, 'owner_endpoint', lambda home: (_ for _ in ()).throw(AssertionError('foreign principal must not discover owner')))
        try:
            assert 'error' in server._methods['groups.preview'](6, {'profile': 'helper', 'room_id': 'selected', 'text': text})
            assert len(calls) == count
        finally:
            transport.reset_transport(token)
        assert text not in caplog.text
        with sqlite3.connect(store) as conn: assert tuple(conn.iterdump()) == standalone_before
        for home, db in [(root, root_db), (secondary, other_db)]:
            with db._read_ctx() as conn: assert tuple(conn.iterdump()) == snapshots[home]


def test_preview_never_initializes_an_unready_standalone_store(tmp_path, monkeypatch):
    import sqlite3
    from gateway.hosted_rooms import default_db_path
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = default_db_path()
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE sentinel (value TEXT)')
        conn.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before = path.read_bytes()
    monkeypatch.setattr(methods_groups_forward, 'owner_endpoint', lambda home: (_ for _ in ()).throw(AssertionError('must not forward')))
    token = transport.bind_transport(SimpleNamespace(auth_identity={'user_id': 'owner', 'provider': 'basic'}))
    try:
        result = server._methods['groups.preview'](1, {'room_id': 'room', 'text': 'draft'})
        assert result['error']
    finally:
        transport.reset_transport(token)
    assert path.read_bytes() == before
    assert not (tmp_path / 'install_id').exists()
