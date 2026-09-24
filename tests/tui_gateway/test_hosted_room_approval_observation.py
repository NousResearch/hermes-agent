"""Real approval publication must not cross same-key profile observations."""
from pathlib import Path
import threading

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError


@pytest.fixture
def approval_server(tmp_path, monkeypatch):
    from hermes_cli import profiles
    from tools import approval, approval_context
    from tui_gateway import server, server_requests

    home = tmp_path / '.hermes'
    named = home / 'profiles' / 'ops'
    named.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: home)
    monkeypatch.setattr(server, '_hermes_home', str(home))
    monkeypatch.setattr(server, '_served_profile_homes', {named})
    records = {
        'root-runtime': {'session_key': 'same-key', 'profile_home': None,
                         'history_lock': threading.Lock(), 'running': False},
        'ops-runtime': {'session_key': 'same-key', 'profile_home': str(named),
                        'history_lock': threading.Lock(), 'running': False},
    }
    monkeypatch.setattr(server, '_sessions', records)
    monkeypatch.setattr(server, '_sessions_lock', threading.Lock())
    monkeypatch.setattr(approval, '_gateway_queues', {})
    monkeypatch.setattr(server_requests, '_open', {})
    frames, events = [], []
    monkeypatch.setattr(server_requests, '_write', frames.append)
    monkeypatch.setattr(server_requests, '_emit', lambda *args: events.append(args))
    monkeypatch.setattr(approval_context, '_fire_approval_hook', lambda *args, **kwargs: None)
    root = Path(__file__).resolve().parents[2]
    for module in (server, approval, server_requests):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(root)
    try:
        yield server, approval, server_requests
    finally:
        records.clear()


def _inspect(rpc, profile):
    runtime = object.__new__(HostedRoomRuntime)
    runtime.pending_action = None
    task = {'identity': TaskIdentity('room', 'task', 'thread', 'turn'),
            'execution_generation': 7, 'payload': {'target_profile': profile}}
    return runtime._inspect_session(rpc, task, 'same-key', read_history=False)


@pytest.mark.parametrize('owner,foreign,sid', [
    ('default', 'ops', 'root-runtime'),
    ('ops', 'default', 'ops-runtime'),
])
def test_real_pending_request_is_visible_only_to_its_runtime(approval_server, owner, foreign, sid):
    from tools.approval_gateway_wait import _await_gateway_decision

    server, approval, requests = approval_server
    rpc = HostedRoomServerRPC(server)
    observed = {}

    def notify(data):
        server._emit_approval_request(sid, data)
        try:
            observed['owner'] = rpc.info(profile=owner, session_id='same-key', source='bot_room')
            observed['foreign'] = rpc.info(profile=foreign, session_id='same-key', source='bot_room')
            observed['owner_active'] = _inspect(rpc, owner).active
            observed['foreign_active'] = _inspect(rpc, foreign).active
        finally:
            approval.resolve_gateway_approval('same-key', 'deny', request_id=data['request_id'])

    decision = _await_gateway_decision('same-key', notify, {
        'command': 'synthetic-owner-only', 'description': 'private synthetic approval',
        'pattern_key': 'synthetic', 'request_id': 'synthetic-request',
    })
    assert decision['resolved'] is True and decision['choice'] == 'deny'
    assert observed['owner']['pending_approval']['request_id'] == 'synthetic-request'
    assert observed['owner_active'] is True
    assert 'pending_approval' not in observed['foreign']
    # A key-only queue cannot prove whose pending wait it is. Preserve uncertainty,
    # rather than treating omitted foreign details as permission to retry or stop.
    assert observed['foreign']['status'] == 'unknown'
    assert observed['foreign_active'] is True
    assert not approval.has_blocking_approval('same-key')
    assert requests.open_requests(sid) == []
    assert rpc.info(profile=owner, session_id='same-key', source='bot_room')['active'] is False


def test_queue_without_owned_request_remains_unknown_not_idle(approval_server):
    from tools.approval_gateway_wait import _await_gateway_decision

    server, approval, _ = approval_server
    rpc = HostedRoomServerRPC(server)
    observed = {}

    def notify(data):
        try:
            observed.update(rpc.info(profile='ops', session_id='same-key', source='bot_room'))
            observed['inspection_active'] = _inspect(rpc, 'ops').active
        finally:
            approval.resolve_gateway_approval('same-key', 'deny', request_id=data['request_id'])

    decision = _await_gateway_decision('same-key', notify, {
        'command': 'unrouted-synthetic-command', 'pattern_key': 'synthetic',
    })
    assert decision['resolved'] is True and decision['choice'] == 'deny'
    assert 'pending_approval' not in observed
    assert observed['status'] == 'unknown'
    assert observed['inspection_active'] is True
    assert not approval.has_blocking_approval('same-key')


def test_replacement_during_request_read_cannot_publish_observation(approval_server, monkeypatch):
    server, _, _ = approval_server
    read_requests = server._open_requests

    def replace_owner(sid):
        snapshot = read_requests(sid)
        server._sessions[sid] = dict(server._sessions[sid])
        return snapshot

    monkeypatch.setattr(server, '_open_requests', replace_owner)
    with pytest.raises(HostedRoomSessionError, match='owner changed'):
        HostedRoomServerRPC(server).info(profile='ops', session_id='same-key', source='bot_room')


def test_owned_compute_mirror_is_read_without_reentering_history_lock(approval_server):
    server, _, requests = approval_server
    server._sessions['ops-runtime']['_compute_host_open_request'] = {
        'id': 'synthetic-mirror', 'method': 'approval',
        'params': {'session_id': 'ops-runtime', 'request_id': 'mirror-approval',
                   'command': 'synthetic-mirror-only', 'choices': ['once', 'deny']},
    }
    assert requests.open_requests('ops-runtime') == []
    result = HostedRoomServerRPC(server).info(profile='ops', session_id='same-key', source='bot_room')
    assert result['pending_approval']['request_id'] == 'mirror-approval'
    assert _inspect(HostedRoomServerRPC(server), 'ops').active is True
    other = HostedRoomServerRPC(server).info(profile='default', session_id='same-key', source='bot_room')
    assert 'pending_approval' not in other
