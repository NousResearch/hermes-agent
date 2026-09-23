"""Recovery observations must select the requested profile, not a same-key neighbour."""
from pathlib import Path
import threading

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError


@pytest.fixture
def scoped_server(tmp_path, monkeypatch):
    from hermes_cli import profiles
    from tui_gateway import server

    home = tmp_path / '.hermes'
    named = home / 'profiles' / 'ops'
    named.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: home)
    monkeypatch.setattr(server, '_hermes_home', str(home))
    monkeypatch.setattr(server, '_served_profile_homes', {named})
    records = {
        'root-runtime': {'session_key': 'same-stored-key', 'profile_home': None,
                         'history_lock': threading.Lock(), 'running': False},
        'ops-runtime': {'session_key': 'same-stored-key', 'profile_home': str(named),
                        'history_lock': threading.Lock(), 'running': True,
                        '_hosted_room_task': {'task_id': 'task', 'execution_generation': 7}},
    }
    monkeypatch.setattr(server, '_sessions', records)
    monkeypatch.setattr(server, '_sessions_lock', threading.Lock())
    pending_reads = []
    monkeypatch.setattr(server, '_pending_approval_request_payload', lambda key: pending_reads.append(key))
    assert Path(server.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2]
    try:
        yield server, pending_reads
    finally:
        # These are observation-only records, with no worker or agent to close.
        # Remove them before the suite's real session-lifecycle finalizer runs.
        records.clear()


def test_same_stored_key_is_observed_in_requested_profile(scoped_server):
    server, _ = scoped_server
    rpc = HostedRoomServerRPC(server)
    runtime = object.__new__(HostedRoomRuntime)
    runtime.pending_action = None
    task = {'identity': TaskIdentity('room', 'task', 'thread', 'turn'),
            'execution_generation': 7, 'payload': {'target_profile': 'ops'}}

    assert rpc.info(profile='default', session_id='same-stored-key', source='bot_room')['active'] is False
    # Exercise the real Recovery consumer too: a foreign idle record must not turn
    # the named live attempt into an idle observation usable for Retry/Stop/defer.
    inspection = runtime._inspect_session(rpc, task, 'same-stored-key', read_history=False)
    assert inspection.active is True
    assert rpc.info(profile='ops', session_id='same-stored-key', source='bot_room') == {
        'active': True, 'task_id': 'task', 'execution_generation': 7}
    assert rpc.info(profile='default', session_id='same-stored-key', source='bot_room')['active'] is False


def test_foreign_runtime_id_and_missing_profile_are_not_idle_proof(scoped_server):
    server, pending_reads = scoped_server
    rpc = HostedRoomServerRPC(server)
    with pytest.raises(HostedRoomSessionError, match='profile'):
        rpc.info(profile='ops', session_id='root-runtime', source='bot_room')
    with pytest.raises(server.ProfileUnavailableError):
        rpc.info(profile='deleted-profile', session_id='same-stored-key', source='bot_room')
    assert pending_reads == []
    assert rpc.info(profile='ops', session_id='absent-in-ops', source='bot_room') == {
        'active': False, 'task_id': None, 'execution_generation': None}
