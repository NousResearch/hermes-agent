"""Legacy closeout uses the same validated primary submit and exact caller transport."""
import threading

import pytest

from hermes_cli.conversation_closeout import CloseoutStore
from hermes_state import SessionDB
from tests.waiting import wait_until
from tui_gateway import server


class Peer:
    _closed = False
    def __init__(self):
        self.frames = []
    def write(self, frame):
        self.frames.append(frame)
        return True


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(server, '_sessions', {})
    monkeypatch.setattr(server, '_CLOSEOUT_WAIT_SECONDS', 0, raising=False)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('stored', source='desktop')
        peer = Peer()
        ready = threading.Event()
        ready.set()
        session = {'session_key': 'stored', 'source': 'desktop', 'profile_home': str(tmp_path),
            'history_lock': threading.RLock(), 'history': [], 'running': False, 'agent_ready': ready,
            'transport': peer, 'viewers': {peer: 1}}
        server._sessions['runtime'] = session
        monkeypatch.setattr(server, '_get_db', lambda: db)
        monkeypatch.setattr(server, '_viewer_detach_is_settled', lambda *args: False)
        yield db, peer, session


def test_rpc_submits_one_review_without_replacing_transport_identity(env, monkeypatch):
    db, peer, session = env
    calls = []
    def primary(rid, params):
        from tui_gateway.transport import current_transport
        assert current_transport() is peer
        assert params['session_id'] == 'runtime' and params['queued'] is True
        operation = params['text'].split('[Hermes closeout: ', 1)[1].split(']', 1)[0]
        calls.append(operation)
        store = CloseoutStore(db)
        assert store.claim_capture(operation, 'stored')
        store.record_capture(operation, 'stored', memory='skipped', tasks='reconciled')
        return server._ok(rid, {'status': 'streaming'})
    monkeypatch.setitem(server._methods, 'prompt.submit', primary)
    request = {'id': 1, 'method': 'session.closeout', 'params': {
        'session_id': 'runtime', 'request_id': 'archive-request', 'intent': 'new'}}
    result = server.dispatch(request, peer)
    if result is None:
        assert wait_until(lambda: any(frame.get('id') == 1 for frame in peer.frames), timeout=5)
        result = next(frame for frame in peer.frames if frame.get('id') == 1)
    assert 'result' in result, result
    operation = result['result']['closeout_id']
    assert wait_until(lambda: CloseoutStore(db).read(operation)['archived'], timeout=5)
    assert calls == [operation] and not CloseoutStore(db).read(operation)['settled']
    assert server._session_transport_contains(session, peer)
    assert db.get_session('stored')['archived']


def test_foreign_viewer_cannot_close_a_live_conversation(env):
    db, _peer, _session = env
    foreign = Peer()
    request = {'id': 2, 'method': 'session.closeout', 'params': {
        'session_id': 'runtime', 'request_id': 'foreign-request', 'intent': 'archive'}}
    response = server.dispatch(request, foreign)
    if response is None:
        assert wait_until(lambda: bool(foreign.frames), timeout=5)
        response = next(frame for frame in foreign.frames if frame.get('id') == 2)
    assert 'error' in response
    assert not db.get_session('stored')['archived']


def test_restart_reads_receipt_without_replaying_primary_review(env):
    db, peer, _session = env
    store = CloseoutStore(db)
    old = {'logical_session_id': 'stored', 'stored_session_id': 'stored', 'runtime_id': 'runtime', 'profile': 'default'}
    record, _ = store.prepare(old, 'original-request', 'new', 'stored:0')
    store.claim_review(record['closeout_id'])
    server._sessions.clear()
    response = server.dispatch({'id': 3, 'method': 'session.closeout', 'params': {
        'session_id': 'runtime', 'request_id': 'original-request', 'intent': 'new'}}, peer)
    if response is None:
        assert wait_until(lambda: any(frame.get('id') == 3 for frame in peer.frames), timeout=5)
        response = next(frame for frame in peer.frames if frame.get('id') == 3)
    assert response['result']['archived']
    assert response['result']['destinations'] == {'memory': 'unknown', 'tasks': 'unknown'}
    assert not response['result']['settled']


def test_actual_prompt_rpc_rejects_public_closeout_nonce_before_build(env):
    db, peer, _session = env
    store = CloseoutStore(db)
    old = {'logical_session_id': 'stored', 'stored_session_id': 'stored', 'runtime_id': 'runtime', 'profile': 'default'}
    record, _ = store.prepare(old, 'closing', 'archive', 'stored:0')
    response = server.dispatch({'id': 4, 'method': 'prompt.submit', 'params': {
        'session_id': 'runtime', 'text': 'archive\n\n[Hermes closeout: ' + record['closeout_id'] + ']'}}, peer)
    assert response['error']['code'] == 4092
    assert response['error']['data']['reason'] == 'conversation_closing'
