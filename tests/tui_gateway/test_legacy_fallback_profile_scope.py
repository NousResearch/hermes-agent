"""B11: a ticket bound to a secondary profile reads ITS profile through the legacy session fallback.

``session.most_recent`` / ``session.workspace.move`` / ``session.foreign.*`` open ``_profile_db``
(the LAUNCH store when no ``profile`` is named) and ``session.active_list`` lists every legacy
session in the process, so a profile-bound Ink/CLI ticket got the launch profile's latest session id
and title (and Ink's auto-resume resumed it) and another profile's live rows."""
import threading
import types

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from tui_gateway import server
    launch = tmp_path / 'root'
    bee = launch / 'profiles' / 'bee'
    bee.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(launch))
    monkeypatch.setattr(server, '_hermes_home', launch)
    monkeypatch.setattr(server, '_db', None)
    return server, launch, bee


def test_profile_bound_ticket_reads_only_its_own_profile(homes, monkeypatch):
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from hermes_state import SessionDB
    from tui_gateway.ws_legacy_fallback import dispatch_legacy, legacy_fallback_allowed
    server, launch, bee = homes
    with SessionDB(launch / 'state.db') as db:
        db.create_session('launch-sid', 'cli')
        db.set_session_title('launch-sid', 'LAUNCH_PROFILE_SECRET_TITLE')
    with SessionDB(bee / 'state.db') as db:
        db.create_session('bee-sid', 'cli')
        db.set_session_title('bee-sid', 'BEE_TITLE')
    for sid, home in (('live-launch', None), ('live-bee', str(bee))):
        monkeypatch.setitem(server._sessions, sid, {
            'session_key': sid, 'history': [{'role': 'user', 'content': 'PRIVATE ' + sid}],
            'history_lock': threading.Lock(), 'profile_home': home, 'created_at': 1, 'last_active': 1})

    def call(profile_id, method):
        actor = types.SimpleNamespace(capabilities=frozenset(_PURPOSE_CAPABILITIES['interactive']),
                                      profile_id=str(profile_id))
        assert legacy_fallback_allowed(actor, method)
        written, done = [], threading.Event()
        transport = types.SimpleNamespace(write=lambda frame: written.append(frame) or done.set() or True)
        reply = dispatch_legacy(server, {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': {}}, transport, actor)
        # A long handler answers from the pool through the bound transport.
        return reply if reply is not None else (done.wait(10) and written[0])

    recent = call(bee, 'session.most_recent')['result']
    assert (recent['session_id'], recent['title']) == ('bee-sid', 'BEE_TITLE'), recent
    assert [row['id'] for row in call(bee, 'session.active_list')['result']['sessions']] == ['live-bee']
    # The launch profile's own ticket is unchanged (main's single-profile behaviour).
    assert call(launch, 'session.most_recent')['result']['session_id'] == 'launch-sid'
    assert {row['id'] for row in call(launch, 'session.active_list')['result']['sessions']} == {
        'live-launch', 'live-bee'}
