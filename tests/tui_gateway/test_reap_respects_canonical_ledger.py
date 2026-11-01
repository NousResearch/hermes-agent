"""A viewer's reap never stamps a session ended while the gateway's ledger holds its turn (#140642).

A remote viewer of a Desktop session disconnected mid-turn; its process's ws-orphan reaper ran
``_finalize_session`` and stamped the row ``ws_orphan_reap`` while the owning gateway was still
running the admitted turn. The owner then read its own live route as stale (#54878).
"""
import threading
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tui_gateway.server import _teardown_session


def _session(tmp_path, monkeypatch, source, session_id):
    from run_agent import AIAgent
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(session_id, source=source)
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent._owns_session_db = True
    agent._end_session_on_close = True
    agent._session_messages = []
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._memory_manager = None
    session = {'agent': agent, 'history': [{'role': 'user', 'content': 'x'}], 'history_lock': None,
               'session_key': session_id, 'profile_home': str(home), 'source': source}
    return home / 'state.db', db, session, agent


@pytest.mark.parametrize('claimed', [True, False], ids=['started', 'queued'])
@pytest.mark.parametrize('source', ['gui', 'tui'])
def test_reap_leaves_row_open_while_the_ledger_holds_its_turn(tmp_path, monkeypatch, source, claimed):
    sid = f'{source}-live'
    path, db, session, agent = _session(tmp_path, monkeypatch, source, sid)
    epoch = begin_runtime_epoch(db, instance_id='gateway')
    admit_session_input(db, epoch=epoch, principal_id='owner', session_id=sid, request_id='r1',
                        payload={'text': 'long turn'})
    if claimed:
        claim_session_input(db, epoch=epoch, session_id=sid)

    _teardown_session(session, end_reason='ws_orphan_reap')

    check = SessionDB(db_path=path)
    try:
        row = check.get_session(sid)
        assert (row['ended_at'], row['end_reason']) == (None, None)
        assert agent._end_session_on_close is False  # agent.close() must not end it either
    finally:
        check.close()


def test_reap_still_ends_an_idle_row(tmp_path, monkeypatch):
    path, db, session, _agent = _session(tmp_path, monkeypatch, 'tui', 'tui-idle')
    epoch = begin_runtime_epoch(db, instance_id='gateway')
    db.create_session('other', source='tui')
    admit_session_input(db, epoch=epoch, principal_id='owner', session_id='other', request_id='r1',
                        payload={'text': 'another session'})

    _teardown_session(session, end_reason='ws_orphan_reap')

    check = SessionDB(db_path=path)
    try:
        assert check.get_session('tui-idle')['end_reason'] == 'ws_orphan_reap'
    finally:
        check.close()



@pytest.mark.asyncio
async def test_reap_of_a_segment_of_an_adopted_conversation_respects_the_owner_ledger(tmp_path, monkeypatch):
    """An adopted (pre-canonical) conversation keeps its legacy id as owner and as every compression
    segment's ``chat_id``, not a ``local-`` id; the reap must still check the owner's ledger through
    the receipt's lineage (review of #140971). Real adoption, compression publish and admission."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local_recovery import reopen_local_session
    from gateway.session_local_title import resolve_titled_session
    from tui_gateway import server
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('old-tui', source='tui', model='frozen', cwd=str(tmp_path))
    db.append_message('old-tui', 'user', 'old question')
    db.end_session('old-tui', 'tui_close')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='review')
    actor = Principal('native', str(tmp_path), frozenset({'session:create', 'session:read'}), 'sock')
    authority._native_legacy_transports = {actor.transport_id: actor}
    ref = resolve_titled_session(authority, actor, 'old-tui')
    reopen_local_session(authority, ref)
    assert db.try_acquire_compression_lock(ref.session_id, 'review')
    try:
        db.publish_compression_child(parent_session_id=ref.session_id, child_session_id='child', source='tui',
                                     messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='review')
    finally:
        db.release_compression_lock(ref.session_id, 'review')
    assert db.get_session('child')['chat_id'] == 'old-tui'
    admit_session_input(db, epoch=authority.epoch, principal_id=actor.subject, session_id=ref.session_id,
                        request_id='running', payload={'text': 'continue'})
    claim_session_input(db, epoch=authority.epoch, session_id=ref.session_id)

    @contextmanager
    def session_db(_session):
        yield db
    monkeypatch.setattr(server, '_session_db', session_db)
    viewer = {'agent': SimpleNamespace(session_id='child', _end_session_on_close=True),
              'session_key': 'child', 'history': [], 'profile_home': str(tmp_path), 'source': 'tui'}
    server._finalize_session(viewer, end_reason='ws_orphan_reap')
    assert db.get_session('child')['ended_at'] is None
