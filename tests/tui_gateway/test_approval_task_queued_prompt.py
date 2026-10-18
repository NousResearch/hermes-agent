"""Busy local prompts keep their server-stamped submitter through the drain."""
from types import SimpleNamespace

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


def test_authenticated_busy_submit_keeps_distinct_origins(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / 'state.db')
    monkeypatch.setattr(server, '_get_db', lambda: db)
    monkeypatch.setattr(server, '_schedule_agent_build', lambda _sid: None)
    monkeypatch.setattr(server, '_schedule_session_cap_enforcement', lambda: None)
    monkeypatch.setattr(server, '_register_session_cwd', lambda _session: None)
    transport = server._stdio_transport
    token = bind_transport(transport)
    try:
        created = server.handle_request({'id': 'c', 'method': 'session.create',
                                         'params': {'cols': 96, 'source': 'tui'}})
        assert 'result' in created, created
        sid = created['result']['session_id']
        session = server._sessions[sid]
        session['agent'] = SimpleNamespace()
        with session['history_lock']:
            session['running'] = True
            server._start_inflight_turn(session, 'first running prompt')
        for index, text in enumerate(('Edit only alpha.py', 'Edit only beta.py')):
            reply = server.handle_request({'id': f'q{index}', 'method': 'prompt.submit',
                                           'params': {'session_id': sid, 'text': text, 'queued': True}})
            assert reply['result']['status'] == 'queued', reply
        queue = [session['queued_prompt'], *session.get('queued_prompts', [])]
        assert len(queue) == 2  # merged text would falsely merge two distinct consents
        assert [entry['approval_origin'][3] for entry in queue] == ['Edit only alpha.py', 'Edit only beta.py']
        assert queue[0]['approval_origin'][0] == queue[1]['approval_origin'][0] == 'local:stdio'
        assert queue[0]['approval_origin'][1] != queue[1]['approval_origin'][1]
        drained = []
        monkeypatch.setattr(server, '_run_prompt_submit',
                            lambda _rid, _sid, _session, text, **kwargs:
                            drained.append((text, kwargs.get('approval_origin'))))
        with session['history_lock']:
            session['running'] = False
            server._clear_inflight_turn(session)
        assert server._drain_queued_prompt('drain', sid, session)
        assert drained[0][1] == queue[0]['approval_origin']
    finally:
        reset_transport(token)
        db.close()
