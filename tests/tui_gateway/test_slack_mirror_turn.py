"""The real prompt worker, not a source-string assertion, drives the Slack mirror."""
import contextlib
import threading
import types
import pytest

from tui_gateway import server
from tui_gateway import slack_mirror


@pytest.mark.parametrize('display_kind, row_surface, session_surface, expected', [
    (None, 'desktop', '', ['user', 'assistant']),
    (None, '', 'desktop', []),
    ('auto_continue', 'desktop', 'desktop', []),
    ('hidden', 'desktop', 'desktop', []),
])
def test_prompt_worker_delivers_accepted_user_then_final(
    monkeypatch, tmp_path, display_kind, row_surface, session_surface, expected
):
    calls = []
    agent = types.SimpleNamespace(session_id='s1', clear_interrupt=lambda: None)
    session = dict(source='desktop', session_key='s1', agent=agent, history=[],
                   history_lock=threading.Lock(), running=True, attached_images=[],
                   client_surface=session_surface,
                   _submit_user_row={'content': 'Hello', '_row_id': 11, '_client_surface': row_surface})
    monkeypatch.setattr(server, '_ensure_session_db_row', lambda session: True)
    monkeypatch.setattr(server, '_admit_prompt_turn', lambda *args: ([], agent))
    monkeypatch.setattr(server, '_resolve_session_platform', lambda: 'tui')
    monkeypatch.setattr(server, '_session_db', lambda session: contextlib.nullcontext('fake-db'))
    monkeypatch.setattr(server, '_record_turn_marker', lambda *args, **kwargs: '')
    monkeypatch.setattr(server, '_retire_turn_marker', lambda *args: None)
    monkeypatch.setattr(server, '_prepare_turn_input', lambda *args: ('Hello', 'Hello', 80, None))
    monkeypatch.setattr(server, '_invoke_agent', lambda sid, session, st, *a: setattr(st, 'result', {'final_response': 'Answer'}))
    monkeypatch.setattr(server, '_absorb_turn_result', lambda *args: None)
    monkeypatch.setattr(server, '_complete_turn_payload', lambda *args: (
        {'persisted_turn': {'complete': True, 'final_assistant_row_id': 12}}, 'Answer', 'complete'))
    monkeypatch.setattr(server, '_finish_turn', lambda *args: None)
    monkeypatch.setattr(server, '_after_complete_turn', lambda *args: None)
    monkeypatch.setattr(server, '_goal_followup_after_turn', lambda *args: None)
    monkeypatch.setattr(server, '_emit_settled_session_info', lambda *args: None)
    monkeypatch.setattr(server, '_emit', lambda *args: None)
    monkeypatch.setattr(server, '_start_session_work', lambda fn, **kw: fn() or True)
    monkeypatch.setattr(server, '_routing_provenance_db', lambda session: contextlib.nullcontext(None))
    monkeypatch.setattr(server, '_run_post_turn_followups', lambda *args: None)
    monkeypatch.setattr(server, '_profile_runtime_scope_tokens', lambda *args: None)
    monkeypatch.setattr(server, '_session_profile_runtime_scope', lambda session: contextlib.nullcontext())
    monkeypatch.setattr(server, '_release_hosted_room_turn_slot', lambda session: None)
    monkeypatch.setattr(server, '_clear_inflight_turn', lambda session: None)
    monkeypatch.setattr(server, '_publish_session_control_snapshot', lambda *args, **kwargs: None)
    monkeypatch.setattr(server, '_emit', lambda *args: None)
    monkeypatch.setattr(slack_mirror, 'accepted_user', lambda db, sid, row, **kw: calls.append(('user', db, sid, row)))
    monkeypatch.setattr(slack_mirror, 'completed_final', lambda db, sid, receipt, **kw: calls.append(('assistant', db, sid, receipt)))
    server._run_prompt_submit('rid', 'ui', session, 'Hello', display_kind=display_kind)
    assert [item[0] for item in calls] == expected
    if expected:
        assert calls == [('user', 'fake-db', 's1', 11),
                         ('assistant', 'fake-db', 's1', {'complete': True, 'final_assistant_row_id': 12})]
