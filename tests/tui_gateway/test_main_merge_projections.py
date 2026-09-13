"""Extracted discovery and snapshot projections retain main's wire contracts."""
import threading
from contextlib import nullcontext

def test_live_message_count_tracks_serialized_projection(monkeypatch):
    from tui_gateway import server
    monkeypatch.setattr(server, '_session_db', lambda session: nullcontext(None))
    monkeypatch.setattr(server, '_fallback_session_info', lambda session: {})
    monkeypatch.setattr(server, '_pending_approval_request_payload', lambda key: None)
    monkeypatch.setattr(server, '_pending_clarify_request_payload', lambda sid: None)
    # A display row may expand or disappear in the serializer. The count is the wire size.
    monkeypatch.setattr(server, '_history_to_messages', lambda rows: [{'text': 'visible'}])
    session = {'history_lock': threading.Lock(), 'history': [{}, {}], 'session_key': 'owned'}
    payload = server._live_session_payload('owned', session)
    assert payload['message_count'] == len(payload['messages']) == 1
    omitted = server._live_session_payload('owned', session, omit_messages=True)
    assert omitted['messages'] == [] and omitted['message_count'] == 2
