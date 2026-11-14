"""The completion retains the modality of the dispatching human turn."""
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_context import clear_session_vars, set_session_vars
from tools.async_delegation import _capture_routing_origin, completion_event


def test_completion_retains_dispatch_voice_origin():
    record = {'delegation_id': 'one', 'session_key': 'route', 'origin_message_type': 'voice'}
    event, _, _ = completion_event(record, {'summary': 'done'}, 'completed')
    assert event.get('origin_message_type') == 'voice'


def test_dispatch_captures_voice_from_bound_turn_not_process_environment():
    event = MessageEvent(text='request', message_type=MessageType.VOICE)
    tokens = set_session_vars(turn_event=event)
    try:
        assert _capture_routing_origin().get('origin_message_type') == 'voice'
    finally:
        clear_session_vars(tokens)


def test_real_dispatch_persists_bound_origin(monkeypatch):
    import json
    import threading
    from tools import async_delegation as ad

    ad._reset_for_tests()
    release = threading.Event()
    finished = threading.Event()
    monkeypatch.setenv("HERMES_SESSION_MESSAGE_TYPE", "text")
    tokens = set_session_vars(turn_event=MessageEvent(text="request", message_type=MessageType.VOICE))
    try:
        def runner():
            release.wait(5)
            finished.set()
            return {"status": "completed", "summary": "done"}

        handle = ad.dispatch_async_delegation(
            goal="work", context=None, toolsets=None, role="leaf", model=None,
            session_key="route", runner=runner,
        )
        assert handle["status"] == "dispatched"
        with ad._DB_LOCK, ad._transaction() as conn:
            row = conn.execute("SELECT task_json FROM async_delegations WHERE delegation_id=?",
                               (handle["delegation_id"],)).fetchone()
        assert json.loads(row[0])["origin_message_type"] == "voice"
    finally:
        release.set()
        finished.wait(5)
        clear_session_vars(tokens)
