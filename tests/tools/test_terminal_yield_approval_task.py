"""A yielded foreground process retains the launching task for its completion."""
from types import SimpleNamespace

from tools.approval_context import approval_task_context
from tools.terminal_tool_background import yield_to_background_handler


def test_yielded_process_reserves_origin_only_for_supported_completion(monkeypatch):
    from tools.process_registry import process_registry

    session = SimpleNamespace(id='yielded', pid=42, notify_on_complete=True)
    monkeypatch.setattr(process_registry, 'adopt_local', lambda *_a, **_k: session)
    monkeypatch.setattr('tools.terminal_tool_background._stamp_routing_if_gateway', lambda *_a: None)
    handler = yield_to_background_handler(command='build', env_type='local', cwd='.',
                                          effective_task_id='build', task_id='build', session_key='route')
    with approval_task_context('Build only alpha.py', identity='local:user', profile='profile',
                               session='route', task_id='admission', is_current=lambda: True):
        result = handler(object(), 'partial')
    assert result['yielded_session_id'] == 'yielded'
    reservation = session.approval_task_reservation
    try:
        assert reservation.evidence.text == 'Build only alpha.py'
        assert reservation.lease.live()
    finally:
        reservation.release()

    session.notify_on_complete = True
    monkeypatch.setattr('tools.terminal_tool_background._stamp_routing_if_gateway',
                        lambda *_a: setattr(session, 'notify_on_complete', False))
    with approval_task_context('Build only beta.py', identity='local:user', profile='profile',
                               session='route', task_id='second', is_current=lambda: True):
        handler(object(), 'partial')
    assert session.approval_task_reservation is None
