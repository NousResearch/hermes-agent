from uuid import uuid4

from tui_gateway.desktop_work import admit, finish


def test_terminal_requires_classified_provider_failure():
    session = {'source': 'desktop', 'session_key': 'stored'}
    proof = {'origin': 'desktop_user', 'root_id': str(uuid4())}
    seen = []
    work = admit('sid', session, proof)
    finish(work, lambda *args: seen.append(args[2]), {'error': 'local runtime crashed', 'failed': True})
    assert seen[-1]['kind'] == 'interrupted'
    seen.clear()
    work = admit('sid', session, {**proof, 'root_id': str(uuid4())})
    finish(work, lambda *args: seen.append(args[2]), {'error': 'quota', 'failed': True, 'failure_reason': 'billing'})
    assert seen[-1]['kind'] == 'provider_failed'


def test_title_is_read_from_canonical_db_at_outcome(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stored", source="desktop")
    db.set_session_title("stored", "First title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    session = {
        'source': 'desktop', 'session_key': 'stored',
        'pending_title': 'Stale pending title', 'title': 'Stale cached title',
    }
    work = admit('sid', session, {'origin': 'desktop_user', 'root_id': str(uuid4())})
    seen = []
    work.emit(lambda *args: seen.append(args[2]), 'started')
    db.set_session_title("stored", "Renamed title")
    finish(work, lambda *args: seen.append(args[2]), {'completed': True})
    assert [e['title'] for e in seen] == ['First title', 'Renamed title']
    db.close()


def test_notification_dispatch_exception_requeues_delivery(monkeypatch):
    import threading

    from tools import async_delegation
    from tui_gateway import server

    event = {'type': 'completion', 'session_id': 'child'}
    session = {'running': True, 'history_lock': threading.RLock()}
    completed = []
    retried = []

    def fail_submit(*args, **kwargs):
        raise RuntimeError('dispatcher failed')

    monkeypatch.setattr(server, '_emit', lambda *args: None)
    monkeypatch.setattr(server, '_run_prompt_submit', fail_submit)
    monkeypatch.setattr(async_delegation, 'claim_event_delivery', lambda *args: 'claim')
    monkeypatch.setattr(async_delegation, 'complete_event_delivery', lambda *args: completed.append(args))
    monkeypatch.setattr(async_delegation, 'retry_event_delivery',
                        lambda *args, **kwargs: retried.append((args, kwargs)))

    failure = server._notif_submit('rid', 'sid', session, 'text', 'dispatch failed')
    monkeypatch.setattr(server, '_notif_submit', lambda *args, **kwargs: failure)
    server._notif_dispatch_event('sid', session, event, 'text')

    assert completed == []
    assert retried == [((event, 'claim'), {})]


def test_notification_refused_admission_requeues_delivery(monkeypatch):
    import threading

    from tools import async_delegation
    from tui_gateway import server

    event = {'type': 'async_delegation', 'delegation_id': 'batch'}
    session = {'running': True, 'history_lock': threading.RLock()}
    completed = []
    retried = []
    sleeps = []

    monkeypatch.setattr(server, '_notif_submit', lambda *args, **kwargs: False)
    monkeypatch.setattr(async_delegation, 'claim_event_delivery', lambda *args: 'claim')
    monkeypatch.setattr(async_delegation, 'complete_event_delivery', lambda *args: completed.append(args))
    monkeypatch.setattr(async_delegation, 'retry_event_delivery',
                        lambda *args, **kwargs: retried.append((args, kwargs)))
    monkeypatch.setattr(server.time, 'sleep', lambda seconds: sleeps.append(seconds))

    server._notif_dispatch_event('sid', session, event, 'text')

    assert completed == []
    assert retried == [((event, 'claim'), {'defer': True})]
    assert sleeps == [0.25]


def test_async_requires_consumption_and_settled_continuation():
    work = admit('sid', {'source': 'desktop', 'session_key': 'stored'},
                 {'origin': 'desktop_user', 'root_id': str(uuid4())})
    seen = []
    publish = lambda *args: seen.append(args[2]['kind'])
    work.begin_turn()
    work.emit(publish, 'started')
    work.retain_async('batch')
    finish(work, publish, {'completed': True})
    assert seen == ['started', 'waiting']
    work.begin_turn()
    # Fast continuation can finish before the poller commits delivery.
    finish(work, publish, {'completed': True})
    assert 'completed' not in seen
    work.consume_async('batch')
    work.consume_async('batch')
    assert seen == ['started', 'waiting', 'completed']


def test_missing_or_synthetic_provenance_is_not_human():
    session = {'source': 'desktop', 'session_key': 'stored'}
    proof = {'origin': 'desktop_user', 'root_id': str(uuid4())}
    for invalid in (None, {}, {'root_id': str(uuid4())}, {**proof, 'origin': 'tool'}, {**proof, 'root_id': 'bad'}):
        assert admit('sid', session, invalid) is None
    assert admit('sid', session, proof, 'hidden') is None
    for extra in ({'hidden': True}, {'source': 'tool'}, {'parent_session_id': 'parent'}):
        assert admit('sid', {**session, **extra}, proof) is None


def test_publish_failure_cannot_interrupt_agent_lifecycle():
    from tui_gateway.desktop_work import DesktopWork

    work = DesktopWork("root", "sid", "stored", "default", "Title")
    work.begin_turn()

    def fail_publish(*args):
        raise RuntimeError("observer unavailable")

    work.emit(fail_publish, "started")
    finish(work, fail_publish, {"completed": True})

    assert work.closed is True
