"""Regressions requested by independent Astra review."""
import threading
import pytest
from tests.tools.test_browser_supervisor_capture import h, bs  # noqa: F401


def test_exact_default_detach_invalidates_but_child_detach_does_not(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    def event(sid):
        import json
        h.loop.call_soon_threadsafe(h.wire.incoming.put_nowait,
            json.dumps({'method':'Target.detachedFromTarget','params':{'sessionId':sid}}))
        h.on_loop(lambda: None)
    event('unrelated-child')
    assert captured.valid
    event('default-page')
    assert not captured.valid
    with pytest.raises(bs.CapturedCDPInvalid):
        captured.call('Runtime.evaluate', {}, session_id='child', timeout=1)
    with pytest.raises(bs.CapturedCDPInvalid):
        h.registry.capture(h.sup.task_id, timeout=1)
    assert not h.wire.sent


def test_default_attachment_aba_stays_invalid(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    h.on_loop(lambda: h.sup._set_page_session('replacement'))
    h.on_loop(lambda: h.sup._set_page_session('default-page'))
    assert not captured.valid


def test_publication_before_claim_preserves_cleanup_after_default_detach(h, monkeypatch):
    from tools.browser_supervisor_capture import _AttachmentHandoff
    captured = h.registry.capture(h.sup.task_id, timeout=2)
    entered, release = threading.Event(), threading.Event()
    original_claim = _AttachmentHandoff.claim
    def delayed_claim(handoff):
        entered.set()
        assert release.wait(3)
        return original_claim(handoff)
    monkeypatch.setattr(_AttachmentHandoff, 'claim', delayed_claim)
    with h.caller(lambda: captured.call('Target.attachToTarget', {'targetId':'child'}, session_id=None, timeout=2)) as caller:
        request = h.wire.command('Target.attachToTarget')
        h.reply(request, {'sessionId':'owned-but-unclaimed'})
        assert entered.wait(2)
        try:
            import json
            h.loop.call_soon_threadsafe(h.wire.incoming.put_nowait, json.dumps({'method':'Target.detachedFromTarget', 'params':{'sessionId':'default-page'}}))
            h.on_loop(lambda: None)
            assert not captured.valid
        finally:
            release.set()
        with pytest.raises(bs.CapturedCDPInvalid):
            caller.result(2)
        cleanup = h.wire.command('Target.detachFromTarget')
        assert cleanup['params'] == {'sessionId':'owned-but-unclaimed'}
        assert len([x for x in h.wire.sent if x['method']=='Target.detachFromTarget']) == 1
        h.reply(cleanup)
        h.retired()


def test_wrong_wire_response_does_not_resolve_captured_request(h):
    captured = h.registry.capture(h.sup.task_id, timeout=2)
    with h.caller(lambda: captured.call('Runtime.evaluate', {'expression':'void 0'}, session_id=None, timeout=2)) as caller:
        old = h.wire
        request = old.command('Runtime.evaluate')
        h.reconnect()
        h.reply(request, {'wrong-wire':True})
        h.on_loop(lambda: None)
        assert request['id'] in h.sup._pending_calls
        assert not caller.done()
        h.reply(request, {'original-wire':True}, wire=old)
        with pytest.raises(bs.CapturedCDPInvalid):
            caller.result(2)
        h.retired()


def test_old_wire_default_detach_cannot_invalidate_replacement_capture(h):
    import json
    old = h.reconnect()
    # ABA of the visible session ID must not let an old-wire event touch the new capture.
    h.on_loop(lambda: h.sup._set_page_session('default-page'))
    replacement = h.registry.capture(h.sup.task_id, timeout=1)
    h.on_loop(lambda: old.incoming.put_nowait(json.dumps({'method':'Target.detachedFromTarget','params':{'sessionId':'default-page'}})))
    h.on_loop(lambda: None)
    assert replacement.valid


def test_timed_out_queued_write_never_dispatches(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    entered, release = threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(3)
    h.loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    try:
        with h.caller(lambda: captured.call('Runtime.evaluate', {'expression':'globalThis.fixture=1'},
                                           session_id='default-page',timeout=.1)) as caller:
            with pytest.raises(TimeoutError):
                caller.result(2)
    finally:
        release.set()
    h.on_loop(lambda: None)
    h.retired()
    assert not h.wire.sent


def test_async_cancel_before_dispatch_from_foreign_loop(h, monkeypatch):
    import asyncio
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
    original = captured._submit
    def submit(*args):
        result = original(*args)
        submitted.set()
        return result
    monkeypatch.setattr(captured, '_submit', submit)
    def block():
        entered.set()
        assert release.wait(3)
    h.loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    async def client():
        task = asyncio.create_task(captured.acall('Runtime.evaluate', {'expression':'globalThis.fixture=1'},
                                                 session_id='default-page', timeout=2))
        await asyncio.to_thread(submitted.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    try:
        with h.caller(lambda: asyncio.run(client())) as caller:
            caller.result(2)
    finally:
        release.set()
    h.on_loop(lambda: None)
    h.retired()
    assert not h.wire.sent
