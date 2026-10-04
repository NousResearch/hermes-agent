"""Invalidation and finite-wait contracts for the captured transport."""
import pytest
import asyncio
from tests.tools.test_browser_supervisor_capture import h, bs  # noqa: F401

@pytest.mark.parametrize('change', ['wire', 'registry', 'attachment'])
def test_invalid_capture_refuses_before_dispatch(h, change):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    old = h.wire
    if change == 'wire':
        h.reconnect()
    elif change == 'registry':
        h.registry._pop(h.sup.task_id)
    else:
        h.on_loop(lambda: h.sup._set_page_session('new-page'))
    with pytest.raises(bs.CapturedCDPInvalid):
        captured.call('Runtime.evaluate', {'expression': '1'}, session_id='child', timeout=1)
    assert not old.sent and not h.wire.sent


def test_reconnect_during_send_refuses_result_and_cleanup_stays_old(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    h.on_loop(lambda: setattr(h.wire, 'gate', asyncio.Event()))
    with h.caller(lambda: captured.call('Runtime.evaluate', {'expression': '1'},
                                       session_id='child', timeout=1)) as caller:
        msg = h.wire.command('Runtime.evaluate')
        old = h.reconnect()
        h.on_loop(old.gate.set)
        h.reply(msg, {'value': 'old'}, wire=old)
        with pytest.raises(bs.CapturedCDPInvalid):
            caller.result(2)
    with h.caller(lambda: captured.cleanup_call('Runtime.releaseObject', {'objectId': 'owned-object'},
                                               session_id='child', timeout=1)) as caller:
        cleanup = old.command('Runtime.releaseObject')
        h.reply(cleanup, wire=old)
        caller.result(2)
    assert not h.wire.sent
    h.retired()


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True, '1', None])
def test_invalid_timeouts_do_not_dispatch(h, value):
    with pytest.raises(ValueError):
        h.registry.capture(h.sup.task_id, timeout=value)
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    with pytest.raises(ValueError):
        captured.call('Runtime.evaluate', {}, session_id=None, timeout=value)
    assert not h.wire.sent


def test_non_attach_timeout_retires_response_table(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    with h.caller(lambda: captured.call('Runtime.evaluate', {}, session_id=None, timeout=.1)) as caller:
        msg = h.wire.command('Runtime.evaluate')
        with pytest.raises(TimeoutError):
            caller.result(2)
    h.retired()
    h.reply(msg, {'value': 'late ignored'})
    assert not h.sup._pending_calls


def test_cleanup_survives_async_caller_cancellation(h):
    import threading
    cancelled = threading.Event()
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    h.on_loop(lambda: setattr(h.wire, 'gate', asyncio.Event()))
    async def run():
        task = asyncio.create_task(captured.acleanup_call('Runtime.releaseObject', {'objectId':'owned'},
                                                       session_id='child', timeout=2))
        while not cancelled.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    caller = h.submit(run())
    msg = h.wire.command('Runtime.releaseObject')
    cancelled.set()
    caller.result(2)
    assert h.on_loop(lambda: msg['id'] in h.sup._pending_calls)
    h.on_loop(h.wire.gate.set)
    h.reply(msg)
    h.retired()
