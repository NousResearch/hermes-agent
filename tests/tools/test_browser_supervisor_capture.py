"""Captured transport contract against the actual supervisor response reader."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from contextlib import contextmanager

import pytest

from tools import browser_supervisor as bs


class Wire:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.changed = threading.Condition()
        self.closed = False
        self.gate = None
        self.fail_send = False

    async def send(self, raw):
        msg = json.loads(raw)
        with self.changed:
            self.sent.append(msg)
            self.changed.notify_all()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_send:
            raise OSError("uncertain send")

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self.incoming.get()
        if raw is None:
            self.closed = True
            raise StopAsyncIteration
        return raw

    def command(self, method, occurrence=0):
        with self.changed:
            assert self.changed.wait_for(
                lambda: len([m for m in self.sent if m['method'] == method]) > occurrence,
                timeout=3,
            ), f"missing command {method}"
            return [m for m in self.sent if m['method'] == method][occurrence]


class Harness:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.sup = bs.CDPSupervisor("captured-test", "ws://unused.invalid")
        self.sup._loop = self.loop
        self.wire = Wire()
        self.sup._ws = self.wire
        self.sup._page_session_id = "default-page"
        self.sup._set_active(True)
        self.registry = bs._SupervisorRegistry()
        self.registry._by_task[self.sup.task_id] = self.sup
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        self.readers = [self.submit(self.sup._read_loop())]

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def reply(self, command, result=None, error=None, wire=None):
        msg = {'id': command['id'], 'result': result or {}}
        if error:
            msg = {'id': command['id'], 'error': error}
        self.loop.call_soon_threadsafe((wire or self.wire).incoming.put_nowait, json.dumps(msg))

    def on_loop(self, fn):
        async def run():
            return fn()
        return self.submit(run()).result(3)

    def reconnect(self):
        old = self.wire
        def replace():
            self.wire = Wire()
            self.sup._ws = self.wire
            self.sup._connection_id = 'replacement-connection'
            self.sup._set_page_session('replacement-default')
            self.readers.append(self.submit(self.sup._read_loop()))
        self.on_loop(replace)
        return old

    def retired(self):
        async def wait():
            while self.sup._pending_calls or self.sup._pending_wires:
                await asyncio.sleep(0)
        self.submit(asyncio.wait_for(wait(), 2)).result(3)

    def finish_wire(self, wire=None):
        self.loop.call_soon_threadsafe((wire or self.wire).incoming.put_nowait, None)

    @contextmanager
    def caller(self, fn):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            yield pool.submit(fn)

    def close(self):
        async def stop():
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            await asyncio.sleep(0)
        self.submit(stop()).result(3)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(3)
        self.loop.close()


@pytest.fixture
def h():
    harness = Harness()
    try:
        yield harness
    finally:
        harness.close()


def test_capture_interleaves_original_calls_and_forwards_explicit_session(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    again = h.registry.capture(h.sup.task_id, timeout=1)
    assert captured.identity == again.identity
    assert captured.page_session_id == 'default-page'
    assert captured.valid
    assert 'unused.invalid' not in repr(captured)
    original = h.submit(h.sup._cdp('Page.getFrameTree', session_id='default-page'))
    original_msg = h.wire.command('Page.getFrameTree')
    with h.caller(lambda: captured.call('Runtime.evaluate', {'expression': '1'},
                                       session_id='child-session', timeout=1)) as caller:
        msg = h.wire.command('Runtime.evaluate')
        assert msg['sessionId'] == 'child-session'
        assert msg['id'] != original_msg['id']
        h.reply(msg, {'value': 'captured'})
        h.reply(original_msg, {'value': 'original'})
        assert caller.result(2)['result']['value'] == 'captured'
    assert original.result(2)['result']['value'] == 'original'
    assert not h.sup._pending_calls


@pytest.mark.parametrize('abandon', ['sync-timeout', 'async-timeout', 'async-cancel'])
def test_late_attach_keeps_response_owner_and_detaches_once(h, abandon):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    timeout = 0.1 if abandon.endswith('timeout') else 2
    if abandon == 'sync-timeout':
        with h.caller(lambda: captured.call('Target.attachToTarget', {'targetId': 'child', 'flatten': True},
                                           session_id=None, timeout=timeout)) as caller:
            msg = h.wire.command('Target.attachToTarget')
            with pytest.raises(TimeoutError):
                caller.result(2)
    else:
        started, cancelled = threading.Event(), threading.Event()
        async def call():
            task = asyncio.create_task(captured.acall('Target.attachToTarget', {'targetId': 'child'},
                                                      session_id=None, timeout=timeout))
            started.set()
            if abandon == 'async-cancel':
                while not cancelled.is_set():
                    await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(TimeoutError):
                    await task
        caller = h.submit(call())
        assert started.wait(2)
        msg = h.wire.command('Target.attachToTarget')
        cancelled.set()
        caller.result(2)
    assert h.on_loop(lambda: msg['id'] in h.sup._pending_calls)
    h.reply(msg, {'sessionId': 'late-session'})
    detach = h.wire.command('Target.detachFromTarget')
    assert detach['params'] == {'sessionId': 'late-session'}
    assert 'sessionId' not in detach
    h.reply(detach)
    h.reply(msg, {'sessionId': 'late-session'})
    h.retired()
    assert len([m for m in h.wire.sent if m['method'] == 'Target.detachFromTarget']) == 1


def test_late_attach_after_reconnect_uses_only_old_wire(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    with h.caller(lambda: captured.call('Target.attachToTarget', {'targetId': 'child'},
                                       session_id='parent-session', timeout=0.1)) as caller:
        msg = h.wire.command('Target.attachToTarget')
        with pytest.raises(TimeoutError):
            caller.result(2)
    old = h.reconnect()
    h.reply(msg, {'sessionId': 'late-old'}, wire=old)
    detach = old.command('Target.detachFromTarget')
    assert detach['params'] == {'sessionId': 'late-old'}
    assert detach['sessionId'] == 'parent-session'
    h.reply(detach, wire=old)
    h.retired()
    assert not h.wire.sent


def test_attach_invalidated_after_reply_is_not_returned_or_leaked(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    with h.caller(lambda: captured.call('Target.attachToTarget', {'targetId': 'child'},
                                       session_id=None, timeout=1)) as caller:
        msg = h.wire.command('Target.attachToTarget')
        h.on_loop(lambda: h.sup._set_page_session('new-default'))
        h.reply(msg, {'sessionId': 'unclaimed'})
        with pytest.raises(bs.CapturedCDPInvalid):
            caller.result(2)
        detach = h.wire.command('Target.detachFromTarget')
        assert detach['params'] == {'sessionId': 'unclaimed'}
        h.reply(detach)
    h.retired()


def test_failed_attach_send_keeps_late_response_owner(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    h.wire.fail_send = True
    with h.caller(lambda: captured.call('Target.attachToTarget', {'targetId': 'child'},
                                       session_id=None, timeout=1)) as caller:
        msg = h.wire.command('Target.attachToTarget')
        with pytest.raises(OSError, match='uncertain send'):
            caller.result(2)
    assert h.on_loop(lambda: msg['id'] in h.sup._pending_calls)
    h.wire.fail_send = False
    h.reply(msg, {'sessionId': 'uncertain-attachment'})
    detach = h.wire.command('Target.detachFromTarget')
    assert detach['params'] == {'sessionId': 'uncertain-attachment'}
    h.reply(detach)
    h.retired()


def test_pending_acquisition_retires_on_closed_old_wire_without_fallback(h):
    captured = h.registry.capture(h.sup.task_id, timeout=1)
    h.on_loop(lambda: setattr(h.wire, 'gate', asyncio.Event()))
    with h.caller(lambda: captured.call('Target.attachToTarget', {'targetId': 'child'},
                                       session_id=None, timeout=0.1)) as caller:
        msg = h.wire.command('Target.attachToTarget')
        with pytest.raises(TimeoutError):
            caller.result(2)
    assert h.on_loop(lambda: msg['id'] in h.sup._pending_calls)
    old = h.reconnect()
    h.finish_wire(old)
    h.readers[0].result(2)
    h.retired()
    assert not h.wire.sent
