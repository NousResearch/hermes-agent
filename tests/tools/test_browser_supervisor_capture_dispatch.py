"""Dispatch/lossless-wait extension exercised through the real public capture API."""
import asyncio
import concurrent.futures
import json
import threading
import time
import pytest
from tests.tools.test_browser_supervisor_capture import _FakeCDP, _wait
from tools.browser_supervisor import _SupervisorRegistry
from tools.browser_supervisor_capture import CapturedCDPInvalid

class DelayedCDP(_FakeCDP):
    async def _handle(self, ws):
        self.connections.append(ws)
        conn = len(self.connections)
        async for raw in ws:
            msg = json.loads(raw)
            method, sid = msg['method'], msg.get('sessionId')
            self.frames.append((conn, method, sid))
            result = {'echo':method}
            if method == 'Target.getTargets':
                result = {'targetInfos':[{'targetId':'fixture-parent','type':'page','url':'about:blank'}]}
            elif method == 'Target.attachToTarget':
                if msg.get('params',{}).get('targetId') == 'late-child':
                    await asyncio.sleep(.2)
                    result = {'sessionId':'late-owned-session'}
                else:
                    result = {'sessionId':'default-session'}
            await ws.send(json.dumps({'id':msg['id'],'result':result}))

@pytest.fixture
def pair():
    server, registry = DelayedCDP(), _SupervisorRegistry()
    sup = registry.get_or_start('fixture', server.url, start_timeout=5)
    try:
        yield server, registry, sup, registry.capture('fixture')
    finally:
        registry.stop_all()


def test_unbounded_reply_retains_late_attachment_for_consumer_cleanup(pair):
    server, _registry, _sup, handle = pair
    abandoned = threading.Event()
    outcomes = []
    def worker():
        result = handle.call('Target.attachToTarget', {'targetId':'late-child','flatten':True}, timeout=None)
        outcomes.append(result['result']['sessionId'])
        if abandoned.is_set():
            handle.call('Target.detachFromTarget', {'sessionId':outcomes[-1]})
    thread = threading.Thread(target=worker)
    thread.start()
    _wait(lambda: len([m for m in server.methods_on(1) if m == 'Target.attachToTarget']) == 2)
    abandoned.set()
    thread.join(3)
    assert not thread.is_alive()
    assert outcomes == ['late-owned-session']
    assert server.methods_on(1).count('Target.detachFromTarget') == 1


def test_dispatch_deadline_refuses_queued_command_after_timeout(pair):
    server, _registry, sup, handle = pair
    entered = threading.Event()
    def block():
        # Stall the loop past the caller's deadline but inside its +1s result grace.
        entered.set()
        time.sleep(.4)
    sup._loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    with pytest.raises(TimeoutError):
        handle.call('Probe.expired', timeout=.1)
    handle.call('Target.getTargets', timeout=2)
    assert 'Probe.expired' not in server.methods_on(1)


def test_before_send_checks_cancellation_on_supervisor_loop(pair):
    server, _registry, sup, handle = pair
    entered, release, abandoned = threading.Event(), threading.Event(), threading.Event()
    failures = []
    def block():
        entered.set()
        release.wait(4)
    def check():
        assert asyncio.get_running_loop() is sup._loop
        if abandoned.is_set():
            raise ValueError('fixture cancelled')
    def caller():
        try:
            handle.call('Probe.cancelled', timeout=2, before_send=check)
        except BaseException as exc:
            failures.append(exc)
    sup._loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    thread = threading.Thread(target=caller)
    thread.start()
    abandoned.set()
    release.set()
    thread.join(3)
    assert failures and isinstance(failures[0], ValueError)
    assert 'Probe.cancelled' not in server.methods_on(1)


def test_unbounded_reply_fails_promptly_on_socket_close(pair):
    server, _registry, _sup, handle = pair
    failures = []
    # This fixture replies to ordinary methods; block the connection handler on attach instead.
    def attach():
        try:
            handle.call('Target.attachToTarget', {'targetId':'late-child'}, timeout=None)
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=attach)
    thread.start()
    _wait(lambda: server.methods_on(1).count('Target.attachToTarget') == 2)
    server.drop_current()
    thread.join(3)
    assert not thread.is_alive()
    assert failures and isinstance(failures[0], CapturedCDPInvalid)



def test_unbounded_wait_ends_when_loop_closes_with_dispatch_queued(pair, monkeypatch):
    _server, registry, _sup, handle = pair
    from agent import async_utils
    # A stop racing call() can close the loop before its queued dispatch runs; the
    # scheduled future then never resolves.
    real = async_utils.safe_schedule_threadsafe
    def dropped(coro, loop, **kw):
        if coro.__qualname__ != 'CapturedCDP._send':
            return real(coro, loop, **kw)
        coro.close()
        return concurrent.futures.Future()
    monkeypatch.setattr(async_utils, 'safe_schedule_threadsafe', dropped)
    failures = []
    def caller():
        try:
            handle.call('Probe.dropped', timeout=None)
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=caller)
    thread.start()
    registry.stop_all()
    thread.join(4)
    assert not thread.is_alive()
    assert failures and isinstance(failures[0], CapturedCDPInvalid)


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), 1e300, True, '5'])
def test_invalid_timeout(pair, timeout):
    server, _, _, cap = pair
    before = list(server.methods_on(1))
    with pytest.raises(ValueError):
        cap.call('Probe.invalid', timeout=timeout)
    assert server.methods_on(1) == before


def test_async_validator_refuses(pair):
    server, _, _, cap = pair
    before = list(server.methods_on(1))
    async def check():
        return None
    with pytest.raises(TypeError, match='synchronously'):
        cap.call('Probe.async', before_send=check)
    assert server.methods_on(1) == before
