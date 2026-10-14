"""Frame acquisition has a finite transport deadline."""
from concurrent.futures import Future
from types import SimpleNamespace
import pytest
from tools.browser_vault_document import OwnedWire, DocumentRefused


def test_missing_frame_reply_is_bounded(monkeypatch):
    from tools import browser_vault_document as module
    budgets = []
    async def cdp(*args, **kwargs):
        budgets.append(kwargs['timeout'])
    def schedule(coroutine, loop):
        import asyncio
        asyncio.run(coroutine)
        future = Future()
        future.set_exception(TimeoutError('no reply'))
        return future
    monkeypatch.setattr(module, 'safe_schedule_threadsafe', schedule)
    wire = object.__new__(OwnedWire)
    wire.check = lambda: None
    wire.supervisor = SimpleNamespace(_cdp=cdp, _loop=None)
    wire.root_session = 'root'
    wire.ws = object()
    wire.attached = []
    with pytest.raises(DocumentRefused):
        wire.attach_frame('child')
    assert budgets == [16]
    assert wire.attached == []


def test_supervisor_missing_reply_removes_pending_call():
    import asyncio
    from tools.browser_supervisor import CDPSupervisor
    async def send(payload):
        pass
    supervisor = object.__new__(CDPSupervisor)
    supervisor._ws = SimpleNamespace(send=send)
    supervisor._next_call_id = 1
    supervisor._pending_calls = {}
    async def exercise():
        with pytest.raises(TimeoutError):
            await supervisor._cdp('Target.attachToTarget', timeout=.01)
    asyncio.run(exercise())
    assert supervisor._pending_calls == {}


def test_late_frame_reply_releases_exact_acquired_session(monkeypatch):
    from tools import browser_vault_document as module
    class LateReply(Future):
        def result(self, timeout=None):
            if not self.done():
                raise TimeoutError('caller deadline')
            return super().result(timeout)
    future = LateReply()
    def schedule(coroutine, loop):
        coroutine.close()
        return future
    async def cdp(*args, **kwargs):
        pass
    monkeypatch.setattr(module, 'safe_schedule_threadsafe', schedule)
    wire = object.__new__(OwnedWire)
    wire.check = lambda: None
    wire.supervisor = SimpleNamespace(_cdp=cdp, _loop=None)
    wire.root_session = 'root'
    wire.ws = object()
    wire.attached = []
    released = []
    wire.release = released.append
    with pytest.raises(DocumentRefused):
        wire.attach_frame('child')
    future.set_result({'result': {'sessionId': 'late-exact-session'}})
    assert released == ['late-exact-session']
    assert wire.attached == []
