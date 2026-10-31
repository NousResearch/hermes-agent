"""A manager finishing after timeout must never authenticate a browser request."""
import asyncio
import threading
from unittest.mock import Mock

import pytest

from tools.browser_supervisor_auth import NativeAuthMixin


@pytest.mark.asyncio
async def test_cancelled_auth_cannot_transmit_late_credentials():
    supervisor = NativeAuthMixin()
    queue = asyncio.Queue()
    supervisor._vault_auth_events = {"session": queue}
    sends = []
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []

    async def cdp(method, params, **kwargs):
        sends.append((method, params))
        return {}

    def resolve(origin, transmit):
        entered.set()
        try:
            assert release.wait(timeout=5)
            transmit("synthetic-user", "synthetic-late-password")
        except Exception as error:
            errors.append(type(error))
        finally:
            finished.set()

    supervisor._cdp = cdp
    request = {"requestId": "request", "frameId": "top", "resourceType": "Document",
               "request": {"url": "http://fixture.test/"}, "networkId": "network"}
    queue.put_nowait(("Fetch.requestPaused", request))
    queue.put_nowait(("Fetch.authRequired", {**request, "authChallenge": {
        "source": "Server", "scheme": "basic", "origin": "http://fixture.test"}}))
    task = asyncio.create_task(supervisor._vault_auth_events_loop(queue, "session", "http://fixture.test", "top", resolve, Mock()))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        assert errors == [TimeoutError]
        assert not any(method == "Fetch.continueWithAuth" for method, _ in sends)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
