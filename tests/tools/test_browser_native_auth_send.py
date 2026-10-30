"""No credential transport may outlive its operation or backend context."""
import asyncio
import threading
from unittest.mock import Mock

import pytest

from tools.browser_supervisor_auth import NativeAuthMixin, _NativeCleanupFailed


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["send_timeout", "operation_timeout", "cancel"])
async def test_inflight_credentials_settle_before_context_exit_or_cleanup(ending):
    supervisor = NativeAuthMixin()
    supervisor._vault_auth_events = {}
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = threading.Event()
    context_exited = threading.Event()
    exit_after_settled = []
    delivered = []

    async def cdp(method, params=None, **kwargs):
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"type": "page", "url": "http://fixture.test/", "targetId": "target"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "session"}}
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {"frame": {"id": "top"}}}}
        if method == "Fetch.enable":
            queue = supervisor._vault_auth_events["session"]
            request = {"requestId": "fetch", "networkId": "network", "frameId": "top", "resourceType": "Document",
                       "request": {"url": "http://fixture.test/"}}
            queue.put_nowait(("Fetch.requestPaused", request))
            queue.put_nowait(("Fetch.authRequired", {**request, "authChallenge": {
                "source": "Server", "scheme": "basic", "origin": "http://fixture.test"}}))
        if method == "Page.navigate":
            return {"result": {"frameId": "top", "loaderId": "loader"}}
        if method == "Fetch.continueWithAuth":
            entered.set()
            try:
                await release.wait()
                delivered.append(True)
            finally:
                settled.set()
        if ending != "send_timeout" and method in {"Fetch.disable", "Target.detachFromTarget"}:
            raise RuntimeError("synthetic cleanup failure")
        return {}

    def resolve(origin, transmit):
        try:
            transmit("synthetic-user", "synthetic-password")
        finally:
            exit_after_settled.append(settled.is_set())
            context_exited.set()

    supervisor._cdp = cdp
    timeout = 0.1 if ending == "operation_timeout" else 10
    task = asyncio.create_task(supervisor._vault_basic(["http://fixture.test"], resolve, timeout, Mock()))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if ending == "cancel":
            task.cancel()
        if ending == "send_timeout":
            result = await asyncio.wait_for(task, 6)
            assert not result["success"]
            assert not supervisor._vault_auth_events
        else:
            with pytest.raises(_NativeCleanupFailed):
                await asyncio.wait_for(task, 6)
            assert supervisor._vault_auth_poisoned
            assert "session" in supervisor._vault_auth_events
        assert settled.is_set()
        assert await asyncio.to_thread(context_exited.wait, 5)
        assert exit_after_settled == [True]
        release.set()
        assert not delivered
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
