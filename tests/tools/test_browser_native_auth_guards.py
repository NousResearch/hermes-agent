"""Security transitions identified by independent native-auth review."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.browser_supervisor_auth import NativeAuthMixin, _NativeCleanupFailed


def event_queue():
    queue = asyncio.Queue()
    request = {"requestId": "fetch", "networkId": "network", "frameId": "top", "resourceType": "Document",
               "request": {"url": "http://fixture.test/"}}
    queue.put_nowait(("Fetch.requestPaused", request))
    queue.put_nowait(("Fetch.authRequired", {**request, "authChallenge": {
        "source": "Server", "scheme": "basic", "origin": "http://fixture.test"}}))
    return queue


@pytest.mark.asyncio
async def test_takeover_after_resolution_blocks_credential_dispatch(monkeypatch):
    from tools import browser_vault_auth, browser_vault_tool
    from tools.bot_desktop import lease
    state = SimpleNamespace(epoch=1)
    monkeypatch.setattr(browser_vault_tool, "_bot_desktop_browser_session", lambda task: True)
    monkeypatch.setattr(lease, "assert_agent_may_act", lambda: state)
    authorize = browser_vault_auth._native_authorizer("synthetic")
    queue = event_queue()
    supervisor = NativeAuthMixin()
    supervisor._vault_auth_events = {"session": queue}
    commands = []

    async def cdp(method, params, **kwargs):
        commands.append(method)
        return {}

    def resolve(origin, transmit):
        state.epoch = 2
        transmit("synthetic", "synthetic-password")

    supervisor._cdp = cdp
    with pytest.raises(RuntimeError, match="control changed"):
        await supervisor._vault_auth_events_loop(queue, "session", "http://fixture.test", "top", resolve, authorize)
    assert "Fetch.continueWithAuth" not in commands


@pytest.mark.asyncio
async def test_unrelated_document_response_cannot_claim_authentication_success():
    queue = event_queue()
    queue.put_nowait(("Network.responseReceived", {"requestId": "unrelated", "frameId": "top", "type": "Document",
                                                 "response": {"url": "http://fixture.test/", "status": 200}}))
    queue.put_nowait(("Network.responseReceived", {"requestId": "network", "frameId": "top", "type": "Document",
                                                 "response": {"url": "http://fixture.test/", "status": 401}}))
    supervisor = NativeAuthMixin()
    supervisor._vault_auth_events = {"session": queue}

    async def cdp(method, params, **kwargs):
        return {}

    def resolve(origin, transmit):
        transmit("synthetic", "synthetic-password")

    supervisor._cdp = cdp
    result = await supervisor._vault_auth_events_loop(queue, "session", "http://fixture.test", "top", resolve, Mock())
    assert not result["success"]
    assert result["error_type"] == "native_auth_rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disable", "detach", "both", "backpressure"])
async def test_cleanup_failures_keep_interception_owned_and_fail_closed(failure):
    supervisor = NativeAuthMixin()
    supervisor._vault_auth_events = {}
    never = asyncio.Event()

    async def cdp(method, params=None, **kwargs):
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"type": "page", "url": "http://fixture.test/", "targetId": "target"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "session"}}
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {"frame": {"id": "top"}}}}
        if method in {"Fetch.disable", "Target.detachFromTarget"}:
            if failure == "backpressure":
                await never.wait()
            if failure == "both" or (failure == "disable" and method == "Fetch.disable") or (
                    failure == "detach" and method == "Target.detachFromTarget"):
                raise RuntimeError("synthetic transport failure")
        return {}

    supervisor._cdp = cdp
    if failure in {"both", "backpressure"}:
        with pytest.raises(_NativeCleanupFailed):
            await asyncio.wait_for(supervisor._vault_basic(["http://fixture.test"], Mock(), 0.01, Mock()), 8)
        assert "session" in supervisor._vault_auth_events
        assert supervisor._vault_auth_poisoned
        result = await supervisor._vault_basic(["http://fixture.test"], Mock(), 0.01, Mock())
        assert result["error_type"] == "native_auth_cleanup_failed"
    else:
        result = await supervisor._vault_basic(["http://fixture.test"], Mock(), 0.01, Mock())
        assert not result["success"]
        assert not supervisor._vault_auth_events
    assert not supervisor._vault_auth_busy


def test_native_auth_cannot_use_another_profiles_supervisor():
    supervisor = NativeAuthMixin()
    supervisor._vault_owner_home = "another-profile"
    resolve = Mock()
    result = supervisor.authenticate_vault_basic(["http://fixture.test"], resolve, Mock())
    assert result["error_type"] == "native_auth_profile_mismatch"
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_takeover_during_setup_prevents_navigation():
    supervisor = NativeAuthMixin()
    supervisor._vault_auth_events = {}
    commands = []

    async def cdp(method, params=None, **kwargs):
        commands.append(method)
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"type": "page", "url": "http://fixture.test/", "targetId": "target"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "session"}}
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {"frame": {"id": "top"}}}}
        return {}

    supervisor._cdp = cdp
    result = await supervisor._vault_basic(["http://fixture.test"], Mock(), 0.01,
                                          Mock(side_effect=RuntimeError("human took over")))
    assert not result["success"]
    assert "Page.navigate" not in commands
    assert not supervisor._vault_auth_events
