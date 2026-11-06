"""Cancellation while a Pocket read's initial refresh is waiting on a peer-held refresh fence.

The fixture owns the real fence; the flow polls ``acquire_refresh_fence`` on its real 50 ms timer.
The sleep wrapper only OBSERVES timer entry and the cancellation landing in it, then delegates."""

import asyncio

import pytest

from tests.tools.test_mcp_oauth_pocket_recovery_bounds import (
    _assert_idle, _assert_no_interactive, _client, _provider, _rpc_request, _Server,
)

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")


class _ObservedAsyncio:
    """``asyncio`` for ``tools.mcp_oauth`` with ``sleep`` observed, not replaced."""

    def __init__(self):
        self.polling = asyncio.Event()
        self.delays: list[float] = []
        self.cancelled_in_timer = 0

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay, *args, **kwargs):
        self.delays.append(delay)
        self.polling.set()
        try:
            # health: allow HX013 -- delegates unchanged to the production fence poll timer; that pending timer IS the behaviour under test
            return await asyncio.sleep(delay, *args, **kwargs)
        except asyncio.CancelledError:
            self.cancelled_in_timer += 1
            raise


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "wait_for"])
async def test_cancel_while_initial_refresh_polls_a_peer_held_fence(tmp_path, monkeypatch, mode):
    import tools.mcp_oauth as fence_mod
    from tools.mcp_oauth import RefreshFenceTimeout, acquire_refresh_fence, release_refresh_fence

    provider, storage, interactive = await _provider(tmp_path, monkeypatch, expired=True)
    observed = _ObservedAsyncio()
    monkeypatch.setattr(fence_mod, "asyncio", observed)
    fence_path = storage._tokens_path()
    owner_fd = await acquire_refresh_fence(fence_path, timeout=0)  # the fixture is the peer holding the fence
    server = _Server()
    try:
        async with _client(provider, server) as client:
            send = client.send(_rpc_request())
            task = asyncio.create_task(send if mode == "cancel" else asyncio.wait_for(send, timeout=0.3))
            await asyncio.wait_for(observed.polling.wait(), 5)
            assert observed.delays and set(observed.delays) == {0.05}  # the real polling timer is pending
            assert server.log == []  # no refresh/resource/discovery/browser/code-exchange sent yet
            if mode == "cancel":
                task.cancel()
            with pytest.raises(asyncio.CancelledError if mode == "cancel" else TimeoutError):
                await task
        assert observed.cancelled_in_timer == 1  # the cancellation landed inside the real timer
        assert server.log == []
        _assert_no_interactive(server, interactive)
        _assert_idle(provider)  # semaphore free, no waiters, provider holds no fence
        with pytest.raises(RefreshFenceTimeout):
            await acquire_refresh_fence(fence_path, timeout=0)  # still the fixture's: cancel released nothing
    finally:
        release_refresh_fence(owner_fd)
    fd = await acquire_refresh_fence(fence_path, timeout=0)  # free only after the owner released it
    release_refresh_fence(fd)
