"""Routing policy tests; native OS primitives are never switched to another OS."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from tools import mcp_windows


@pytest.mark.parametrize('network', [None, True, [], {}, 'typo'])
@pytest.mark.asyncio
async def test_invalid_network_fails_before_opening_a_connection(network):
    with pytest.raises(mcp_windows.InvalidMcpNetworkError):
        async with mcp_windows.mcp_http_route('http://localhost:8080/mcp', network=network):
            pytest.fail('invalid network was accepted')


@pytest.mark.parametrize('url', ['https://example.com/mcp', 'file:///mcp', 'http://127.0.0.2/mcp'])
@pytest.mark.asyncio
async def test_explicit_windows_target_never_bridges_other_origins(url):
    with pytest.raises(mcp_windows.InvalidMcpNetworkError):
        async with mcp_windows.mcp_http_route(url, network='windows'):
            pytest.fail('non-loopback origin was accepted')


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_wsl_legacy_omission_stays_local_and_explicit_auto_or_windows_can_bridge(monkeypatch):
    # Substitute only WSL's environment-detection boundary; all socket/process
    # primitives still run on the real Linux kernel (WSL is Linux too).
    monkeypatch.setattr(mcp_windows, 'is_wsl', lambda: True)
    calls = []
    marker = object()

    @asynccontextmanager
    async def bridge(url, **kwargs):
        calls.append(url)
        yield marker

    monkeypatch.setattr(mcp_windows, 'windows_loopback_route', bridge)
    server = await asyncio.start_server(lambda reader, writer: writer.close(), '127.0.0.1', 0)
    url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/mcp'
    try:
        async with mcp_windows.mcp_http_route(url) as route:
            assert route is None
        assert not calls
        async with mcp_windows.mcp_http_route(url, network='windows') as route:
            assert route is marker
        assert calls == [url]
    finally:
        server.close()
        await server.wait_closed()
    # The listener is now absent. Omission is the pre-feature legacy state and
    # must fail closed in the backend namespace instead of retargeting Windows.
    async with mcp_windows.mcp_http_route(url) as route:
        assert route is None
    assert calls == [url]
    async with mcp_windows.mcp_http_route(url, network='auto') as route:
        assert route is marker
    assert calls == [url, url]
    async with mcp_windows.mcp_http_route(url, network='local') as route:
        assert route is None
    async with mcp_windows.mcp_http_route('https://example.com/mcp') as route:
        assert route is None
    assert calls == [url, url]


@pytest.mark.parametrize('transport', ['http', 'sse'])
@pytest.mark.parametrize('auth', ['', 'oauth'])
@pytest.mark.asyncio
async def test_bad_network_fails_fast_even_when_preflight_is_skipped(transport, auth):
    pytest.importorskip('mcp')
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask('invalid-network-fixture')
    config = {'url': 'http://localhost:1/mcp', 'network': 'typo',
              'transport': transport, 'auth': auth, 'skip_preflight': True}
    assert not await task._prepare_run(config)
    assert isinstance(task._error, mcp_windows.InvalidMcpNetworkError)
    assert task._ready.is_set()


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_native_windows_explicit_target_is_direct():
    async with mcp_windows.mcp_http_route('http://localhost:8080/mcp', network='windows') as route:
        assert route is None
