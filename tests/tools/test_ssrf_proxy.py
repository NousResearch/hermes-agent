"""Tests for the in-process SSRF-filtering forward proxy (tools/ssrf_proxy.py).

These exercise the REAL proxy over real loopback sockets — a local origin HTTP
server, a real HTTP client speaking through the proxy — not mocks of the socket
layer. DNS resolution is monkeypatched (via loop.getaddrinfo) so we can assert
policy without depending on public DNS, but the connect + pipe path is real.
"""

from __future__ import annotations

import asyncio
import socket
import resource

import pytest

from tools.ssrf_proxy import SsrfFilteringProxy


# ---------------------------------------------------------------------------
# Helpers: a minimal loopback origin server + a raw client that speaks HTTP
# proxy (absolute-form GET) and CONNECT through the proxy.
# ---------------------------------------------------------------------------

async def _start_origin(handler):
    """Start a loopback TCP server running `handler(reader, writer)`; return (host, port, server)."""
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return "127.0.0.1", port, server


async def _http_get_via_proxy(proxy_host, proxy_port, target_host, target_port, path="/"):
    """Send an absolute-form HTTP GET through the proxy; return (status_line, body)."""
    reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
    req = (
        f"GET http://{target_host}:{target_port}{path} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        f"Connection: close\r\n\r\n"
    )
    writer.write(req.encode())
    await writer.drain()
    data = await reader.read(-1)
    writer.close()
    text = data.decode("latin-1", "replace")
    status = text.split("\r\n", 1)[0]
    body = text.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in text else ""
    return status, body


async def _connect_via_proxy(proxy_host, proxy_port, target_host, target_port):
    """Send CONNECT through the proxy; return (status_line, reader, writer)."""
    reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
    req = f"CONNECT {target_host}:{target_port} HTTP/1.1\r\nHost: {target_host}:{target_port}\r\n\r\n"
    writer.write(req.encode())
    await writer.drain()
    line = await reader.readuntil(b"\r\n")
    # consume the blank line after the status if a 200
    try:
        await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=0.5)
    except Exception:
        pass
    return line.decode("latin-1", "replace").strip(), reader, writer


def _patch_resolve(monkeypatch, mapping):
    """Patch loop.getaddrinfo so host -> list of IPs (per test), real connect otherwise."""
    async def fake_getaddrinfo(host, port, *args, **kwargs):
        ips = mapping.get(host)
        if ips is None:
            raise socket.gaierror(f"no test mapping for {host}")
        out = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            out.append((fam, socket.SOCK_STREAM, 6, "", (ip, port)))
        return out
    loop = asyncio.get_event_loop()
    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proxy_binds_loopback_ephemeral():
    async with SsrfFilteringProxy() as url:
        assert url.startswith("http://127.0.0.1:")
        port = int(url.rsplit(":", 1)[1])
        assert port != 0 and port > 1024


@pytest.mark.asyncio
async def test_proxy_allows_public_http(monkeypatch):
    """A public host (mapped to a real loopback origin) is proxied through."""
    async def origin(reader, writer):
        await reader.readuntil(b"\r\n")  # request line
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nHEY")
        await writer.drain()
        writer.close()

    ohost, oport, oserver = await _start_origin(origin)
    async with oserver:
        # Map a "public" hostname to the loopback origin, but tell the policy
        # the resolved IP is public (8.8.8.8) so it isn't blocked — then connect
        # to the origin. To keep it real, map directly to 127.0.0.1 but disable
        # private blocking for THIS test so loopback is allowed.
        _patch_resolve(monkeypatch, {"public.example": ["127.0.0.1"]})
        async with SsrfFilteringProxy(block_private=False) as url:
            phost, pport = url[len("http://"):].split(":")
            status, body = await _http_get_via_proxy(phost, int(pport), "public.example", oport)
            assert "200" in status
            assert body == "HEY"


@pytest.mark.asyncio
async def test_proxy_blocks_metadata(monkeypatch):
    """A host resolving to the cloud-metadata IP is refused, even block_private=False."""
    _patch_resolve(monkeypatch, {"evil.example": ["169.254.169.254"]})
    async with SsrfFilteringProxy(block_private=False) as url:
        phost, pport = url[len("http://"):].split(":")
        status, _ = await _http_get_via_proxy(phost, int(pport), "evil.example", 80)
        assert "403" in status


@pytest.mark.asyncio
async def test_proxy_blocks_private(monkeypatch):
    """A host resolving to a private IP is refused when block_private=True."""
    _patch_resolve(monkeypatch, {"lan.example": ["192.168.1.208"]})
    async with SsrfFilteringProxy(block_private=True) as url:
        phost, pport = url[len("http://"):].split(":")
        status, _ = await _http_get_via_proxy(phost, int(pport), "lan.example", 3000)
        assert "403" in status


@pytest.mark.asyncio
async def test_proxy_pins_validated_ip_mixed_answer(monkeypatch):
    """If ANY resolved address is blocked, the whole connection is refused."""
    _patch_resolve(monkeypatch, {"mixed.example": ["8.8.8.8", "169.254.169.254"]})
    async with SsrfFilteringProxy(block_private=True) as url:
        phost, pport = url[len("http://"):].split(":")
        status, _ = await _http_get_via_proxy(phost, int(pport), "mixed.example", 80)
        assert "403" in status


@pytest.mark.asyncio
async def test_proxy_connect_metadata_ip_literal_blocked(monkeypatch):
    """CONNECT to a raw metadata IP literal is refused."""
    _patch_resolve(monkeypatch, {"169.254.169.254": ["169.254.169.254"]})
    async with SsrfFilteringProxy(block_private=False) as url:
        phost, pport = url[len("http://"):].split(":")
        status, _r, _w = await _connect_via_proxy(phost, int(pport), "169.254.169.254", 443)
        assert "403" in status


@pytest.mark.asyncio
async def test_proxy_validates_port_still_blocks_private(monkeypatch):
    """Port is carried through; a private host on any port is still blocked."""
    _patch_resolve(monkeypatch, {"lan.example": ["10.0.0.5"]})
    async with SsrfFilteringProxy(block_private=True) as url:
        phost, pport = url[len("http://"):].split(":")
        for port in (22, 6379, 8080):
            status, _ = await _http_get_via_proxy(phost, int(pport), "lan.example", port)
            assert "403" in status


@pytest.mark.asyncio
async def test_proxy_malformed_request_refused(monkeypatch):
    """A malformed request line is refused cleanly, never an un-validated connect."""
    async with SsrfFilteringProxy() as url:
        phost, pport = url[len("http://"):].split(":")
        reader, writer = await asyncio.open_connection(phost, int(pport))
        writer.write(b"GARBAGE-NOT-HTTP\r\n\r\n")
        await writer.drain()
        data = await reader.read(-1)
        writer.close()
        assert b"400" in data


@pytest.mark.asyncio
async def test_proxy_connection_count_increments(monkeypatch):
    """connection_count reflects opened validated connections; blocked_count the refusals."""
    async def origin(reader, writer):
        await reader.readuntil(b"\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nX")
        await writer.drain()
        writer.close()

    ohost, oport, oserver = await _start_origin(origin)
    async with oserver:
        _patch_resolve(monkeypatch, {"pub.example": ["127.0.0.1"], "bad.example": ["169.254.169.254"]})
        proxy = SsrfFilteringProxy(block_private=False)
        async with proxy as url:
            phost, pport = url[len("http://"):].split(":")
            await _http_get_via_proxy(phost, int(pport), "pub.example", oport)
            await _http_get_via_proxy(phost, int(pport), "pub.example", oport)
            await _http_get_via_proxy(phost, int(pport), "bad.example", 80)
        assert proxy.connection_count == 2
        assert proxy.blocked_count == 1


@pytest.mark.asyncio
async def test_proxy_no_fd_leak(monkeypatch):
    """N sequential proxied requests leave the process fd count flat (handler cleanup)."""
    async def origin(reader, writer):
        await reader.readuntil(b"\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nY")
        await writer.drain()
        writer.close()

    ohost, oport, oserver = await _start_origin(origin)
    async with oserver:
        _patch_resolve(monkeypatch, {"pub.example": ["127.0.0.1"]})

        def _open_fds():
            import os
            try:
                return len(os.listdir(f"/proc/{os.getpid()}/fd"))
            except FileNotFoundError:
                # macOS: fall back to resource-based soft check via a dup probe.
                return len([f for f in _list_fds_macos()])

        # Warm up one full proxy lifecycle, measure, then run many more.
        async def one():
            async with SsrfFilteringProxy(block_private=False) as url:
                phost, pport = url[len("http://"):].split(":")
                await _http_get_via_proxy(phost, int(pport), "pub.example", oport)

        await one()
        before = _count_fds()
        for _ in range(15):
            await one()
        after = _count_fds()
        # Allow a small slop for interpreter/gc jitter, but no per-call leak.
        assert after - before <= 5, f"fd leak: before={before} after={after}"


def _count_fds():
    import os
    pid = os.getpid()
    proc_fd = f"/proc/{pid}/fd"
    try:
        return len(os.listdir(proc_fd))
    except FileNotFoundError:
        pass
    # macOS: count via lsof-free method using resource + probing is unreliable;
    # use the psutil-free approach of scanning with os.
    count = 0
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = min(soft, 4096)
    import os as _os
    for fd in range(limit):
        try:
            _os.fstat(fd)
            count += 1
        except OSError:
            pass
    return count


def _list_fds_macos():
    return []


@pytest.mark.asyncio
async def test_proxy_blocks_redirect_hop_to_metadata(monkeypatch):
    """The whole point: a PUBLIC page that 302-redirects to a metadata IP —
    the redirect hop (a NEW proxied request to the metadata host) is blocked,
    not just the initial URL. Simulates a client that follows redirects by
    issuing a second proxied request to the Location target."""
    async def public_origin(reader, writer):
        await reader.readuntil(b"\r\n")
        # 302 to the metadata endpoint
        writer.write(
            b"HTTP/1.1 302 Found\r\n"
            b"Location: http://metadata.evil:80/latest/meta-data/\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        writer.close()

    ohost, oport, oserver = await _start_origin(public_origin)
    async with oserver:
        # public.example → loopback origin (public per policy: block_private off);
        # metadata.evil → the cloud metadata IP (always blocked).
        _patch_resolve(monkeypatch, {
            "public.example": ["127.0.0.1"],
            "metadata.evil": ["169.254.169.254"],
        })
        async with SsrfFilteringProxy(block_private=False) as url:
            phost, pport = url[len("http://"):].split(":")
            # 1) initial fetch to the public page → 302 (allowed, proxied)
            status1, _ = await _http_get_via_proxy(phost, int(pport), "public.example", oport)
            assert "302" in status1
            # 2) client follows the redirect → NEW proxied request to metadata → BLOCKED
            status2, _ = await _http_get_via_proxy(phost, int(pport), "metadata.evil", 80)
            assert "403" in status2, "redirect hop to metadata must be blocked at the proxy"


@pytest.mark.asyncio
async def test_handler_tasks_registered_and_cancelled(monkeypatch):
    """A handler task registers in _tasks while in-flight and is cancelled on
    __aexit__ — proves the fd-leak guard is actually wired (regression for the
    _spawn-never-called bug: start_server calls _handle_client directly)."""
    # An origin that hangs so the handler stays in-flight (pipe never ends).
    stop = asyncio.Event()

    async def hang_origin(reader, writer):
        await reader.readuntil(b"\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")  # promises 100, sends 0
        await writer.drain()
        await stop.wait()  # hold the connection open
        writer.close()

    ohost, oport, oserver = await _start_origin(hang_origin)
    async with oserver:
        _patch_resolve(monkeypatch, {"pub.example": ["127.0.0.1"]})
        proxy = SsrfFilteringProxy(block_private=False)
        async with proxy as url:
            phost, pport = url[len("http://"):].split(":")
            # Kick off a request but DON'T await it — leave the handler in-flight.
            r, w = await asyncio.open_connection(phost, int(pport))
            w.write(
                f"GET http://pub.example:{oport}/ HTTP/1.1\r\nHost: x\r\n\r\n".encode()
            )
            await w.drain()
            # Give the proxy a moment to accept + spawn the handler task.
            await asyncio.sleep(0.2)
            assert len(proxy._tasks) >= 1, "handler task must be registered while in-flight"
            w.close()
        # After __aexit__: tasks cancelled + cleared, no leak.
        assert len(proxy._tasks) == 0
        stop.set()
