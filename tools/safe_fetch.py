"""Connection-level SSRF protection for httpx (PLAT-F2).

``is_safe_url`` is a *pre-flight* check: it resolves the hostname, validates the
IPs, and returns. The HTTP client then resolves the hostname **again** at
connect time. An attacker-controlled DNS server with TTL=0 can return a public
IP for the pre-flight check and a private/metadata IP for the actual connect —
a DNS-rebinding TOCTOU bypass (documented as a limitation in ``url_safety``).

This module closes that window. It wraps httpx's httpcore network backend so
that resolution, validation, and connection happen as **one** step: the backend
resolves the host, validates every candidate address with
``url_safety.is_connectable_ip``, and dials a *validated* IP directly. Because
the address that was validated is the address that is connected to, a rebind
cannot swap in a private target after the check.

TLS is unaffected: httpcore derives ``server_hostname`` (SNI + certificate
verification) from the request origin / ``sni_hostname`` extension, independently
of the address handed to ``connect_tcp`` — so connecting to the pinned IP still
presents and verifies the original hostname.

Usage:
    from tools.safe_fetch import safe_client, safe_async_client

    with safe_client(timeout=10, follow_redirects=True) as client:
        r = client.get(url)

    async with safe_async_client(timeout=10, follow_redirects=True) as client:
        r = await client.get(url)

Every connection the client makes — including redirect targets httpx follows —
is validated, so this also covers redirect-based SSRF without a separate hook.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Optional

import httpcore
import httpx

from tools.url_safety import is_blocked_hostname, is_connectable_ip


def _scheme_for_port(port: int) -> str:
    """Approximate the URL scheme from the port for the trusted-host (HTTPS)
    private-IP exception. 443 → https, everything else → http."""
    return "https" if port == 443 else "http"


def _pin_safe_address(host: str, port: int) -> str:
    """Resolve ``host`` and return a single validated IP string to connect to.

    Raises :class:`httpcore.ConnectError` (fail closed) when the host is a
    blocked metadata hostname, when DNS resolution fails, or when no resolved
    address passes the SSRF policy. This is the single point where resolution
    and validation are unified, eliminating the rebind TOCTOU.
    """
    raw = (host or "").strip()
    normalized = raw.lower().rstrip(".").strip("[]")
    scheme = _scheme_for_port(port)

    if is_blocked_hostname(normalized):
        raise httpcore.ConnectError(f"SSRF: blocked hostname {host!r}")

    # IP-literal host (e.g. http://[::1]/ or http://127.0.0.1/): validate the
    # literal directly — no DNS, so no rebind risk, but the address class still
    # has to pass policy.
    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        literal = None
    if literal is not None:
        if is_connectable_ip(literal, normalized, scheme):
            return str(literal)
        raise httpcore.ConnectError(f"SSRF: blocked address {host!r}")

    # Hostname: resolve once, validate every candidate, dial the first safe one.
    try:
        addr_info = socket.getaddrinfo(raw, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise httpcore.ConnectError(f"SSRF: DNS resolution failed for {host!r}") from exc

    for _family, _type, _proto, _canon, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            candidate = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if is_connectable_ip(candidate, normalized, scheme):
            return ip_str

    raise httpcore.ConnectError(
        f"SSRF: no resolved address for {host!r} passed the safety policy"
    )


class _SsrfGuardSyncBackend:
    """Wraps a sync httpcore network backend, validating + pinning each TCP
    connection target. Delegates everything else unchanged."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options=None,
    ):
        pinned = _pin_safe_address(host, port)
        return self._inner.connect_tcp(
            pinned,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(self, *args, **kwargs):
        return self._inner.connect_unix_socket(*args, **kwargs)

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


class _SsrfGuardAsyncBackend:
    """Async counterpart of :class:`_SsrfGuardSyncBackend`. DNS resolution runs
    in a worker thread so it never blocks the event loop."""

    def __init__(self, inner) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options=None,
    ):
        import anyio

        pinned = await anyio.to_thread.run_sync(_pin_safe_address, host, port)
        return await self._inner.connect_tcp(
            pinned,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, *args, **kwargs):
        return await self._inner.connect_unix_socket(*args, **kwargs)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class SsrfGuardTransport(httpx.HTTPTransport):
    """``httpx.HTTPTransport`` that validates and pins every connection target
    at the network layer (sync)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pool._network_backend = _SsrfGuardSyncBackend(self._pool._network_backend)


class SsrfGuardAsyncTransport(httpx.AsyncHTTPTransport):
    """``httpx.AsyncHTTPTransport`` that validates and pins every connection
    target at the network layer (async)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pool._network_backend = _SsrfGuardAsyncBackend(self._pool._network_backend)


def safe_client(*, verify=True, http2: bool = False, **client_kwargs) -> httpx.Client:
    """Return an ``httpx.Client`` with connection-level SSRF/DNS-rebind
    protection. Extra kwargs (timeout, headers, follow_redirects, event_hooks,
    …) are forwarded to ``httpx.Client``."""
    transport = SsrfGuardTransport(verify=verify, http2=http2)
    return httpx.Client(transport=transport, **client_kwargs)


def safe_async_client(
    *, verify=True, http2: bool = False, **client_kwargs
) -> httpx.AsyncClient:
    """Async counterpart of :func:`safe_client`."""
    transport = SsrfGuardAsyncTransport(verify=verify, http2=http2)
    return httpx.AsyncClient(transport=transport, **client_kwargs)
