"""Tests for tools.safe_fetch — connection-level SSRF / DNS-rebind protection
(PLAT-F2).

The defining property: resolution, validation and connection happen as one
step, so the address that was validated is the address that is dialed. A
DNS-rebinding attacker cannot swap in a private/metadata target after the check.
"""

import asyncio
import socket
from unittest import mock

import httpcore
import httpx
import pytest

from tools import safe_fetch as SF


def _gai(ip, family=socket.AF_INET):
    sockaddr = (ip, 0) if family == socket.AF_INET else (ip, 0, 0, 0)
    return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]


@pytest.fixture(autouse=True)
def _block_private(monkeypatch):
    # Make the policy deterministic regardless of ambient config.
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
    from tools import url_safety
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


class TestPinSafeAddress:
    def test_public_hostname_pins_resolved_ip(self):
        with mock.patch("socket.getaddrinfo", return_value=_gai("93.184.216.34")):
            assert SF._pin_safe_address("example.com", 443) == "93.184.216.34"

    def test_private_resolution_blocked(self):
        with mock.patch("socket.getaddrinfo", return_value=_gai("10.0.0.1")):
            with pytest.raises(httpcore.ConnectError):
                SF._pin_safe_address("evil.example", 80)

    def test_metadata_resolution_blocked(self):
        with mock.patch("socket.getaddrinfo", return_value=_gai("169.254.169.254")):
            with pytest.raises(httpcore.ConnectError):
                SF._pin_safe_address("rebind.example", 80)

    def test_nat64_tunneled_metadata_blocked(self):
        with mock.patch(
            "socket.getaddrinfo",
            return_value=_gai("64:ff9b::169.254.169.254", socket.AF_INET6),
        ):
            with pytest.raises(httpcore.ConnectError):
                SF._pin_safe_address("tunnel.example", 443)

    def test_blocked_metadata_hostname_not_resolved(self):
        # Should reject on the hostname floor before any DNS lookup.
        with mock.patch("socket.getaddrinfo", side_effect=AssertionError("must not resolve")):
            with pytest.raises(httpcore.ConnectError):
                SF._pin_safe_address("metadata.google.internal", 80)

    def test_ip_literal_loopback_blocked(self):
        with pytest.raises(httpcore.ConnectError):
            SF._pin_safe_address("127.0.0.1", 80)

    def test_ip_literal_public_allowed(self):
        assert SF._pin_safe_address("93.184.216.34", 443) == "93.184.216.34"

    def test_dns_failure_fails_closed(self):
        with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
            with pytest.raises(httpcore.ConnectError):
                SF._pin_safe_address("nonexistent.example", 80)

    def test_mixed_resolution_pins_the_safe_address(self):
        infos = _gai("10.0.0.1") + _gai("93.184.216.34")
        with mock.patch("socket.getaddrinfo", return_value=infos):
            # The private address is skipped; the public one is pinned.
            assert SF._pin_safe_address("dualstack.example", 443) == "93.184.216.34"


class _FakeSyncInner:
    def __init__(self):
        self.dialed = None

    def connect_tcp(self, host, port, **kwargs):
        self.dialed = (host, port)
        return "STREAM"

    def connect_unix_socket(self, *a, **k):
        return "UNIX"

    def sleep(self, seconds):
        pass


class TestSyncGuardBackend:
    def test_dials_validated_ip_not_hostname(self):
        """Rebind defeat: the inner backend receives the validated IP, so the
        validated address is exactly the connected address."""
        inner = _FakeSyncInner()
        guard = SF._SsrfGuardSyncBackend(inner)
        with mock.patch("socket.getaddrinfo", return_value=_gai("93.184.216.34")):
            guard.connect_tcp("example.com", 443, timeout=5)
        assert inner.dialed == ("93.184.216.34", 443)

    def test_unsafe_resolution_never_dials(self):
        inner = _FakeSyncInner()
        guard = SF._SsrfGuardSyncBackend(inner)
        with mock.patch("socket.getaddrinfo", return_value=_gai("127.0.0.1")):
            with pytest.raises(httpcore.ConnectError):
                guard.connect_tcp("loopback.example", 80)
        assert inner.dialed is None


class _FakeAsyncInner:
    def __init__(self):
        self.dialed = None

    async def connect_tcp(self, host, port, **kwargs):
        self.dialed = (host, port)
        return "STREAM"

    async def connect_unix_socket(self, *a, **k):
        return "UNIX"

    async def sleep(self, seconds):
        pass


class TestAsyncGuardBackend:
    def test_async_dials_validated_ip(self):
        inner = _FakeAsyncInner()
        guard = SF._SsrfGuardAsyncBackend(inner)
        with mock.patch("socket.getaddrinfo", return_value=_gai("93.184.216.34")):
            asyncio.run(guard.connect_tcp("example.com", 443, timeout=5))
        assert inner.dialed == ("93.184.216.34", 443)

    def test_async_unsafe_never_dials(self):
        inner = _FakeAsyncInner()
        guard = SF._SsrfGuardAsyncBackend(inner)

        async def _run():
            with mock.patch("socket.getaddrinfo", return_value=_gai("10.0.0.1")):
                await guard.connect_tcp("evil.example", 80)

        with pytest.raises(httpcore.ConnectError):
            asyncio.run(_run())
        assert inner.dialed is None


class TestClientFactories:
    def test_safe_client_uses_guard_backend(self):
        client = SF.safe_client(timeout=5)
        try:
            transport = client._transport
            assert isinstance(transport, SF.SsrfGuardTransport)
            assert isinstance(transport._pool._network_backend, SF._SsrfGuardSyncBackend)
        finally:
            client.close()

    def test_safe_async_client_uses_guard_backend(self):
        client = SF.safe_async_client(timeout=5)
        transport = client._transport
        assert isinstance(transport, SF.SsrfGuardAsyncTransport)
        assert isinstance(transport._pool._network_backend, SF._SsrfGuardAsyncBackend)

    def test_safe_client_blocks_private_connect_end_to_end(self):
        with mock.patch("socket.getaddrinfo", return_value=_gai("10.0.0.5")):
            with SF.safe_client(timeout=5) as client:
                with pytest.raises(httpx.ConnectError):
                    client.get("http://internal.attacker.test/")
