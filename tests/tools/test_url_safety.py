"""Tests for SSRF protection in url_safety module."""

import socket
from unittest.mock import patch

from tools.url_safety import is_safe_url, _is_blocked_ip

import ipaddress
import pytest


class TestIsSafeUrl:
    def test_public_url_allowed(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert is_safe_url("https://example.com/image.png") is True

    def test_localhost_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]):
            assert is_safe_url("http://localhost:8080/secret") is False

    def test_loopback_ip_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]):
            assert is_safe_url("http://127.0.0.1/admin") is False

    def test_private_10_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("10.0.0.1", 0)),
        ]):
            assert is_safe_url("http://internal-service.local/api") is False

    def test_private_172_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("172.16.0.1", 0)),
        ]):
            assert is_safe_url("http://private.corp/data") is False

    def test_private_192_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]):
            assert is_safe_url("http://router.local") is False

    def test_link_local_169_254_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("169.254.169.254", 0)),
        ]):
            assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False

    def test_metadata_google_internal_blocked(self):
        assert is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False

    def test_ipv6_loopback_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::1", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[::1]:8080/") is False

    def test_dns_failure_blocked(self):
        """DNS failures now fail closed — block the request."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name resolution failed")):
            assert is_safe_url("https://nonexistent.example.com") is False

    def test_empty_url_blocked(self):
        assert is_safe_url("") is False

    def test_no_hostname_blocked(self):
        assert is_safe_url("http://") is False

    def test_public_ip_allowed(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert is_safe_url("https://example.com") is True

    # ── New tests for hardened SSRF protection ──

    def test_cgnat_100_64_blocked(self):
        """100.64.0.0/10 (CGNAT/Shared Address Space) is NOT covered by
        ipaddress.is_private — must be blocked explicitly."""
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("100.64.0.1", 0)),
        ]):
            assert is_safe_url("http://some-cgnat-host.example/") is False

    def test_cgnat_100_127_blocked(self):
        """Upper end of CGNAT range (100.127.255.255)."""
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("100.127.255.254", 0)),
        ]):
            assert is_safe_url("http://tailscale-peer.example/") is False

    def test_multicast_blocked(self):
        """Multicast addresses (224.0.0.0/4) not caught by is_private."""
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("224.0.0.251", 0)),
        ]):
            assert is_safe_url("http://mdns-host.local/") is False

    def test_multicast_ipv6_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("ff02::1", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[ff02::1]/") is False

    def test_ipv4_mapped_ipv6_loopback_blocked(self):
        """::ffff:127.0.0.1 — IPv4-mapped IPv6 loopback."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:127.0.0.1", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[::ffff:127.0.0.1]/") is False

    def test_ipv4_mapped_ipv6_metadata_blocked(self):
        """::ffff:169.254.169.254 — IPv4-mapped IPv6 cloud metadata."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:169.254.169.254", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[::ffff:169.254.169.254]/") is False

    def test_unspecified_address_blocked(self):
        """0.0.0.0 — unspecified address, can bind to all interfaces."""
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("0.0.0.0", 0)),
        ]):
            assert is_safe_url("http://0.0.0.0/") is False

    def test_unexpected_error_fails_closed(self):
        """Unexpected exceptions should block, not allow."""
        with patch("tools.url_safety.urlparse", side_effect=ValueError("bad url")):
            assert is_safe_url("http://evil.com/") is False

    def test_metadata_goog_blocked(self):
        assert is_safe_url("http://metadata.goog/computeMetadata/v1/") is False

    def test_ipv6_unique_local_blocked(self):
        """fc00::/7 — IPv6 unique local addresses."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("fd12::1", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[fd12::1]/internal") is False

    def test_non_cgnat_100_allowed(self):
        """100.0.0.1 is NOT in CGNAT range (100.64.0.0/10), should be allowed."""
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("100.0.0.1", 0)),
        ]):
            # 100.0.0.1 is a global IP, not in CGNAT range
            assert is_safe_url("http://legit-host.example/") is True


class TestIsBlockedIp:
    """Direct tests for the _is_blocked_ip helper."""

    @pytest.mark.parametrize("ip_str", [
        "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
        "169.254.169.254", "0.0.0.0", "224.0.0.1", "255.255.255.255",
        "100.64.0.1", "100.100.100.100", "100.127.255.254",
        "::1", "fe80::1", "fc00::1", "fd12::1", "ff02::1",
        "::ffff:127.0.0.1", "::ffff:169.254.169.254",
    ])
    def test_blocked_ips(self, ip_str):
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is True, f"{ip_str} should be blocked"

    @pytest.mark.parametrize("ip_str", [
        "8.8.8.8", "93.184.216.34", "1.1.1.1", "100.0.0.1",
        "2606:4700::1", "2001:4860:4860::8888",
    ])
    def test_allowed_ips(self, ip_str):
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is False, f"{ip_str} should be allowed"
