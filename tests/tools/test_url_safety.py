"""Tests for SSRF protection in url_safety module."""

from unittest.mock import patch

from tools.url_safety import is_safe_url


class TestIsSafeUrl:
    def test_public_url_allowed(self):
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

    def test_dns_failure_allowed(self):
        """Unresolvable hosts should pass — let HTTP client handle DNS errors."""
        import socket
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name resolution failed")):
            assert is_safe_url("https://nonexistent.example.com") is True

    def test_empty_url_blocked(self):
        assert is_safe_url("") is False

    def test_no_hostname_blocked(self):
        assert is_safe_url("http://") is False

    def test_public_ip_allowed(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert is_safe_url("https://example.com") is True
