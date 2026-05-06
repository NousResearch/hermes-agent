"""Tests for SSRF protection in url_safety module."""

import socket
from unittest.mock import patch

from tools.url_safety import (
    is_safe_url,
    _is_blocked_ip,
    _global_allow_private_urls,
    _reset_allow_private_cache,
)

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

    def test_benchmark_ip_blocked_for_non_allowlisted_host(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("198.18.0.23", 0)),
        ]):
            assert is_safe_url("https://example.com/file.jpg") is False

    def test_qq_multimedia_hostname_allowed_with_benchmark_ip(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("198.18.0.23", 0)),
        ]):
            assert is_safe_url("https://multimedia.nt.qq.com.cn/download?id=123") is True

    def test_qq_multimedia_hostname_exception_is_exact_match(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("198.18.0.23", 0)),
        ]):
            assert is_safe_url("https://sub.multimedia.nt.qq.com.cn/download?id=123") is False

    def test_qq_multimedia_hostname_exception_requires_https(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("198.18.0.23", 0)),
        ]):
            assert is_safe_url("http://multimedia.nt.qq.com.cn/download?id=123") is False

    def test_qq_multimedia_hostname_dns_failure_still_blocked(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name resolution failed")):
            assert is_safe_url("https://multimedia.nt.qq.com.cn/download?id=123") is False


class TestIsBlockedIp:
    """Direct tests for the _is_blocked_ip helper."""

    @pytest.mark.parametrize("ip_str", [
        "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
        "169.254.169.254", "0.0.0.0", "224.0.0.1", "255.255.255.255",
        "100.64.0.1", "100.100.100.100", "100.127.255.254", "198.18.0.23",
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


class TestGlobalAllowPrivateUrls:
    """Tests for the security.allow_private_urls config toggle."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Reset the module-level toggle cache before and after each test."""
        _reset_allow_private_cache()
        yield
        _reset_allow_private_cache()

    def test_default_is_false(self, monkeypatch):
        """Toggle defaults to False when no env var or config is set."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        with patch("hermes_cli.config.read_raw_config", side_effect=Exception("no config")):
            assert _global_allow_private_urls() is False

    def test_env_var_true(self, monkeypatch):
        """HERMES_ALLOW_PRIVATE_URLS=true enables the toggle."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert _global_allow_private_urls() is True

    def test_env_var_1(self, monkeypatch):
        """HERMES_ALLOW_PRIVATE_URLS=1 enables the toggle."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "1")
        assert _global_allow_private_urls() is True

    def test_env_var_yes(self, monkeypatch):
        """HERMES_ALLOW_PRIVATE_URLS=yes enables the toggle."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "yes")
        assert _global_allow_private_urls() is True

    def test_env_var_false(self, monkeypatch):
        """HERMES_ALLOW_PRIVATE_URLS=false keeps it disabled."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
        assert _global_allow_private_urls() is False

    def test_config_security_section(self, monkeypatch):
        """security.allow_private_urls in config enables the toggle."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        cfg = {"security": {"allow_private_urls": True}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is True

    def test_config_browser_fallback(self, monkeypatch):
        """browser.allow_private_urls works as legacy fallback."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        cfg = {"browser": {"allow_private_urls": True}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is True

    def test_config_security_string_false_stays_disabled(self, monkeypatch):
        """Quoted false must not opt out of SSRF protection."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        cfg = {"security": {"allow_private_urls": "false"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is False

    def test_config_browser_string_false_stays_disabled(self, monkeypatch):
        """Legacy browser.allow_private_urls also normalises quoted false."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        cfg = {"browser": {"allow_private_urls": "false"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is False

    def test_config_security_takes_precedence_over_browser(self, monkeypatch):
        """security section is checked before browser section."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        cfg = {"security": {"allow_private_urls": True}, "browser": {"allow_private_urls": False}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is True

    def test_env_var_overrides_config(self, monkeypatch):
        """Env var takes priority over config."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
        cfg = {"security": {"allow_private_urls": True}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is False

    def test_result_is_cached(self, monkeypatch):
        """Second call uses cached result, doesn't re-read config."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert _global_allow_private_urls() is True
        # Change env after first call — should still be True (cached)
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
        assert _global_allow_private_urls() is True


class TestAllowPrivateUrlsIntegration:
    """Integration tests: is_safe_url respects the global toggle."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        _reset_allow_private_cache()
        yield
        _reset_allow_private_cache()

    def test_private_ip_allowed_when_toggle_on(self, monkeypatch):
        """Private IPs pass is_safe_url when toggle is enabled."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]):
            assert is_safe_url("http://router.local") is True

    def test_benchmark_ip_allowed_when_toggle_on(self, monkeypatch):
        """198.18.x.x (benchmark/OpenWrt proxy range) passes when toggle is on."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("198.18.23.183", 0)),
        ]):
            assert is_safe_url("https://nousresearch.com") is True

    def test_cgnat_allowed_when_toggle_on(self, monkeypatch):
        """CGNAT range (100.64.0.0/10) passes when toggle is on."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("100.100.100.100", 0)),
        ]):
            assert is_safe_url("http://tailscale-peer.example/") is True

    def test_localhost_allowed_when_toggle_on(self, monkeypatch):
        """Even localhost passes when toggle is on."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]):
            assert is_safe_url("http://localhost:8080/api") is True

    # --- Cloud metadata always blocked regardless of toggle ---

    def test_metadata_hostname_blocked_even_with_toggle(self, monkeypatch):
        """metadata.google.internal is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False

    def test_metadata_goog_blocked_even_with_toggle(self, monkeypatch):
        """metadata.goog is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("http://metadata.goog/computeMetadata/v1/") is False

    def test_metadata_ip_blocked_even_with_toggle(self, monkeypatch):
        """169.254.169.254 (AWS/GCP metadata IP) is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("169.254.169.254", 0)),
        ]):
            assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False

    def test_metadata_ipv6_blocked_even_with_toggle(self, monkeypatch):
        """fd00:ec2::254 (AWS IPv6 metadata) is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("fd00:ec2::254", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[fd00:ec2::254]/latest/") is False

    def test_ecs_metadata_blocked_even_with_toggle(self, monkeypatch):
        """169.254.170.2 (AWS ECS task metadata) is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("169.254.170.2", 0)),
        ]):
            assert is_safe_url("http://169.254.170.2/v2/credentials") is False

    def test_alibaba_metadata_blocked_even_with_toggle(self, monkeypatch):
        """100.100.100.200 (Alibaba Cloud metadata) is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("100.100.100.200", 0)),
        ]):
            assert is_safe_url("http://100.100.100.200/latest/meta-data/") is False

    def test_azure_wire_server_blocked_even_with_toggle(self, monkeypatch):
        """169.254.169.253 (Azure IMDS wire server) is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("169.254.169.253", 0)),
        ]):
            assert is_safe_url("http://169.254.169.253/") is False

    def test_entire_link_local_blocked_even_with_toggle(self, monkeypatch):
        """Any 169.254.x.x address is ALWAYS blocked (entire link-local range)."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("169.254.42.99", 0)),
        ]):
            assert is_safe_url("http://169.254.42.99/anything") is False

    def test_dns_failure_still_blocked_with_toggle(self, monkeypatch):
        """DNS failures are still blocked even with toggle on."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("fail")):
            assert is_safe_url("https://nonexistent.example.com") is False

    def test_empty_url_still_blocked_with_toggle(self, monkeypatch):
        """Empty URLs are still blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("") is False
