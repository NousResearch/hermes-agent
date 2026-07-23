"""Tests for SSRF protection in url_safety module."""

import socket
from unittest.mock import patch

import httpx

from tools.url_safety import (
    is_safe_url,
    async_is_safe_url,
    is_always_blocked_url,
    normalize_url_for_request,
    redirect_target_from_response,
    create_ssrf_safe_async_client,
    SSRFConnectionBlocked,
    _SSRFGuardedAsyncNetworkBackend,
    _MAX_SSRF_CONNECT_IPS,
    _resolved_http_connect_ips,
    _is_blocked_ip,
    _global_allow_private_urls,
    _reset_allow_private_cache,
)

import ipaddress
import pytest


class TestNormalizeUrlForRequest:
    def test_percent_encodes_non_ascii_path(self):
        assert (
            normalize_url_for_request("https://wttr.in/Köln")
            == "https://wttr.in/K%C3%B6ln"
        )

    def test_preserves_existing_percent_escapes(self):
        assert (
            normalize_url_for_request("https://wttr.in/K%C3%B6ln")
            == "https://wttr.in/K%C3%B6ln"
        )

    def test_preserves_reserved_query_syntax(self):
        assert (
            normalize_url_for_request("https://example.com/search?q=Köln&lang=de")
            == "https://example.com/search?q=K%C3%B6ln&lang=de"
        )

    def test_idna_encodes_hostname(self):
        assert (
            normalize_url_for_request("https://münich.example/Köln")
            == "https://xn--mnich-kva.example/K%C3%B6ln"
        )

    def test_repairs_space_between_scheme_and_authority(self):
        assert (
            normalize_url_for_request("https:// docs.openclaw.ai")
            == "https://docs.openclaw.ai"
        )

    def test_repairs_tab_between_scheme_and_authority(self):
        assert (
            normalize_url_for_request("https://	docs.openclaw.ai/path")
            == "https://docs.openclaw.ai/path"
        )

    def test_trims_but_preserves_path_and_query_space_semantics(self):
        assert (
            normalize_url_for_request(" https://example.com/a b?q=c d ")
            == "https://example.com/a%20b?q=c%20d"
        )

    def test_does_not_collapse_embedded_scheme_separator_in_query(self):
        assert (
            normalize_url_for_request("https://example.com/r?next=https:// evil.example")
            == "https://example.com/r?next=https://%20evil.example"
        )


class TestIsSafeUrl:
    def test_public_url_allowed(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert is_safe_url("https://example.com/image.png") is True

    def test_ftp_scheme_blocked(self):
        """Only http/https should be allowed for fetch tools."""
        assert is_safe_url("ftp://example.com/file.txt") is False

    def test_missing_scheme_blocked(self):
        """Bare host/path should be rejected to avoid ambiguous handling."""
        assert is_safe_url("example.com/path") is False

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

    def test_ipv6_scope_id_link_local_blocked(self):
        """fe80::1%eth0 — a scope-ID-bearing link-local address must not bypass
        the guard. ``ipaddress.ip_address`` rejects the ``%scope`` suffix, so
        the scope must be stripped before the block check rather than skipped.
        """
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("fe80::1%eth0", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[fe80::1%eth0]/") is False

    def test_ipv6_scope_id_loopback_blocked(self):
        """::1%lo — scoped IPv6 loopback must still be blocked."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::1%lo", 0, 0, 0)),
        ]):
            assert is_safe_url("http://[::1%lo]/") is False

    def test_unparseable_ip_after_scope_strip_fails_closed(self):
        """An address that is still unparseable after stripping the scope ID
        must fail closed (block), not be silently skipped."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("not-an-ip%garbage", 0, 0, 0)),
        ]):
            assert is_safe_url("http://example.invalid/") is False

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


class TestAsyncIsSafeUrl:
    """async_is_safe_url must match is_safe_url (runs DNS in a thread pool)."""

    @pytest.mark.asyncio
    async def test_public_url_allowed(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert await async_is_safe_url("https://example.com/x") is True

    @pytest.mark.asyncio
    async def test_localhost_blocked(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]):
            assert await async_is_safe_url("http://localhost:8080/") is False


class TestSSRFGuardedHttpxClient:
    def test_connect_resolution_caps_safe_ip_candidates(self):
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"93.184.216.{idx}", 80))
            for idx in range(1, _MAX_SSRF_CONNECT_IPS + 4)
        ]

        with patch("socket.getaddrinfo", return_value=answers):
            ips = _resolved_http_connect_ips("example.com", 80, "http")

        assert len(ips) == _MAX_SSRF_CONNECT_IPS
        assert ips[0] == "93.184.216.1"
        assert ips[-1] == f"93.184.216.{_MAX_SSRF_CONNECT_IPS}"

    def test_connect_resolution_checks_private_ip_beyond_candidate_cap(self):
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"93.184.216.{idx}", 80))
            for idx in range(1, _MAX_SSRF_CONNECT_IPS + 1)
        ]
        answers.append(
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))
        )

        with patch("socket.getaddrinfo", return_value=answers):
            with pytest.raises(SSRFConnectionBlocked, match="metadata"):
                _resolved_http_connect_ips("example.com", 80, "http")

    @pytest.mark.asyncio
    async def test_async_client_dials_validated_ip_not_hostname(self, monkeypatch):
        """Direct httpx fetches should connect to the vetted IP, not re-resolve hostnames."""
        import httpcore
        from httpcore._backends.auto import AutoBackend

        for proxy_var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            monkeypatch.delenv(proxy_var, raising=False)

        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port, *args, **kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            ],
        )

        connect_attempts = []

        async def fake_connect_tcp(
            self,
            host,
            port,
            timeout=None,
            local_address=None,
            socket_options=None,
        ):
            connect_attempts.append((host, port))
            raise httpcore.ConnectError("stop before network")

        monkeypatch.setattr(AutoBackend, "connect_tcp", fake_connect_tcp)

        async with create_ssrf_safe_async_client(timeout=0.01, trust_env=False) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get("http://example.com/image.png")

        assert connect_attempts == [("93.184.216.34", 80)]

    @pytest.mark.asyncio
    async def test_async_backend_blocks_unix_socket_connects(self):
        import contextvars

        backend = _SSRFGuardedAsyncNetworkBackend(contextvars.ContextVar("test_schemes"))

        with pytest.raises(SSRFConnectionBlocked, match="Unix socket"):
            await backend.connect_unix_socket("/tmp/hermes.sock")

    def test_async_client_rejects_unpatchable_custom_transport(self):
        class CustomTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, request=request)

        with pytest.raises(SSRFConnectionBlocked, match="Unsupported async httpx transport"):
            create_ssrf_safe_async_client(transport=CustomTransport())

    @pytest.mark.asyncio
    async def test_async_client_preserves_env_proxy_mounts(self, monkeypatch):
        """Installing the guard must not disable or rewrite httpx env proxy setup."""
        for proxy_var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            monkeypatch.delenv(proxy_var, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")

        client = create_ssrf_safe_async_client(timeout=0.01)
        try:
            proxy_transports = [
                transport
                for transport in client.__dict__.get("_mounts", {}).values()
                if transport is not None
            ]
            assert proxy_transports
            assert type(client._transport._pool._network_backend).__name__ == (
                "_SSRFGuardedAsyncNetworkBackend"
            )
            assert all(
                type(transport._pool._network_backend).__name__
                != "_SSRFGuardedAsyncNetworkBackend"
                for transport in proxy_transports
            )
        finally:
            await client.aclose()


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


class TestIsAlwaysBlockedUrl:
    """The always-blocked floor — cloud metadata only, narrower than is_safe_url."""

    # -- The sentinel set that must always block --------------------------------

    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data/",            # AWS / GCP / Azure / DO / Oracle
        "http://169.254.169.253/metadata/instance",              # Azure IMDS wire server
        "http://169.254.170.2/v2/credentials",                   # AWS ECS task metadata
        "http://100.100.100.200/latest/meta-data/",              # Alibaba Cloud
        "http://169.254.42.1/",                                  # Any /16 link-local
    ])
    def test_literal_imds_ips_always_blocked(self, url):
        """Literal IMDS IPs and the /16 link-local range always block."""
        assert is_always_blocked_url(url) is True

    def test_gcp_metadata_hostname_always_blocked_even_without_dns(self):
        """metadata.google.internal blocks by hostname, no DNS needed."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
            assert is_always_blocked_url("http://metadata.google.internal/") is True

    def test_hostname_resolving_to_imds_always_blocked(self):
        """Attacker-controlled hostname resolving to IMDS still blocks."""
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("169.254.169.254", 0)),
        ]):
            assert is_always_blocked_url("http://attacker-controlled.example.com/") is True

    def test_scope_id_imds_in_floor_blocked(self):
        """A scope-ID suffix on an IPv4-mapped IMDS address resolving in the
        always-blocked floor must be caught after the scope is stripped, not
        skipped as unparseable."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:169.254.169.254%eth0", 0, 0, 0)),
        ]):
            assert is_always_blocked_url("http://attacker-controlled.example.com/") is True

    # -- Things the floor must NOT block ----------------------------------------

    def test_public_url_not_blocked(self):
        assert is_always_blocked_url("https://example.com/path") is False

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8080/",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://100.64.0.1/",  # CGNAT — blocked by is_safe_url but not by the floor
    ])
    def test_ordinary_private_urls_not_in_floor(self, url):
        """Floor is narrower than is_safe_url — ordinary private URLs pass."""
        assert is_always_blocked_url(url) is False

    def test_dns_failure_not_in_floor(self):
        """DNS failure on a non-sentinel hostname = not always-blocked.

        Caller's ordinary fail-closed path (is_safe_url) handles that case.
        """
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("fail")):
            assert is_always_blocked_url("http://nonexistent.example.com/") is False

    def test_empty_url_not_in_floor(self):
        """Empty URL falls through — caller decides what to do with a malformed URL."""
        assert is_always_blocked_url("") is False

    def test_malformed_url_not_in_floor(self):
        """Parse errors don't claim always-blocked status."""
        assert is_always_blocked_url("not a url at all") is False

    def test_floor_ignores_allow_private_urls_toggle(self, monkeypatch):
        """security.allow_private_urls can NOT unblock cloud metadata."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert is_always_blocked_url("http://169.254.169.254/") is True


class TestIPv4MappedIPv6SSRF:
    """Regression tests for SSRF bypass via IPv4-mapped IPv6 addresses.

    DNS resolvers may return ``::ffff:x.x.x.x`` for IPv4-only hosts.
    Python's ipaddress module treats these as distinct from the plain
    IPv4 address, so ``ip in frozenset({IPv4Address(...)})`` and
    ``ip in IPv4Network(...)`` both return False.  Without explicit
    handling, an attacker could use IPv4-mapped addresses to bypass
    all SSRF protections.
    """

    # ── _is_blocked_ip direct tests ──

    @pytest.mark.parametrize("ip_str", [
        "::ffff:100.64.0.1",       # CGNAT start
        "::ffff:100.100.100.200",  # Alibaba Cloud metadata (in CGNAT range)
        "::ffff:100.127.255.254",  # CGNAT end
        "::ffff:169.254.42.99",    # Link-local (non-metadata)
        "::ffff:0.0.0.0",          # Unspecified
        "::ffff:224.0.0.1",        # Multicast
    ])
    def test_ipv4_mapped_blocked_ips(self, ip_str):
        """IPv4-mapped IPv6 addresses that should be blocked."""
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is True, f"{ip_str} should be blocked"

    @pytest.mark.parametrize("ip_str", [
        "::ffff:8.8.8.8",          # Public DNS
        "::ffff:93.184.216.34",    # example.com
        "::ffff:100.0.0.1",        # Not in CGNAT range
    ])
    def test_ipv4_mapped_allowed_ips(self, ip_str):
        """IPv4-mapped IPv6 addresses that should be allowed."""
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is False, f"{ip_str} should be allowed"

    # ── is_safe_url integration tests: always-blocked metadata IPs ──

    def test_ipv4_mapped_aws_metadata_blocked(self):
        """::ffff:169.254.169.254 (AWS metadata) must always be blocked."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:169.254.169.254", 0, 0, 0)),
        ]):
            assert is_safe_url("http://aws-metadata.internal/") is False

    def test_ipv4_mapped_ecs_metadata_blocked(self):
        """::ffff:169.254.170.2 (AWS ECS task metadata) must always be blocked."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:169.254.170.2", 0, 0, 0)),
        ]):
            assert is_safe_url("http://ecs-metadata.internal/") is False

    def test_ipv4_mapped_azure_wire_server_blocked(self):
        """::ffff:169.254.169.253 (Azure IMDS wire server) must always be blocked."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:169.254.169.253", 0, 0, 0)),
        ]):
            assert is_safe_url("http://azure-metadata.internal/") is False

    def test_ipv4_mapped_alibaba_metadata_blocked(self):
        """::ffff:100.100.100.200 (Alibaba Cloud metadata) must always be blocked."""
        with patch("socket.getaddrinfo", return_value=[
            (10, 1, 6, "", ("::ffff:100.100.100.200", 0, 0, 0)),
        ]):
            assert is_safe_url("http://aliyun-metadata.internal/") is False


class _FakeResponse:
    """Minimal stand-in for an httpx response as seen inside a response hook."""

    def __init__(self, *, is_redirect, location=None, url="", next_request=None):
        self.is_redirect = is_redirect
        self.headers = {"location": location} if location else {}
        self.url = url
        self.next_request = next_request


class _FakeNextRequest:
    def __init__(self, url):
        self.url = url


class TestRedirectTargetFromResponse:
    """redirect_target_from_response is the SSRF-guard boundary for httpx hooks.

    Inside httpx AsyncClient response hooks, ``response.next_request`` is often
    ``None`` even for a real redirect, so a guard keyed only on it silently
    never fires. Resolving from the ``Location`` header closes that hole.
    """

    def test_absolute_location_without_next_request(self):
        # The exact bypass: redirect present, next_request unset, private target.
        resp = _FakeResponse(
            is_redirect=True,
            location="http://169.254.169.254/latest/meta-data",
            url="https://public.example/image.png",
        )
        assert (
            redirect_target_from_response(resp)
            == "http://169.254.169.254/latest/meta-data"
        )

    def test_relative_location_is_resolved_against_response_url(self):
        resp = _FakeResponse(
            is_redirect=True,
            location="/redir",
            url="https://public.example/image.png",
        )
        assert redirect_target_from_response(resp) == "https://public.example/redir"

    def test_non_redirect_returns_none(self):
        resp = _FakeResponse(is_redirect=False, location="http://169.254.169.254/")
        assert redirect_target_from_response(resp) is None

    def test_falls_back_to_next_request_when_no_location(self):
        resp = _FakeResponse(
            is_redirect=True,
            next_request=_FakeNextRequest("http://10.0.0.1/meta"),
        )
        assert redirect_target_from_response(resp) == "http://10.0.0.1/meta"

    def test_no_location_no_next_request_returns_none(self):
        resp = _FakeResponse(is_redirect=True)
        assert redirect_target_from_response(resp) is None
