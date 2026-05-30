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
    is_blocked_ip_address,
    is_valid_ip_address,
    is_always_blocked_ip_address,
    _is_blocked_ip,
    _global_allow_private_urls,
    _reset_allow_private_cache,
)

import ipaddress
import pytest


def _resolves_to(*ips):
    """Patch ``socket.getaddrinfo`` so any hostname resolves to *ips*.

    The address family field is unused by url_safety (it reads ``sockaddr[0]``
    only), so one shape works for both IPv4 and IPv6 answers.
    """
    return patch(
        "socket.getaddrinfo",
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips
        ],
    )


class TestNormalizeUrlForRequest:
    @pytest.mark.parametrize("raw, expected", [
        # non-ASCII path is percent-encoded
        ("https://wttr.in/Köln", "https://wttr.in/K%C3%B6ln"),
        # existing escapes are preserved (idempotent)
        ("https://wttr.in/K%C3%B6ln", "https://wttr.in/K%C3%B6ln"),
        # hostname is IDNA-encoded
        ("https://münich.example/Köln", "https://xn--mnich-kva.example/K%C3%B6ln"),
    ])
    def test_encodes_url_parts(self, raw, expected):
        assert normalize_url_for_request(raw) == expected


    def test_does_not_collapse_embedded_scheme_separator_in_query(self):
        assert (
            normalize_url_for_request("https://example.com/r?next=https:// evil.example")
            == "https://example.com/r?next=https://%20evil.example"
        )


class TestIsSafeUrl:
    def test_public_url_allowed(self):
        with _resolves_to("93.184.216.34"):
            assert is_safe_url("https://example.com/image.png") is True

    @pytest.mark.parametrize("url", [
        "ftp://example.com/file.txt",  # only http/https allowed for fetch tools
        "example.com/path",            # bare host/path is ambiguous
        "http://",                     # no hostname
        "",                            # empty
    ])
    def test_unusable_urls_blocked(self, url):
        assert is_safe_url(url) is False

    @pytest.mark.parametrize("ip, url", [
        ("127.0.0.1", "http://localhost:8080/secret"),
        ("10.0.0.1", "http://internal-service.local/api"),
        ("::1", "http://[::1]:8080/"),
    ])
    def test_private_and_loopback_targets_blocked(self, ip, url):
        with _resolves_to(ip):
            assert is_safe_url(url) is False

    def test_dns_failure_blocked(self, monkeypatch):
        """DNS failures fail closed — block the request (no proxy configured)."""
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(var, raising=False)
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name resolution failed")):
            assert is_safe_url("https://nonexistent.example.com") is False


class TestProxyEnvironmentDnsDelegation:
    """When an HTTP proxy is configured, DNS is delegated to the proxy.

    Sandbox / proxy-only environments (Docker + Squid, NVIDIA OpenShell,
    iron-proxy egress sandboxes) block direct DNS at the network level;
    only HTTP(S) via the proxy works. is_safe_url must not fail closed on
    the pre-flight DNS check there — the proxy is the egress boundary.
    Regression tests for #32217 / PR #68469.
    """

    @pytest.fixture(autouse=True)
    def _clear_proxy_env(self, monkeypatch):
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(var, raising=False)

    def test_dns_failure_allowed_when_proxy_configured(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://host.docker.internal:9090")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("blocked at network level")):
            assert is_safe_url("https://api.openai.com/v1/models") is True

    def test_metadata_hostname_still_blocked_with_proxy(self, monkeypatch):
        """The blocked-hostname floor runs BEFORE the DNS skip."""
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("no dns")):
            assert is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False

    def test_literal_metadata_ip_still_blocked_with_proxy(self, monkeypatch):
        """Literal IPs never take the DNS-failure path — floor intact."""
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False


    def test_ipv6_scope_id_link_local_blocked(self):
        """fe80::1%eth0 — a scope-ID-bearing link-local address must not bypass
        the guard. ``ipaddress.ip_address`` rejects the ``%scope`` suffix, so
        the scope must be stripped before the block check rather than skipped.
        """
        with _resolves_to("fe80::1%eth0"):
            assert is_safe_url("http://[fe80::1%eth0]/") is False

    def test_unparseable_ip_after_scope_strip_fails_closed(self):
        """An address that is still unparseable after stripping the scope ID
        must fail closed (block), not be silently skipped."""
        with _resolves_to("not-an-ip%garbage"):
            assert is_safe_url("http://example.invalid/") is False

    def test_unexpected_error_fails_closed(self):
        """Unexpected exceptions should block, not allow."""
        with patch("tools.url_safety.urlparse", side_effect=ValueError("bad url")):
            assert is_safe_url("http://evil.com/") is False

    def test_benchmark_ip_blocked_for_non_allowlisted_host(self):
        with _resolves_to("198.18.0.23"):
            assert is_safe_url("https://example.com/file.jpg") is False

    @pytest.mark.parametrize("url, expected", [
        # the allowlisted host itself, over https
        ("https://multimedia.nt.qq.com.cn/download?id=123", True),
        # exception is an exact host match — subdomains stay blocked
        ("https://sub.multimedia.nt.qq.com.cn/download?id=123", False),
        # ... and requires https
        ("http://multimedia.nt.qq.com.cn/download?id=123", False),
    ])
    def test_qq_multimedia_hostname_exception(self, url, expected):
        with _resolves_to("198.18.0.23"):
            assert is_safe_url(url) is expected


class TestAsyncIsSafeUrl:
    """async_is_safe_url must match is_safe_url (runs DNS in a thread pool)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ip, url, expected", [
        ("93.184.216.34", "https://example.com/x", True),
        ("127.0.0.1", "http://localhost:8080/", False),
    ])
    async def test_matches_sync_verdict(self, ip, url, expected):
        with _resolves_to(ip):
            assert await async_is_safe_url(url) is expected


class TestSSRFGuardedHttpxClient:
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
    """Direct tests for the _is_blocked_ip helper — one per blocked range.

    ``::ffff:`` forms cover the IPv4-mapped IPv6 bypass: Python's ipaddress
    module treats them as distinct from the plain IPv4 address, so membership
    checks miss them without explicit handling.
    """

    @pytest.mark.parametrize("ip_str", [
        "169.254.169.254",          # link-local / cloud metadata
        "0.0.0.0",                  # unspecified
        "224.0.0.1",                # multicast (not is_private)
        "100.64.0.1",               # CGNAT boundary (not is_private)
        "198.18.0.23",              # benchmark range
        "fd12::1",                  # IPv6 unique local
        "::ffff:169.254.169.254",   # IPv4-mapped IPv6 metadata
    ])
    def test_blocked_ips(self, ip_str):
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is True, f"{ip_str} should be blocked"

    @pytest.mark.parametrize("ip_str", [
        "100.0.0.1",       # just below the CGNAT range
        "2606:4700::1",    # public IPv6
    ])
    def test_allowed_ips(self, ip_str):
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is False, f"{ip_str} should be allowed"


class TestRemoteIpHelpers:
    """Direct tests for browser/client-reported remote IP classification."""

    def test_remote_private_ip_is_blocked(self):
        assert is_blocked_ip_address("192.168.1.10") is True

    def test_remote_public_ip_is_allowed(self):
        assert is_blocked_ip_address("93.184.216.34") is False

    def test_remote_ipv6_private_ip_is_blocked(self):
        assert is_blocked_ip_address("fc00::1") is True

    def test_remote_ipv6_public_ip_is_allowed(self):
        assert is_blocked_ip_address("2606:4700::1") is False

    def test_remote_ipv4_mapped_ipv6_metadata_ip_is_always_blocked(self):
        assert is_always_blocked_ip_address("::ffff:169.254.169.254") is True

    def test_remote_metadata_ip_is_always_blocked(self):
        assert is_always_blocked_ip_address("169.254.169.254") is True

    def test_remote_ordinary_private_ip_not_always_blocked(self):
        assert is_always_blocked_ip_address("192.168.1.10") is False

    def test_invalid_remote_ip_fails_closed(self):
        assert is_blocked_ip_address("not an ip") is True
        assert is_always_blocked_ip_address("not an ip") is False
        assert is_valid_ip_address("not an ip") is False

    def test_valid_remote_ip_syntax_is_detected(self):
        assert is_valid_ip_address("2606:4700::1") is True


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


    def test_config_security_string_false_stays_disabled(self, monkeypatch):
        """Quoted false must not opt out of SSRF protection."""
        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        cfg = {"security": {"allow_private_urls": "false"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _global_allow_private_urls() is False


    @pytest.mark.parametrize(
        "profile_order",
        [("allowed", "blocked"), ("blocked", "allowed")],
        ids=["allowed-then-blocked", "blocked-then-allowed"],
    )
    def test_profile_scoped_config_does_not_reuse_another_profiles_opt_out(
        self, tmp_path, monkeypatch, profile_order
    ):
        """Multiplexed profiles must resolve their own private-URL policy."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
        allowed_home = tmp_path / "allowed"
        blocked_home = tmp_path / "blocked"
        allowed_home.mkdir()
        blocked_home.mkdir()
        (allowed_home / "config.yaml").write_text(
            "security:\n  allow_private_urls: true\n", encoding="utf-8"
        )
        (blocked_home / "config.yaml").write_text(
            "security:\n  allow_private_urls: false\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [(2, 1, 6, "", ("10.0.0.8", 0))],
        )

        def under_profile(home):
            token = set_hermes_home_override(home)
            try:
                return is_safe_url("http://profile-private.test/resource")
            finally:
                reset_hermes_home_override(token)

        homes = {"allowed": allowed_home, "blocked": blocked_home}
        expected = {"allowed": True, "blocked": False}
        for profile in profile_order:
            assert under_profile(homes[profile]) is expected[profile]


class TestAllowPrivateUrlsIntegration:
    """Integration tests: is_safe_url respects the global toggle."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        _reset_allow_private_cache()
        yield
        _reset_allow_private_cache()

    @pytest.mark.parametrize("ip, url", [
        ("192.168.1.1", "http://router.local"),
        # 198.18.x.x (benchmark / OpenWrt proxy range) must pass too
        ("198.18.23.183", "https://nousresearch.com"),
    ])
    def test_private_ip_allowed_when_toggle_on(self, monkeypatch, ip, url):
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with _resolves_to(ip):
            assert is_safe_url(url) is True

    # --- Cloud metadata always blocked regardless of toggle ---

    @pytest.mark.parametrize("ip, url", [
        ("fd00:ec2::254", "http://[fd00:ec2::254]/latest/"),          # AWS IPv6 IMDS
        ("100.100.100.200", "http://100.100.100.200/latest/meta-data/"),  # Alibaba
    ])
    def test_metadata_ip_blocked_even_with_toggle(self, monkeypatch, ip, url):
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        with _resolves_to(ip):
            assert is_safe_url(url) is False

    def test_metadata_hostname_blocked_even_with_toggle(self, monkeypatch):
        """metadata.google.internal is ALWAYS blocked."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False

    def test_dns_failure_still_blocked_with_toggle(self, monkeypatch):
        """DNS failures are still blocked even with toggle on."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(var, raising=False)
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("fail")):
            assert is_safe_url("https://nonexistent.example.com") is False


class TestIsAlwaysBlockedUrl:
    """The always-blocked floor — cloud metadata only, narrower than is_safe_url."""

    # -- The sentinel set that must always block --------------------------------

    @pytest.mark.parametrize("url", [
        "http://169.254.42.1/",                      # any /16 link-local (incl. IMDS)
        "http://100.100.100.200/latest/meta-data/",   # Alibaba Cloud
    ])
    def test_literal_imds_ips_always_blocked(self, url):
        assert is_always_blocked_url(url) is True

    def test_gcp_metadata_hostname_always_blocked_even_without_dns(self):
        """metadata.google.internal blocks by hostname, no DNS needed."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
            assert is_always_blocked_url("http://metadata.google.internal/") is True

    def test_scope_id_imds_in_floor_blocked(self):
        """An attacker-controlled hostname resolving to a scope-ID-bearing,
        IPv4-mapped IMDS address must be caught after the scope is stripped,
        not skipped as unparseable."""
        with _resolves_to("::ffff:169.254.169.254%eth0"):
            assert is_always_blocked_url("http://attacker-controlled.example.com/") is True

    # -- Things the floor must NOT block ----------------------------------------

    def test_public_url_not_blocked(self):
        assert is_always_blocked_url("https://example.com/path") is False

    def test_ordinary_private_urls_not_in_floor(self):
        """Floor is narrower than is_safe_url — ordinary private URLs pass.

        CGNAT is blocked by is_safe_url but must not be claimed by the floor.
        """
        assert is_always_blocked_url("http://127.0.0.1:8080/") is False
        assert is_always_blocked_url("http://100.64.0.1/") is False

    def test_dns_failure_not_in_floor(self):
        """DNS failure on a non-sentinel hostname = not always-blocked.

        Caller's ordinary fail-closed path (is_safe_url) handles that case.
        """
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("fail")):
            assert is_always_blocked_url("http://nonexistent.example.com/") is False

    def test_malformed_url_not_in_floor(self):
        """Parse errors don't claim always-blocked status."""
        assert is_always_blocked_url("not a url at all") is False

    def test_floor_ignores_allow_private_urls_toggle(self, monkeypatch):
        """security.allow_private_urls can NOT unblock cloud metadata."""
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        assert is_always_blocked_url("http://169.254.169.254/") is True


class TestIPv4MappedIPv6SSRF:
    """Regression tests for SSRF bypass via IPv4-mapped IPv6 addresses.

    DNS resolvers may return ``::ffff:x.x.x.x`` for IPv4-only hosts, which
    Python's ipaddress module treats as distinct from the plain IPv4 address.
    """

    @pytest.mark.parametrize("ip, url", [
        ("::ffff:169.254.169.254", "http://aws-metadata.internal/"),
        # in the CGNAT range, so a different block branch than link-local
        ("::ffff:100.100.100.200", "http://aliyun-metadata.internal/"),
    ])
    def test_ipv4_mapped_metadata_blocked(self, ip, url):
        with _resolves_to(ip):
            assert is_safe_url(url) is False


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


    def test_falls_back_to_next_request_when_no_location(self):
        resp = _FakeResponse(
            is_redirect=True,
            next_request=_FakeNextRequest("http://10.0.0.1/meta"),
        )
        assert redirect_target_from_response(resp) == "http://10.0.0.1/meta"
