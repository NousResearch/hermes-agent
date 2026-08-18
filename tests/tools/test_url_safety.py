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
# ============================================================================
# Region D — shared resolve-and-validate core (PR #84999 class closure)
# ============================================================================


class TestClassifyIp:
    """_classify_ip: single classification core with pinned reasons."""

    def test_reasons_grid(self):
        from tools.url_safety import _classify_ip

        cases = [
            ("1.2.3.4", False, "ok"),
            ("169.254.169.254", True, "metadata-ip"),
            ("169.254.9.9", True, "link-local"),
            ("127.0.0.1", True, "loopback"),
            ("10.0.0.1", True, "private-ip"),
            ("172.16.5.5", True, "private-ip"),
            ("192.168.1.1", True, "private-ip"),
            ("198.18.0.1", True, "private-ip"),
            ("100.64.0.1", True, "cgnat"),
            ("100.127.255.255", True, "cgnat"),
            ("fe80::1", True, "link-local"),
            ("fc00::1", True, "private-ip"),
            ("fd00::1", True, "private-ip"),
            ("::1", True, "loopback"),
            ("::", True, "unspecified"),
            ("224.0.0.1", True, "multicast"),
            ("2001:db8::1", True, "private-ip"),  # docs range: is_private on >=3.11
            ("93.184.216.34", False, "ok"),
            ("2001:4860:4860::8888", False, "ok"),
        ]
        import ipaddress as _ip

        for ip_str, blocked, reason in cases:
            ip = _ip.ip_address(ip_str)
            got = _classify_ip(ip)
            assert got == (blocked, reason), (ip_str, got)

    def test_mapped_prefix_tagged(self):
        from tools.url_safety import _classify_ip
        import ipaddress as _ip

        assert _classify_ip(_ip.ip_address("::ffff:100.64.0.1")) == (True, "mapped:cgnat")
        assert _classify_ip(_ip.ip_address("::ffff:10.0.0.1")) == (True, "mapped:private-ip")
        assert _classify_ip(_ip.ip_address("::ffff:169.254.169.254")) == (True, "mapped:metadata-ip")
        # ::/96 IPv4-compatible — floor class (G7 fix)
        assert _classify_ip(_ip.ip_address("::a9fe:a9fe")) == (True, "ipv4-compatible")

    def test_is_blocked_ip_backcompat(self):
        """_is_blocked_ip stays a thin bool wrapper (back-compat)."""
        from tools.url_safety import _is_blocked_ip
        import ipaddress as _ip

        assert _is_blocked_ip(_ip.ip_address("10.0.0.1")) is True
        assert _is_blocked_ip(_ip.ip_address("93.184.216.34")) is False


class TestResolveAndCheckUrl:
    """resolve_and_check_url — resolve + validate + pin (fail-closed)."""

    def test_public_host_ok_and_pinned(self):
        from tools.url_safety import _MAX_SSRF_CONNECT_IPS, resolve_and_check_url

        with _resolves_to("93.184.216.34", "93.184.216.35", "93.184.216.34"):
            v = resolve_and_check_url("https://example.com/x")
        assert v.ok is True
        assert v.reason == "ok"
        assert v.resolved_ips == ("93.184.216.34", "93.184.216.35")  # deduped
        assert len(v.resolved_ips) <= _MAX_SSRF_CONNECT_IPS

    def test_cap_returns_only_but_classifies_all(self):
        """All answers classified even beyond the return cap (D7.1)."""
        from tools.url_safety import _MAX_SSRF_CONNECT_IPS, resolve_and_check_url

        public = [f"93.184.216.{i}" for i in range(1, _MAX_SSRF_CONNECT_IPS + 1)]
        with _resolves_to(*(public + ["10.0.0.1"])):
            v = resolve_and_check_url("https://example.com/x")
        assert v.ok is False
        assert v.reason == "blocked:private-ip"

    def test_any_private_answer_blocks(self):
        from tools.url_safety import resolve_and_check_url

        with _resolves_to("93.184.216.34", "10.0.0.1"):
            v = resolve_and_check_url("http://example.com/x")
        assert v.ok is False
        assert v.reason == "blocked:private-ip"

    def test_hostname_floor_no_dns(self):
        from tools.url_safety import resolve_and_check_url

        calls = []

        def _no_dns(*a, **k):
            calls.append(a)
            raise AssertionError("resolver must not be consulted")

        assert resolve_and_check_url(
            "http://metadata.google.internal/", resolve=_no_dns
        ).reason == "blocked:metadata-host"
        assert resolve_and_check_url(
            "http://example.internal/", resolve=_no_dns
        ).reason == "blocked:metadata-host"
        assert not calls

    def test_floor_metadata_ip_and_link_local(self):
        from tools.url_safety import resolve_and_check_url

        with _resolves_to("169.254.169.254"):
            assert resolve_and_check_url("http://host.example/").reason == "blocked:metadata-ip"
        with _resolves_to("169.254.9.9"):
            assert resolve_and_check_url("http://host.example/").reason == "blocked:link-local"

    def test_numeric_coercion_resolver_never_called(self):
        from tools.url_safety import resolve_and_check_url

        def _no_dns(*a, **k):
            raise AssertionError("resolver must not be consulted for numeric hosts")

        for url in ("http://2130706433/", "http://0x7f000001/", "http://0177.0.0.1/",
                    "http://127.1/", "http://0x7f.0.0.1/"):
            v = resolve_and_check_url(url, resolve=_no_dns)
            assert v.ok is False, url
            assert v.reason == "blocked:loopback", (url, v.reason)

    def test_strict_parse_backslash_and_control(self):
        from tools.url_safety import resolve_and_check_url

        def _no_dns(*a, **k):
            raise AssertionError("resolver must not be consulted for parse blocks")

        cases = [
            "http://\\user@evil.com/",
            "http://evil.com\\@127.0.0.1/",
            "http://127.0.0.1%5cevil.com/",
            "http://example.com%0a.evil.com/",
            "http://example.com%09.evil.com/",
        ]
        for url in cases:
            v = resolve_and_check_url(url, resolve=_no_dns)
            assert v.reason == "blocked:parse", (url, v.reason)

    def test_unsupported_scheme_and_missing_host(self):
        from tools.url_safety import resolve_and_check_url

        assert resolve_and_check_url("ftp://example.com/f").reason == "blocked:unsupported-scheme"
        assert resolve_and_check_url("http:///path").reason == "blocked:parse"
        assert resolve_and_check_url("").reason == "blocked:parse"

    def test_ipv6_exotic_classes(self):
        from tools.url_safety import resolve_and_check_url

        cases = [
            ("http://[::a9fe:a9fe]/", "blocked:ipv4-compatible"),
            ("http://[::ffff:169.254.169.254]/", "blocked:metadata-ip"),
            ("http://[fe80::1%25eth0]/", "blocked:link-local"),
            ("http://[fc00::1]/", "blocked:private-ip"),
            ("http://[::1]/", "blocked:loopback"),
        ]
        for url, reason in cases:
            v = resolve_and_check_url(url)
            assert v.reason == reason, (url, v.reason)
        assert resolve_and_check_url("http://[2001:4860:4860::8888]/").ok is True

    def test_dns_failure_no_proxy_delegation_inside_helper(self, monkeypatch):
        from tools.url_safety import resolve_and_check_url

        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:9090")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            v = resolve_and_check_url("http://nonexistent.example.com/")
        assert v.ok is False
        assert v.reason == "error:dns"

    def test_error_internal_on_unparseable_answer(self):
        from tools.url_safety import resolve_and_check_url

        with _resolves_to("not-an-ip"):
            v = resolve_and_check_url("http://host.example/")
        assert v.reason == "error:internal"

    def test_resolve_called_exactly_once(self):
        """Pin contract: one resolution, whole answer set evaluated (T11)."""
        from tools.url_safety import resolve_and_check_url

        calls = []

        def _resolver(*a, **k):
            calls.append(a)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
            ]

        v = resolve_and_check_url("http://rebind.example/", resolve=_resolver)
        assert v.ok is False
        assert len(calls) == 1

    def test_trusted_https_host_gate(self):
        from tools.url_safety import resolve_and_check_url

        with _resolves_to("198.18.0.23"):
            assert resolve_and_check_url("https://multimedia.nt.qq.com.cn/x").ok is True
            v = resolve_and_check_url("http://multimedia.nt.qq.com.cn/x")
            assert v.ok is False  # http does NOT get the exemption (D5)

    def test_toggle_skips_private_but_not_floor(self):
        from tools.url_safety import resolve_and_check_url

        with _resolves_to("192.168.1.1"):
            assert resolve_and_check_url("http://router.local/", allow_private=True).ok is True
        with _resolves_to("169.254.169.254"):
            assert resolve_and_check_url("http://router.local/", allow_private=True).ok is False


class TestAsyncResolveAndCheckUrl:
    @pytest.mark.asyncio
    async def test_matches_sync_verdict(self):
        from tools.url_safety import async_resolve_and_check_url

        with _resolves_to("127.0.0.1"):
            v = await async_resolve_and_check_url("http://localhost:8080/")
        assert v.ok is False
        assert v.reason == "blocked:loopback"


class TestIpIsBlocked:
    def test_observed_ip_classification(self):
        from tools.url_safety import ip_is_blocked

        assert ip_is_blocked("1.2.3.4") == (False, "ok")
        assert ip_is_blocked("169.254.169.254") == (True, "metadata-ip")
        assert ip_is_blocked("::ffff:100.64.0.1") == (True, "mapped:cgnat")
        assert ip_is_blocked("fe80::1%eth0") == (True, "link-local")
        assert ip_is_blocked("not-an-ip") == (True, "blocked:parse")
        assert ip_is_blocked("") == (True, "blocked:parse")


class TestUrlBlockReason:
    def test_strict_predicate(self, monkeypatch):
        from tools.url_safety import url_block_reason

        assert url_block_reason("http://169.254.169.254/latest/meta-data/") == "blocked:metadata-ip"
        assert url_block_reason("http://10.0.0.1/x") == "blocked:private-ip"
        assert url_block_reason("https://example.com/") is None
        # fail-closed on DNS even with a proxy configured
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:9090")
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            assert url_block_reason("http://split-horizon.example/") == "error:dns"


class TestOracleConvergence:
    """_url_is_private and is_safe_url must agree on the real divergences (D3)."""

    @pytest.mark.parametrize("ip", [
        "::ffff:100.64.0.1",       # mapped CGNAT — real divergence
        "::ffff:100.100.100.200",  # mapped Alibaba floor
        "::a9fe:a9fe",             # ::/96 IPv4-compatible (G7)
        "100.64.0.1",              # CGNAT
        "198.18.0.1",              # benchmark/private on >=3.11
        "::ffff:10.0.0.1",         # mapped private (convergence control)
    ])
    def test_routing_and_enforcement_agree(self, ip):
        from tools.browser_tool import _url_is_private

        url = f"http://[{ip}]/" if ":" in ip else f"http://{ip}/"
        assert _url_is_private(url) is True, ip
        assert is_safe_url(url) is False, ip

    def test_classification_grid_consistency(self):
        from tools.url_safety import ip_is_blocked, resolve_and_check_url

        grid = ["10.0.0.1", "172.16.5.5", "192.168.1.1", "100.64.0.1", "169.254.9.9",
                "169.254.169.254", "127.0.0.1", "0.0.0.0", "::1", "fe80::1", "fc00::1",
                "fd00::1", "::", "224.0.0.1", "198.18.0.1", "::ffff:10.0.0.1",
                "::ffff:100.64.0.1", "::a9fe:a9fe", "93.184.216.34", "2001:4860:4860::8888"]
        for ip in grid:
            with _resolves_to(ip):
                safe = is_safe_url("http://host.example/")
            blocked, _ = ip_is_blocked(ip)
            literal_url = f"http://[{ip}]/" if ":" in ip else f"http://{ip}/"
            v = resolve_and_check_url(literal_url, allow_private=False)
            assert (not safe) == blocked == (not v.ok), ip


class TestIsAlwaysBlockedUrlRegionD:
    def test_ipv4_compatible_new_floor(self):
        from tools.url_safety import is_always_blocked_url

        assert is_always_blocked_url("http://[::a9fe:a9fe]/") is True

    def test_dns_failure_still_false(self):
        from tools.url_safety import is_always_blocked_url

        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            assert is_always_blocked_url("http://public.example/") is False

    def test_suffix_floor(self):
        from tools.url_safety import is_always_blocked_url

        assert is_always_blocked_url("http://foo.compute.internal/x") is True
        # .local/.lan stay routing-only (not the security floor)
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nxdomain")):
            assert is_always_blocked_url("http://printer.local/") is False
