"""NO_PROXY entries httpx cannot compile are rewritten to the bare-IPv6 form (#118159).

Clash Verge / mihomo export loopback bypasses in bracketed form (``[::1]``); httpx 0.28.1's
``get_environment_proxies`` sends any bracketed entry down its wildcard branch, producing
``all://*[::1]``, and ``URLPattern`` dies at ``Client.__init__`` with ``InvalidURL: Invalid port``.
Every bare ``httpx.Client()`` in the process then raises — codex login, doctor probes — while the
chat transport (explicit mounts) keeps working. ``utils.normalize_proxy_env_vars`` (already the
in-place env sanitizer called when chat transports are built) rewrites those entries to the bare
``::1`` literal that httpx compiles into a working ``all://[::1]`` bypass and that urllib, requests
and the repo's own ``agent.proxy_bypass`` matcher all understand.
"""

from __future__ import annotations

import pytest

import httpx

from utils import normalize_proxy_env_vars, sanitize_no_proxy_entries


@pytest.fixture
def proxy_env(monkeypatch):
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)


# The exact environment Clash Verge / mihomo writes (issue #118159).
_CLASH_NO_PROXY = "127.0.0.1,localhost,::1,[::1]"


class TestSanitizeNoProxyEntries:
    def test_bracketed_ipv6_becomes_bare_literal(self):
        assert sanitize_no_proxy_entries(_CLASH_NO_PROXY) == "127.0.0.1,localhost,::1"

    @pytest.mark.parametrize(
        "entry,expected",
        [
            ("[::1]", "::1"),
            ("[::1]:8080", "::1"),  # httpx cannot express a port-scoped v6 bypass; the literal still bypasses every port
            ("[2001:db8::1]", "2001:db8::1"),
            ("[::ffff:127.0.0.1]", "::ffff:127.0.0.1"),
            ("[::1]/128", "::1"),
            ("::1/128", "::1"),  # httpx cannot compile v6 CIDR bypasses at all; degrade to the bare literal
            ("2001:db8::/32", "2001:db8::"),
        ],
    )
    def test_bracketed_and_cidr_ipv6_entries_are_rewritten(self, entry, expected):
        assert sanitize_no_proxy_entries(f"example.com,{entry}") == f"example.com,{expected}"

    @pytest.mark.parametrize(
        "value",
        [
            "",  # nothing set: nothing invented
            "*",
            "127.0.0.1,localhost,::1",
            "10.0.0.0/8,*.internal",  # IPv4 CIDR / wildcard domains: httpx compiles these fine
            "[127.0.0.1]",  # bracketed IPv4: not the broken form, untouched
            "[not-an-ip]",  # bracketed junk: httpx's wildcard branch compiles it, untouched
            "example.com,  spaced.internal ,10.0.0.5",
            "::1:8080",  # a valid v6 address in its own right (hex), httpx compiles it
        ],
    )
    def test_unaffected_values_pass_through_byte_stable(self, value):
        assert sanitize_no_proxy_entries(value) == value


class TestNormalizeProxyEnvVars:
    def test_rewrites_both_casings_in_place(self, proxy_env, monkeypatch):
        monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
        monkeypatch.setenv("no_proxy", _CLASH_NO_PROXY)
        normalize_proxy_env_vars()
        assert os_get("NO_PROXY") == "127.0.0.1,localhost,::1"
        assert os_get("no_proxy") == "127.0.0.1,localhost,::1"

    def test_clean_environment_is_left_untouched(self, proxy_env, monkeypatch):
        monkeypatch.setenv("NO_PROXY", "example.com,10.0.0.0/8")
        normalize_proxy_env_vars()
        assert os_get("NO_PROXY") == "example.com,10.0.0.0/8"

    def test_socks_scheme_normalization_keeps_working(self, proxy_env, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "socks://127.0.0.1:7890")
        monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
        normalize_proxy_env_vars()
        assert os_get("HTTPS_PROXY") == "socks5://127.0.0.1:7890"  # pre-existing behavior intact
        assert os_get("NO_PROXY") == "127.0.0.1,localhost,::1"


class TestSanitizedEnvYieldsAWorkingHttpxClient:
    def test_client_constructs_and_loopback_v6_bypasses(self, proxy_env, monkeypatch):
        monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
        monkeypatch.setenv("no_proxy", _CLASH_NO_PROXY)
        normalize_proxy_env_vars()
        # Red on base: the unsanitized Clash env raises InvalidURL at construction.
        client = httpx.Client()  # noqa: SIM115 - closed below
        try:
            assert any(
                pattern.matches(httpx.URL("http://[::1]:8080/")) and transport is None
                for pattern, transport in client._mounts.items()
            )
        finally:
            client.close()

    def test_bypass_still_matches_after_normalization(self, proxy_env, monkeypatch):
        from agent.proxy_bypass import should_bypass_proxy

        monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
        normalize_proxy_env_vars()
        assert should_bypass_proxy("http://[::1]:8080/")
        assert should_bypass_proxy("https://[::1]/")
        assert not should_bypass_proxy("https://api.example.com/")


def os_get(key: str) -> str:
    import os

    return os.environ[key]
