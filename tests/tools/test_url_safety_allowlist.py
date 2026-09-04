"""Tests for security.allowed_private_networks CIDR allowlist in url_safety.

Some environments run a local DNS resolver that maps public domains into the
RFC 2544 benchmark range 198.18.0.0/15. The allowlist lets a user exempt only
that range while keeping SSRF blocking in force everywhere else, and crucially
without ever exposing the always-blocked cloud-metadata floor.
"""

import socket

import pytest

import tools.url_safety as url_safety


# IP each test hostname resolves to (no real DNS).
_FAKE_DNS = {
    "example.com": "198.18.15.159",         # mapped into benchmark range
    "router.lan": "192.168.1.1",            # real LAN
    "public.example": "93.184.216.34",      # ordinary public IP
    "metadata.evil": "169.254.169.254",     # cloud metadata (always-blocked floor)
    "mapped.test": "::ffff:198.18.15.159",  # IPv4-mapped IPv6 of benchmark IP
}


@pytest.fixture
def patched(monkeypatch):
    """Patch DNS + config and clear the module's cache around each test."""
    def fake_getaddrinfo(host, *args, **kwargs):
        ip = _FAKE_DNS[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)

    def set_config(cfg):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: cfg, raising=False
        )
        url_safety._reset_allow_private_cache()

    set_config({})
    yield set_config
    url_safety._reset_allow_private_cache()


def test_benchmark_ip_blocked_without_allowlist(patched):
    assert url_safety.is_safe_url("https://example.com/some/path") is False


def test_public_always_allowed(patched):
    assert url_safety.is_safe_url("https://public.example") is True


def test_benchmark_ip_allowed_with_allowlist(patched):
    patched({"security": {"allowed_private_networks": ["198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://example.com/some/path") is True


def test_real_lan_still_blocked_with_allowlist(patched):
    patched({"security": {"allowed_private_networks": ["198.18.0.0/15"]}})
    assert url_safety.is_safe_url("http://router.lan") is False


def test_metadata_floor_cannot_be_allowlisted(patched):
    # Even explicitly listing the link-local range must not expose metadata.
    patched({"security": {"allowed_private_networks": ["169.254.0.0/16", "198.18.0.0/15"]}})
    assert url_safety.is_safe_url("http://metadata.evil") is False


def test_single_string_cidr_accepted(patched):
    patched({"security": {"allowed_private_networks": "198.18.0.0/15"}})
    assert url_safety.is_safe_url("https://example.com/q") is True


def test_invalid_entry_skipped_valid_applies(patched):
    patched({"security": {"allowed_private_networks": ["not-a-cidr", "198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://example.com/q") is True
    assert url_safety.is_safe_url("http://router.lan") is False


def test_ipv4_mapped_ipv6_covered_by_v4_cidr(patched):
    patched({"security": {"allowed_private_networks": ["198.18.0.0/15"]}})
    assert url_safety.is_safe_url("https://mapped.test") is True


def test_non_matching_allowlist_leaves_benchmark_ip_blocked(patched):
    patched({"security": {"allowed_private_networks": ["10.0.0.0/8"]}})
    assert url_safety.is_safe_url("https://example.com/q") is False
