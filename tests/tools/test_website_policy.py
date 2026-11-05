"""Website blocklist rule normalization and matching (Regression for #127699).

A URL-form rule used to be reduced from ``netloc`` (host + optional userinfo +
optional port), so ``http://example.com:8080/`` became the pattern
``example.com:8080`` and could never match a real host — a silently inert rule.
Rules reduce from ``hostname``; a schemeless rule that still carries a port or
userinfo is dropped with a warning instead of becoming an unmatchable pattern.
"""
from __future__ import annotations

import pytest

from tools.website_policy import _extract_host_from_urlish, _match_host_against_rule, _normalize_rule


@pytest.mark.parametrize("rule, expected", [
    ("example.com", "example.com"),
    ("*.example.com", "*.example.com"),
    ("example.com/path", "example.com"),
    ("https://WWW.Example.com./", "example.com"),
    ("http://example.com:8080/", "example.com"),
    ("https://user:pw@blocked.example.com/", "blocked.example.com"),
    # IPv6 literals are all-colons, no port — the drop rule must not swallow them.
    ("2001:db8::1", "2001:db8::1"),
    ("::1", "::1"),
    ("http://[2001:db8::1]/", "2001:db8::1"),
])
def test_url_form_rule_reduces_to_its_hostname(rule, expected):
    assert _normalize_rule(rule) == expected


@pytest.mark.parametrize("rule", ["example.com:8080", "user:pw@example.com"])
def test_schemeless_rule_with_port_or_userinfo_is_dropped_with_a_warning(rule, caplog):
    with caplog.at_level("WARNING", logger="tools.website_policy"):
        assert _normalize_rule(rule) is None
    assert any("cannot be matched" in record.message for record in caplog.records)


def test_a_url_form_rule_with_a_port_now_matches_its_host():
    # The reduced pattern must actually be matchable by the request-side host,
    # which is normalized to host-only (no port).
    pattern = _normalize_rule("http://example.com:8080/")
    assert pattern is not None
    assert _match_host_against_rule("example.com", pattern)


def test_a_bare_ipv6_rule_still_blocks_its_host():
    # Regression: dropping every rule containing ":" silently removed all IPv6
    # entries and re-opened the fail-open hole this guard was meant to close.
    pattern = _normalize_rule("2001:db8::1")
    assert pattern == "2001:db8::1"
    assert _match_host_against_rule(_extract_host_from_urlish("http://[2001:db8::1]/page"), pattern)
