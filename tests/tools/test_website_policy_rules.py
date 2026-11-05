"""Tests for website_blocklist rule normalization.

A rule reduced from a URL kept its authority's port/userinfo (``netloc``), while
the matching host side is always a bare host — such rules could never match and
failed silently (#127699).
"""

import logging

import pytest

from tools.website_policy import _normalize_rule, check_website_access


# -------------------------------------------------------------------------
# Rule reduction
# -------------------------------------------------------------------------

@pytest.mark.parametrize("rule,expected", [
    # URL-form rules: the port and userinfo must not leak into the pattern.
    ("http://example.com:8080/", "example.com"),
    ("https://u:p@blocked.example.com/", "blocked.example.com"),
    # Same defect without a scheme — re-parsed as an authority.
    ("example.com:8080/path", "example.com"),
    ("user:pw@host.example.com/x", "host.example.com"),
    # IPv6: bracketless on both sides, so the literal stays matchable.
    ("http://[::1]/", "::1"),
    ("[2001:db8::1]:8080", "2001:db8::1"),
    ("::1", "::1"),
    # Pre-existing forms keep their behavior.
    ("example.com", "example.com"),
    ("*.example.com", "*.example.com"),
    ("https://WWW.Example.com./", "example.com"),
    ("  ", None),
    ("# comment", None),
    (None, None),
])
def test_normalize_rule_reduces_to_a_bare_host(rule, expected):
    assert _normalize_rule(rule) == expected


def test_normalize_rule_warns_when_userinfo_survives(caplog):
    # ``http://user@/`` has no hostname, so the netloc fallback keeps the userinfo;
    # no extracted host can ever contain ``@`` — warn and drop instead of sitting inert.
    with caplog.at_level(logging.WARNING, logger="tools.website_policy"):
        assert _normalize_rule("http://user@/") is None
    assert "userinfo" in caplog.text


# -------------------------------------------------------------------------
# End to end through the config file
# -------------------------------------------------------------------------

def _write_config(tmp_path, domains):
    config = tmp_path / "config.yaml"
    lines = ["security:", "  website_blocklist:", "    enabled: true", "    domains:"]
    lines += [f'      - "{domain}"' for domain in domains]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


def test_url_rule_with_port_now_blocks_the_host(tmp_path):
    # The reproduction config from #127699: a URL-form rule with a port.
    config = _write_config(tmp_path, ["http://example.com:8080/"])
    blocked = check_website_access("http://example.com:8080/page", config_path=config)
    assert blocked is not None
    assert blocked["host"] == "example.com"
    assert blocked["rule"] == "example.com"


def test_url_rule_with_userinfo_now_blocks_the_host(tmp_path):
    config = _write_config(tmp_path, ["https://u:p@blocked.example.com/"])
    blocked = check_website_access("https://blocked.example.com/secret", config_path=config)
    assert blocked is not None
    assert blocked["rule"] == "blocked.example.com"


def test_unrelated_host_still_allowed(tmp_path):
    config = _write_config(tmp_path, ["http://example.com:8080/"])
    assert check_website_access("https://other.example.org/page", config_path=config) is None
