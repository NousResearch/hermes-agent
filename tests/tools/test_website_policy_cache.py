"""Enabling the website blocklist must take effect without a process restart.

``check_website_access`` short-circuits on a cached *disabled* policy and returns
"allowed" without consulting the cache TTL, so a long-lived gateway/cron process
that once resolved the (default) disabled state kept allowing hosts after the
operator enabled ``security.website_blocklist`` — the control was silently inert
until restart. The fast path must honor the same TTL the loader does.
"""
from __future__ import annotations

import time

import pytest

from tools import website_policy as wp

CONFIG_DISABLED = "security:\n  website_blocklist:\n    enabled: false\n"
CONFIG_ENABLED = (
    "security:\n  website_blocklist:\n    enabled: true\n    domains:\n      - example.com\n"
)


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(wp, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(wp, "_cached_policy", None)
    monkeypatch.setattr(wp, "_cached_policy_path", None)
    monkeypatch.setattr(wp, "_cached_policy_time", 0.0)


def test_enabling_the_blocklist_takes_effect_after_the_cache_ttl(isolated_cache, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG_DISABLED, encoding="utf-8")
    # First check caches the default disabled policy.
    assert wp.check_website_access("http://example.com/") is None

    config.write_text(CONFIG_ENABLED, encoding="utf-8")
    # Simulate the TTL elapsing rather than sleeping _CACHE_TTL_SECONDS.
    wp._cached_policy_time = time.monotonic() - wp._CACHE_TTL_SECONDS - 1

    blocked = wp.check_website_access("http://example.com/")
    assert blocked is not None
    assert blocked["rule"] == "example.com"


def test_a_fresh_disabled_cache_still_allows_without_reloading(isolated_cache, tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(CONFIG_DISABLED, encoding="utf-8")
    assert wp.check_website_access("http://example.com/") is None
    # Cache is fresh: the fast path returns without touching the config again.
    calls = []
    original = wp.load_website_blocklist
    monkeypatch.setattr(wp, "load_website_blocklist", lambda *a, **k: calls.append(1) or original(*a, **k))
    assert wp.check_website_access("http://example.com/") is None
    assert calls == []
