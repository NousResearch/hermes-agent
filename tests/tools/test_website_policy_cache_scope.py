"""Regression tests for website-policy cache scope and TTL."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import tools.website_policy as website_policy
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _profile(home: Path):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _write_policy(home: Path, *, enabled: bool, domains: list[str] | None = None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": enabled,
                        "domains": domains or [],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _reset_website_policy_cache():
    website_policy._cached_policy = None
    website_policy._cached_policy_path = None
    website_policy._cached_policy_time = 0.0
    yield
    website_policy._cached_policy = None
    website_policy._cached_policy_path = None
    website_policy._cached_policy_time = 0.0


def test_disabled_fast_path_does_not_cross_profile_boundary(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_policy(first, enabled=False)
    _write_policy(second, enabled=True, domains=["blocked.test"])

    with _profile(first):
        assert website_policy.check_website_access("https://blocked.test") is None

    with _profile(second):
        result = website_policy.check_website_access("https://blocked.test")

    assert result is not None
    assert result["rule"] == "blocked.test"


def test_disabled_fast_path_respects_cache_ttl(tmp_path):
    home = tmp_path / "profile"
    _write_policy(home, enabled=False)

    with _profile(home):
        assert website_policy.check_website_access("https://blocked.test") is None

        _write_policy(home, enabled=True, domains=["blocked.test"])
        website_policy._cached_policy_time -= website_policy._CACHE_TTL_SECONDS + 1

        result = website_policy.check_website_access("https://blocked.test")

    assert result is not None
    assert result["rule"] == "blocked.test"
