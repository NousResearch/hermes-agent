"""Tests for credential-pool cross-profile env leakage (#65940).

In multiplex mode, ``_seed_from_env`` must NOT fall through to
``os.environ.get()`` when the active profile's scope has no matching
key — that would leak another profile's credential into the pool.

Fixtures and helpers are duplicated inline (no dependency on the base
``isolated_hermes_home`` fixture from conftest) to keep this test
self-contained and easy to reason about.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from agent.credential_pool import PooledCredential, _seed_from_env
from agent.secret_scope import (
    get_secret,
    set_multiplex_active,
    set_secret_scope,
)


def _empty_pool() -> list[PooledCredential]:
    return []


class TestCredentialPoolCrossProfile:
    """Regression tests for #65940."""

    def test_single_profile_fallthrough_to_os_environ(self, monkeypatch):
        """When multiplex is OFF (default), an env var set in os.environ IS
        picked up — single-profile behaviour is unchanged."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-single-profile-key")
        entries = _empty_pool()
        changed, _sources = _seed_from_env("openrouter", entries)
        assert changed is True, "should seed from os.environ when multiplex off"
        assert any(
            e.access_token == "sk-single-profile-key" for e in entries
        ), "single-profile fallthrough broken"

    def test_multiplex_active_no_scope_raises(self):
        """When multiplex is ACTIVE but no secret scope is installed,
        ``get_secret`` raises ``UnscopedSecretError`` — fail-closed."""
        set_multiplex_active(True)
        try:
            with pytest.raises(RuntimeError, match="no profile secret scope"):
                get_secret("OPENROUTER_API_KEY")
        finally:
            set_multiplex_active(False)

    def test_cross_profile_leak_not_reproduced(self, monkeypatch):
        """When multiplex is ACTIVE and profile B's scope (no OPENROUTER key)
        is installed, _seed_from_env must NOT see profile A's key in
        os.environ."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leaked-from-A")
        set_multiplex_active(True)
        token = set_secret_scope({"ANTHROPIC_API_KEY": "profile-B-key"})
        try:
            entries = _empty_pool()
            changed, _sources = _seed_from_env("openrouter", entries)
            # Expected: profile B has no OPENROUTER key, so nothing seeds
            assert changed is False, "should NOT seed cross-profile leaked key"
            assert not any(
                e.access_token == "sk-leaked-from-A" for e in entries
            ), "profile A's key leaked into profile B's pool"
        finally:
            set_secret_scope(None)
            set_multiplex_active(False)

    def test_dotenv_still_wins_in_single_profile(self, monkeypatch, tmp_path):
        """When multiplex is off, ~/.hermes/.env takes priority over
        os.environ — the old contract is preserved."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-os-environ")
        # Point at a tmp .env
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        dotenv = hermes_home / ".env"
        dotenv.write_text("OPENROUTER_API_KEY=sk-from-dotenv\n")
        monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: dotenv)
        # Force reload
        from hermes_cli.config import _env_cache
        _env_cache = None  # type: ignore[attr-defined]

        entries = _empty_pool()
        changed, _sources = _seed_from_env("openrouter", entries)
        assert changed is True
        # The .env value should win
        seeded = [e for e in entries if e.access_token == "sk-from-dotenv"]
        assert len(seeded) == 1, ".env value should be seeded, not os.environ"

    def test_multiplex_scope_takes_priority(self, monkeypatch):
        """When multiplex is ACTIVE and scope IS installed, the scope's
        value seeds the pool — not os.environ."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leaked")
        set_multiplex_active(True)
        token = set_secret_scope({"OPENROUTER_API_KEY": "sk-from-scope"})
        try:
            entries = _empty_pool()
            changed, _sources = _seed_from_env("openrouter", entries)
            assert changed is True
            # Must seed from scope, not os.environ
            assert any(
                e.access_token == "sk-from-scope" for e in entries
            ), "scope value should seed the pool"
            assert not any(
                e.access_token == "sk-leaked" for e in entries
            ), "os.environ leaked despite scope being installed"
        finally:
            set_secret_scope(None)
            set_multiplex_active(False)
