"""Tests for lazy_deps.ensure() managed-install guard (issue #48628)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.lazy_deps import FeatureUnavailable, ensure


class TestEnsureManagedInstallGuard:
    """Verify that ensure() raises early on managed/package-manager installs."""

    def test_raises_when_managed(self, monkeypatch):
        """Managed install (NixOS/Homebrew) → FeatureUnavailable before any pip attempt."""
        # Pick a feature whose deps are NOT installed so ensure() doesn't
        # short-circuit at the ``if not missing: return`` check.
        # We mock feature_missing to return a non-empty tuple.
        monkeypatch.setattr(
            "tools.lazy_deps.feature_missing",
            lambda _feat: ("some-pkg==1.0",),
        )
        with (
            patch("hermes_cli.config.is_managed", return_value=True),
            patch("hermes_cli.config.get_managed_system", return_value="NixOS"),
        ):
            with pytest.raises(FeatureUnavailable, match="cannot install on NixOS"):
                ensure("provider.anthropic")

    def test_proceeds_when_not_managed(self, monkeypatch):
        """Non-managed install → does NOT raise at the managed guard."""
        # Mock feature_missing to return empty (all deps satisfied) so
        # ensure() returns cleanly without reaching the install path.
        monkeypatch.setattr(
            "tools.lazy_deps.feature_missing",
            lambda _feat: (),
        )
        with (
            patch("hermes_cli.config.is_managed", return_value=False),
        ):
            # Should return without error (deps already satisfied).
            ensure("provider.anthropic")

    def test_proceeds_when_config_unreadable(self, monkeypatch):
        """Config import failure → fail open, proceed normally."""
        monkeypatch.setattr(
            "tools.lazy_deps.feature_missing",
            lambda _feat: (),
        )
        # Simulate config import failure by making is_managed raise.
        with patch(
            "hermes_cli.config.is_managed",
            side_effect=ImportError("no module"),
        ):
            ensure("provider.anthropic")

    def test_homebrew_managed_message(self, monkeypatch):
        """Homebrew managed install → message mentions Homebrew."""
        monkeypatch.setattr(
            "tools.lazy_deps.feature_missing",
            lambda _feat: ("some-pkg==1.0",),
        )
        with (
            patch("hermes_cli.config.is_managed", return_value=True),
            patch("hermes_cli.config.get_managed_system", return_value="Homebrew"),
        ):
            with pytest.raises(FeatureUnavailable, match="cannot install on Homebrew"):
                ensure("provider.anthropic")

    def test_unknown_managed_system_fallback(self, monkeypatch):
        """Unknown managed system → generic 'managed' in message."""
        monkeypatch.setattr(
            "tools.lazy_deps.feature_missing",
            lambda _feat: ("some-pkg==1.0",),
        )
        with (
            patch("hermes_cli.config.is_managed", return_value=True),
            patch("hermes_cli.config.get_managed_system", return_value=None),
        ):
            with pytest.raises(FeatureUnavailable, match="cannot install on managed"):
                ensure("provider.anthropic")
