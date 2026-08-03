"""Managed-install guard in :func:`tools.lazy_deps.ensure` (#48628).

A package-manager install (NixOS, and anything else shipping Hermes from a
read-only store) cannot receive lazy pip installs: the venv's site-packages
lives in the store, so the uv -> pip -> ensurepip ladder burns ~15s
bootstrapping ensurepip only to fail. ``ensure()`` must fail fast instead.
"""

import pytest

from tools import lazy_deps
from tools.lazy_deps import FeatureUnavailable


FEATURE = "provider.anthropic"


@pytest.fixture(autouse=True)
def _missing_and_installable(monkeypatch):
    """Reach the guard: deps missing, installs allowed, no durable target.

    ``_allow_lazy_installs`` is patched explicitly so the suite does not
    depend on the host's ~/.hermes/config.yaml (a local
    ``allow_lazy_installs: false`` otherwise short-circuits with a different
    rejection reason).
    """
    monkeypatch.setattr(lazy_deps, "feature_missing", lambda _f: ("some-pkg==1.0",))
    monkeypatch.setattr(lazy_deps, "_allow_lazy_installs", lambda: True)
    monkeypatch.setattr(lazy_deps, "_lazy_install_target", lambda: None)


def _no_installer(monkeypatch):
    """Fail loudly if the guard lets execution reach the install ladder."""
    def _boom(*_a, **_kw):
        raise AssertionError("guard let execution reach the install ladder")

    monkeypatch.setattr(lazy_deps.subprocess, "run", _boom)


def test_nixos_install_fails_fast_without_touching_the_installer(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    _no_installer(monkeypatch)

    with pytest.raises(FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert "nixos" in excinfo.value.reason
    # refresh_active_features classifies by this prefix — anything else is
    # reported to the user as a hard failure instead of a skip.
    assert excinfo.value.reason.startswith("unsupported ")


def test_reason_is_classified_as_skipped_not_failed(monkeypatch):
    """The wording contract with refresh_active_features, pinned directly."""
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")

    with pytest.raises(FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert excinfo.value.reason.startswith("unsupported "), (
        "refresh_active_features would report this as failed: rather than skipped:"
    )


def test_unmanaged_install_is_not_blocked_by_the_guard(monkeypatch):
    """On a normal pip install the guard must be transparent."""
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: None)

    with pytest.raises(FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    # Whatever stops the install here, it must NOT be the managed guard.
    assert "managed installs" not in excinfo.value.reason


def test_durable_install_target_overrides_the_guard(monkeypatch, tmp_path):
    """The container deployment sets HERMES_MANAGED *and* a writable target.

    Dockerfile sets HERMES_LAZY_INSTALL_TARGET and the NixOS container module
    passes HERMES_MANAGED=true; blocking there would break that deployment.
    """
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    monkeypatch.setattr(lazy_deps, "_lazy_install_target", lambda: tmp_path)

    with pytest.raises(FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert "nixos" not in excinfo.value.reason.lower(), (
        "durable-target installs must not be blocked by the managed guard"
    )


def test_platform_unsupported_takes_precedence(monkeypatch):
    """A platform-specific reason is more actionable than 'managed install'.

    Also required for consistency: refresh_active_features pre-checks
    _unsupported_feature_reason before calling ensure().
    """
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    monkeypatch.setattr(
        lazy_deps, "_unsupported_feature_reason", lambda _f: "unsupported on win32"
    )

    with pytest.raises(FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert excinfo.value.reason == "unsupported on win32"


def test_unreadable_config_fails_open(monkeypatch):
    """A broken config must not block installs on a normal pip install."""
    def _raise():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.get_managed_system", _raise)

    with pytest.raises(FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert "managed" not in excinfo.value.reason.lower()
