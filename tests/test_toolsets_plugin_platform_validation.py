"""Pins for validate_toolset()/resolve_toolset() agreement on plugin platforms.

resolve_toolset() synthesizes a ``hermes-<platform>`` bundle for any platform
in the platform registry, including plugin platforms absent from TOOLSETS.
validate_toolset() did not, so config validation reported a resolvable bundle
as unknown -- and suggested the very name it had just rejected.
"""

import pytest

from toolsets import TOOLSETS, resolve_toolset, validate_toolset


class _FakeRegistry:
    def __init__(self, registered):
        self._registered = set(registered)

    def is_registered(self, name):
        return name in self._registered


@pytest.fixture
def plugin_platform(monkeypatch):
    """Register 'wibble' as a plugin platform, absent from TOOLSETS."""
    import gateway.platform_registry as pr_mod

    fake = _FakeRegistry({"wibble"})
    monkeypatch.setattr(pr_mod, "platform_registry", fake)
    assert "hermes-wibble" not in TOOLSETS
    return fake


def test_plugin_platform_bundle_validates(plugin_platform):
    assert validate_toolset("hermes-wibble") is True


def test_validate_agrees_with_resolve_for_plugin_platform(plugin_platform):
    # The invariant: anything that resolves to real tools must validate.
    assert resolve_toolset("hermes-wibble")
    assert validate_toolset("hermes-wibble") is True


def test_unregistered_platform_bundle_still_rejected(plugin_platform):
    # The widening is registry-gated, so a typo is still caught.
    assert resolve_toolset("hermes-nosuchplatform") == []
    assert validate_toolset("hermes-nosuchplatform") is False


def test_non_bundle_names_unaffected(plugin_platform):
    assert validate_toolset("wibble") is False
    assert validate_toolset("nonexistent") is False
    assert validate_toolset("web") is True


def test_registry_failure_does_not_raise(monkeypatch):
    # platform_registry is imported lazily inside a try/except; an import or
    # lookup failure must degrade to False, never propagate into config load.
    import gateway.platform_registry as pr_mod

    class _Boom:
        def is_registered(self, name):
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(pr_mod, "platform_registry", _Boom())
    assert validate_toolset("hermes-wibble") is False
