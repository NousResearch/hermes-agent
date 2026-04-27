"""Regression tests for #16027: image_generate `check_fn` must not be
gated by a plugin-discovery race at session init.

The previous behavior gated `image_generate` on a successful probe of
the plugin registry. When session init fired `check_fn` before plugin
discovery had registered providers, the tool was permanently excluded
from the session — even though `_handle_image_generate` works correctly
when invoked directly because it does its own retry with
`_ensure_plugins_discovered(force=True)`.

The fix adds a soft-accept gate: an explicitly configured
`image_gen.provider` in config.yaml means "the user wants this tool" —
the actual availability check is deferred to call time.
"""
from __future__ import annotations

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider
from tools import image_generation_tool


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_fal_key(monkeypatch):
    # Strip every variant of the FAL key plus any managed-gateway env so
    # the in-tree FAL branch never short-circuits the test.
    for var in (
        "FAL_KEY", "FAL_AI_API_KEY", "FAL_API_KEY",
        "FAL_GATEWAY_URL", "FAL_GATEWAY_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def _force_no_fal(monkeypatch):
    """Belt-and-suspenders: also stub check_fal_api_key to False."""
    monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)


class _UnavailableProvider(ImageGenProvider):
    """A provider that's registered but not available — simulates a plugin
    that has loaded its module but its underlying SDK/credentials aren't
    ready (e.g. OAuth still pending)."""

    @property
    def name(self) -> str:
        return "codex"

    def is_available(self) -> bool:
        return False

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"success": False, "error": "unavailable"}


def test_check_fn_returns_true_when_provider_configured_but_registry_empty(monkeypatch):
    """The original repro from #16027: image_gen.provider is set, but
    plugin discovery hasn't populated the registry yet at session init.
    Without the fix, the tool is excluded for the entire session."""
    _force_no_fal(monkeypatch)
    monkeypatch.setattr(
        image_generation_tool, "_read_configured_image_provider",
        lambda: "openai-codex",
    )
    # Registry is intentionally empty — simulates pre-discovery state.
    assert image_generation_tool.check_image_generation_requirements() is True


def test_check_fn_returns_true_when_provider_configured_but_unavailable(monkeypatch):
    """If the provider is registered but its is_available() is still
    flaky at session init, the soft-accept gate must still surface the
    tool. Provider availability is re-checked per call."""
    _force_no_fal(monkeypatch)
    image_gen_registry.register_provider(_UnavailableProvider())
    monkeypatch.setattr(
        image_generation_tool, "_read_configured_image_provider",
        lambda: "codex",
    )
    assert image_generation_tool.check_image_generation_requirements() is True


def test_check_fn_returns_false_with_no_fal_no_config_no_providers(monkeypatch):
    """Regression guard: with nothing configured and no providers
    registered, the tool stays gated. The fix must not flip the default
    on for users who have never opted in."""
    _force_no_fal(monkeypatch)
    monkeypatch.setattr(
        image_generation_tool, "_read_configured_image_provider",
        lambda: None,
    )
    assert image_generation_tool.check_image_generation_requirements() is False


def test_check_fn_returns_false_when_only_fal_configured_but_unavailable(monkeypatch):
    """If image_gen.provider is "fal" but FAL_KEY is unset, we should
    NOT soft-accept — fal is the in-tree backend handled by the FAL
    branch above, not by the dispatcher's plugin lookup."""
    _force_no_fal(monkeypatch)
    monkeypatch.setattr(
        image_generation_tool, "_read_configured_image_provider",
        lambda: "fal",
    )
    assert image_generation_tool.check_image_generation_requirements() is False
