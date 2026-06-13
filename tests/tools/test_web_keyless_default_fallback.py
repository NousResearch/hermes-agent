"""Regression: the keyless Parallel web default must survive a failed sweep.

``web_search`` / ``web_extract`` are documented to work out of the box with
zero setup via the bundled keyless Parallel free-MCP backend. That guarantee
only holds if the bundled ``plugins/web/*`` providers are registered in
``agent.web_search_registry``. The dispatch triggers the general plugin sweep
(:func:`hermes_cli.plugins._ensure_plugins_discovered`) to do that — but the
sweep can finish without registering them (its exception swallowed as a
warning, a packaged layout where it ran before the bundled tree was
importable, or a stale empty-discovery cache). When that happened, *both*
tools dead-ended on "No web {search,extract} provider configured" even though
no setup should be needed.

These tests pin the invariant that :func:`tools.web_tools._ensure_web_plugins_loaded`
guarantees the keyless default is registered regardless of the sweep's outcome,
and that the direct-registration fallback honors an explicit ``plugins.disabled``
entry. Real imports from the bundled plugin modules — no provider mocking.
"""
from __future__ import annotations

import pytest

import agent.web_search_registry as reg
import hermes_cli.plugins as plugins
from tools import web_tools


@pytest.fixture(autouse=True)
def _clean_registry():
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


def _boom(*_a, **_k):
    raise RuntimeError("discovery boom")


def test_keyless_default_registered_when_discovery_raises(monkeypatch):
    """A swallowed discovery failure must not strand the keyless default."""
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", _boom)
    assert reg.get_provider("parallel") is None

    web_tools._ensure_web_plugins_loaded()

    parallel = reg.get_provider("parallel")
    assert parallel is not None, "keyless Parallel default not restored"
    # It is the universal keyless default precisely because it does both.
    assert parallel.supports_search()
    assert parallel.supports_extract()


def test_fallback_registers_full_bundled_set(monkeypatch):
    """The fix covers the whole bundled provider class, not just parallel."""
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", _boom)

    web_tools._ensure_web_plugins_loaded()

    names = {p.name for p in reg.list_providers()}
    # Every bundled backend a user might have configured should be reachable
    # again, so an explicit ``web.extract_backend: firecrawl`` etc. resolves.
    for expected in ("parallel", "firecrawl", "tavily", "exa"):
        assert expected in names, f"{expected} missing after fallback"


def test_fallback_honors_explicit_disable(monkeypatch):
    """A backend the user turned off via plugins.disabled stays off."""
    monkeypatch.setattr(plugins, "_get_disabled_plugins", lambda: {"web-parallel"})

    web_tools._register_bundled_web_providers_directly()

    names = {p.name for p in reg.list_providers()}
    assert "parallel" not in names, "explicit disable was ignored"
    # Other bundled backends are unaffected by the parallel disable.
    assert "tavily" in names


def test_fallback_is_noop_when_discovery_already_registered(monkeypatch):
    """Healthy path: don't pay for the direct sweep when parallel is present."""
    # Pretend the general sweep already registered the keyless default.
    import importlib

    class _Ctx:
        def register_web_search_provider(self, provider):
            reg.register_provider(provider)

    importlib.import_module("plugins.web.parallel").register(_Ctx())
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda *a, **k: None)

    calls = {"n": 0}
    real = web_tools._register_bundled_web_providers_directly

    def _spy():
        calls["n"] += 1
        real()

    monkeypatch.setattr(web_tools, "_register_bundled_web_providers_directly", _spy)
    web_tools._ensure_web_plugins_loaded()

    assert calls["n"] == 0, "direct-registration ran on the healthy path"
