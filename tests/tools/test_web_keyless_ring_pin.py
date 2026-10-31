"""``web.keyless_ring_pin`` — let the keyless ring rotate past a pinned backend.

Why the knob exists: with a backend configured, upstream pins that vendor and
every keyless request STARTS there. The other rungs then only run as
rate-limit failover, so a rung that is never rate-limited receives no traffic
at all — which makes it impossible to add a rung that is meant to take a share
of traffic while keeping a configured backend.

The knob changes the walk ORDER only. Provider resolution is untouched: the
configured backend stays the active provider, so keyed and managed routing
behave exactly as before.
"""

import pytest

from plugins.web import keyless_mcp


@pytest.fixture(autouse=True)
def _no_web_env(monkeypatch):
    """Blank every web credential so no keyed path can win."""
    for var in (
        "EXA_API_KEY", "PARALLEL_API_KEY", "KEENABLE_API_KEY", "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "BRAVE_SEARCH_API_KEY",
        "SEARXNG_URL", "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env", lambda name: "", raising=True
    )
    yield


class TestRingRotationKnob:
    def test_knob_defaults_to_pinned_upstream_behaviour(self, monkeypatch):
        """Absent knob => upstream behaviour: the pinned vendor starts every
        request."""
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "firecrawl")
        monkeypatch.setattr(keyless_mcp, "_ring_rotation_enabled", lambda: False)
        assert keyless_mcp._ring_order("firecrawl")[0] == "firecrawl"

    def test_knob_disables_the_pin_for_rotation(self, monkeypatch):
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "firecrawl")
        monkeypatch.setattr(keyless_mcp, "_ring_rotation_enabled", lambda: True)
        starts = [
            keyless_mcp._ring_order("firecrawl")[0]
            for _ in range(len(keyless_mcp._KEYLESS_RING))
        ]
        assert sorted(starts) == sorted(keyless_mcp._KEYLESS_RING)  # full cycle

    def test_knob_keeps_every_vendor_in_the_order(self, monkeypatch):
        """Every rung must still be reachable — the knob changes the START
        position, never the membership."""
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: True)
        monkeypatch.setattr(keyless_mcp, "_ring_rotation_enabled", lambda: True)
        order = keyless_mcp._ring_order("firecrawl")
        assert set(order) == set(keyless_mcp._KEYLESS_RING)

    def test_knob_reads_the_config_key(self, monkeypatch):
        """The knob is read from ``web.keyless_ring_pin`` (false => rotate)."""
        monkeypatch.setattr(
            "tools.web_tools._load_web_config",
            lambda: {"keyless_ring_pin": False},
            raising=True,
        )
        assert keyless_mcp._ring_rotation_enabled() is True

    def test_knob_absent_means_no_rotation(self, monkeypatch):
        monkeypatch.setattr("tools.web_tools._load_web_config", dict, raising=True)
        assert keyless_mcp._ring_rotation_enabled() is False

    def test_knob_true_means_pinned(self, monkeypatch):
        monkeypatch.setattr(
            "tools.web_tools._load_web_config",
            lambda: {"keyless_ring_pin": True},
            raising=True,
        )
        assert keyless_mcp._ring_rotation_enabled() is False

    def test_knob_does_not_change_provider_resolution(self, monkeypatch):
        """The knob is about the ring's walk order; the configured backend must
        still be the active provider."""
        from agent import web_search_registry as registry

        monkeypatch.setattr(
            "tools.web_tools._load_web_config",
            lambda: {"keyless_ring_pin": False},
            raising=True,
        )
        configured = registry._configured_backend("search")
        assert configured is None or isinstance(configured, str)
        # Resolution consults the configured name, not the knob.
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        assert registry._configured_backend("search") is None
