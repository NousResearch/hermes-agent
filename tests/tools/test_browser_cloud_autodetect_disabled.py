"""Auto-detect must only consider *registered* browser providers (#134152).

``_autodetect_cloud_provider`` used to instantiate the bundled provider classes directly,
bypassing the browser provider registry. A provider whose plugin is in ``plugins.disabled``
is never registered, yet auto-detect still handed it back while the explicit
``browser.cloud_provider`` path correctly refused it. Auto-detect now resolves through the
registry (``agent.browser_registry._resolve(None)``), so disabled plugins are skipped on
both paths.
"""

import pytest

import tools.browser_tool as browser_tool
from agent.browser_provider import BrowserProvider
from tools import browser_tool_cloud as bt_cloud


class _Probe(BrowserProvider):
    """Minimal registry-visible provider."""

    def __init__(self, name: str, available):
        self._name = name
        self.available = available

    @property
    def name(self) -> str:
        return self._name

    def is_available(self):
        if isinstance(self.available, Exception):
            raise self.available
        return self.available

    def create_session(self, task_id: str):
        return {}

    def close_session(self, session_id: str) -> bool:
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Run each test against an empty real registry.

    Discovery is suppressed (real discovery would register the bundled plugins and
    defeat the disabled scenario) and the registry is cleared before and after so
    neither direction leaks state.
    """
    import agent.browser_registry as browser_registry

    browser_registry._reset_for_tests()
    yield browser_registry
    browser_registry._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_plugin_discovery(monkeypatch):
    monkeypatch.setattr(bt_cloud, "_ensure_browser_plugins_loaded", lambda: None)


class TestAutodetectRespectsDisabledPlugins:
    def test_disabled_plugin_with_credentials_is_not_autodetected(self, monkeypatch):
        """plugins.disabled keeps browser-use unregistered; auto-detect must not fall back to
        instantiating the bundled class even though its credentials are present (#134152)."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "k1")
        from plugins.browser.browser_use.provider import BrowserUseBrowserProvider

        # The pre-fix escape hatch really is satisfied here: the bundled class,
        # instantiated directly, reports available. The registry (plugin disabled) is empty.
        assert BrowserUseBrowserProvider().is_available() is True

        assert bt_cloud._autodetect_cloud_provider() is None

    def test_registered_provider_is_autodetected(self, _isolated_registry):
        provider = _Probe("browser-use", True)
        _isolated_registry.register_provider(provider)

        assert bt_cloud._autodetect_cloud_provider() is provider

    def test_preference_order_is_browser_use_then_browserbase(self, _isolated_registry):
        browser_use = _Probe("browser-use", True)
        browserbase = _Probe("browserbase", True)
        _isolated_registry.register_provider(browser_use)
        _isolated_registry.register_provider(browserbase)

        assert bt_cloud._autodetect_cloud_provider() is browser_use

    def test_unavailable_registered_provider_falls_through(self, _isolated_registry):
        browser_use = _Probe("browser-use", False)
        browserbase = _Probe("browserbase", True)
        _isolated_registry.register_provider(browser_use)
        _isolated_registry.register_provider(browserbase)

        assert bt_cloud._autodetect_cloud_provider() is browserbase

    def test_raising_provider_is_treated_as_unavailable(self, _isolated_registry):
        raising = _Probe("browser-use", RuntimeError("boom"))
        browserbase = _Probe("browserbase", True)
        _isolated_registry.register_provider(raising)
        _isolated_registry.register_provider(browserbase)

        assert bt_cloud._autodetect_cloud_provider() is browserbase


class TestAutodetectThroughFullResolution:
    """``browser.cloud_provider`` unset → resolution lands on auto-detect, same registry view."""

    @pytest.fixture(autouse=True)
    def _reset_resolver_state(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
        monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
        yield

    def test_unset_cloud_provider_with_disabled_plugin_yields_local(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: {"browser": {}}
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "k1")

        assert bt_cloud._resolve_cloud_provider_uncached() is None
