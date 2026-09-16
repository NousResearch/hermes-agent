"""Tests for ``_get_cloud_provider()`` caching policy.

Regression coverage for issue #22324: a transient ``None`` from the resolver
must not be cached for the lifetime of the process. Cache only when:

* The user explicitly opts in to ``cloud_provider: local``, OR
* A provider is successfully resolved.

All other ``None`` outcomes (no credentials yet, config read error, explicit
provider instantiation failure) leave the cache unset so the next call retries.
"""
import logging
from unittest.mock import Mock

import pytest

import tools.browser_tool as browser_tool
from tools import browser_tool_cloud as bt_cloud


@pytest.fixture(autouse=True)
def _reset_resolver_state(monkeypatch):
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    yield


class TestCloudProviderCachePolicy:
    def test_cache_is_isolated_by_hermes_home(self, tmp_path, monkeypatch):
        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "profile-provider"}},
        )
        providers = {}
        resolutions = []

        def resolve(_name):
            home = str(get_hermes_home())
            resolutions.append(home)
            return providers[home]

        monkeypatch.setattr("tools.browser_tool_cloud._ensure_browser_plugins_loaded", lambda: None)
        monkeypatch.setattr("tools.browser_tool_cloud._registry_get_browser_provider", resolve)
        home_a = tmp_path / "browser-a"
        home_b = tmp_path / "browser-b"
        providers[str(home_a)] = Mock(name="provider-a")
        providers[str(home_b)] = Mock(name="provider-b")

        def resolve_for(home):
            token = set_hermes_home_override(home)
            try:
                return bt_cloud._get_cloud_provider()
            finally:
                reset_hermes_home_override(token)

        assert resolve_for(home_a) is providers[str(home_a)]
        assert resolve_for(home_b) is providers[str(home_b)]
        assert resolve_for(home_a) is providers[str(home_a)]
        assert resolutions == [str(home_a), str(home_b)]

    def test_same_profile_registry_replacement_invalidates_cache(
        self, tmp_path, monkeypatch
    ):
        from agent.browser_provider import BrowserProvider
        import agent.browser_registry as browser_registry
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        class Provider(BrowserProvider):
            def __init__(self, marker):
                self.marker = marker

            @property
            def name(self):
                return "cache-replacement"

            def is_available(self):
                return True

            def create_session(self, task_id):
                return {"marker": self.marker}

            def close_session(self, session_id):
                return True

            def emergency_cleanup(self, session_id):
                return None

        home = str((tmp_path / "same-profile").resolve())
        first = Provider("first")
        second = Provider("second")
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "cache-replacement"}},
        )
        monkeypatch.setattr("tools.browser_tool_cloud._ensure_browser_plugins_loaded", lambda: None)
        token = set_hermes_home_override(home)
        try:
            browser_registry.register_provider(first, scope=home)
            assert bt_cloud._get_cloud_provider() is first
            browser_registry.register_provider(second, scope=home)
            assert bt_cloud._get_cloud_provider() is second
        finally:
            current = browser_registry.snapshot_registration(
                "cache-replacement", scope=home
            )
            if current is not None:
                browser_registry.restore_registration(
                    "cache-replacement", current, None, scope=home
                )
            reset_hermes_home_override(token)

    def test_concurrent_registry_replacement_discards_stale_resolution(
        self, tmp_path, monkeypatch
    ):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        from agent.browser_provider import BrowserProvider
        import agent.browser_registry as browser_registry
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        class Provider(BrowserProvider):
            def __init__(self, marker):
                self.marker = marker

            @property
            def name(self):
                return "cache-race"

            def is_available(self):
                return True

            def create_session(self, task_id):
                return {"marker": self.marker}

            def close_session(self, session_id):
                return True

            def emergency_cleanup(self, session_id):
                return None

        home = str((tmp_path / "race-profile").resolve())
        first = Provider("first")
        second = Provider("second")
        paused = Event()
        release = Event()
        calls = 0
        original_get = browser_registry.get_provider

        def racing_get(name):
            nonlocal calls
            calls += 1
            resolved = original_get(name, scope=home)
            if calls == 1:
                paused.set()
                assert release.wait(timeout=2)
            return resolved

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "cache-race"}},
        )
        monkeypatch.setattr("tools.browser_tool_cloud._ensure_browser_plugins_loaded", lambda: None)
        monkeypatch.setattr("tools.browser_tool_cloud._registry_get_browser_provider", racing_get)
        browser_registry.register_provider(first, scope=home)

        def resolve():
            token = set_hermes_home_override(home)
            try:
                return bt_cloud._get_cloud_provider()
            finally:
                reset_hermes_home_override(token)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(resolve)
                assert paused.wait(timeout=1)
                browser_registry.register_provider(second, scope=home)
                release.set()
                assert future.result(timeout=2) is second
            assert calls == 2
        finally:
            release.set()
            current = browser_registry.snapshot_registration("cache-race", scope=home)
            if current is not None:
                browser_registry.restore_registration(
                    "cache-race", current, None, scope=home
                )

    def test_explicit_local_caches_permanently(self, monkeypatch):
        """`cloud_provider: local` is a positive choice and must stick."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "local"}},
        )

        assert bt_cloud._get_cloud_provider() is None
        assert browser_tool._cloud_provider_resolved is True

        # Even if config later changes, the cache stays.
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "browser-use"}},
        )
        assert bt_cloud._get_cloud_provider() is None


    def test_no_credentials_yet_does_not_cache_none(self, monkeypatch):
        """Auto-detect path with no creds: must NOT poison the cache."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {}},
        )

        bu_unconfigured = Mock()
        bu_unconfigured.is_available.return_value = False
        bb_unconfigured = Mock()
        bb_unconfigured.is_available.return_value = False
        monkeypatch.setattr(
            "tools.browser_tool_cloud.BrowserUseBrowserProvider", lambda: bu_unconfigured
        )
        monkeypatch.setattr(
            "tools.browser_tool_cloud.BrowserbaseBrowserProvider", lambda: bb_unconfigured
        )

        assert bt_cloud._get_cloud_provider() is None
        assert browser_tool._cloud_provider_resolved is False

        # Credentials self-heal — next call must retry and pick up the provider.
        healed = Mock(name="healed-provider")
        healed.is_available.return_value = True
        monkeypatch.setattr("tools.browser_tool_cloud.BrowserUseBrowserProvider", lambda: healed)

        assert bt_cloud._get_cloud_provider() is healed
        assert browser_tool._cloud_provider_resolved is True


    def test_explicit_provider_instantiation_failure_does_not_cache(
        self, monkeypatch, caplog
    ):
        """If instantiating the registered provider raises, log warning and don't cache."""
        def exploding_factory(name):
            raise RuntimeError("missing dependency")

        monkeypatch.setattr("tools.browser_tool_cloud._ensure_browser_plugins_loaded", lambda: None)
        monkeypatch.setattr("tools.browser_tool_cloud._registry_get_browser_provider", exploding_factory)
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "browser-use"}},
        )

        with caplog.at_level(logging.WARNING, logger="tools.browser_tool"):
            assert bt_cloud._get_cloud_provider() is None

        assert browser_tool._cloud_provider_resolved is False
        assert any(
            "browser-use" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )
