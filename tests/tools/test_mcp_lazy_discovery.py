"""Tests for Phase 2 lazy MCP discovery.

When the active platform's toolsets include the ``no_mcp`` sentinel
(e.g. api_server), the gateway/CLI skip eager MCP discovery at startup and
defer it until the first child-agent build that needs MCP toolsets. This
module verifies:

* ``mark_eager_discovery_skipped()`` / ``ensure_mcp_discovered()`` behavior,
  including the one-shot, thread-safe, failure-tolerant contract.
* ``_active_platform_uses_no_mcp()`` config-dict resolution in gateway.run.

See NousResearch/hermes-agent#32668.
"""

import threading
import time
from unittest.mock import patch

import pytest

import tools.mcp_tool as mcp_tool


@pytest.fixture(autouse=True)
def reset_lazy_discovery_state():
    """Save/restore the module-level lazy-discovery globals around each test.

    Discovery state is process-global; without this fixture a test that sets
    ``_eager_discovery_skipped`` or the done Event would leak into siblings.
    """
    saved_skipped = mcp_tool._eager_discovery_skipped
    saved_event = mcp_tool._lazy_discovery_done

    # Start each test from a clean slate.
    mcp_tool._eager_discovery_skipped = False
    mcp_tool._lazy_discovery_done = threading.Event()

    try:
        yield
    finally:
        mcp_tool._eager_discovery_skipped = saved_skipped
        mcp_tool._lazy_discovery_done = saved_event


class TestMarkEagerDiscoverySkipped:
    def test_sets_flag_true(self):
        assert mcp_tool._eager_discovery_skipped is False
        mcp_tool.mark_eager_discovery_skipped()
        assert mcp_tool._eager_discovery_skipped is True


class TestEnsureMcpDiscovered:
    def test_noop_when_not_skipped(self):
        """When eager discovery ran (skipped=False), ensure_* is a no-op."""
        with patch.object(mcp_tool, "discover_mcp_tools") as mock_discover:
            mcp_tool.ensure_mcp_discovered()
            mock_discover.assert_not_called()
        # The done-Event must remain unset so a later skipped flow can still run.
        assert not mcp_tool._lazy_discovery_done.is_set()

    def test_calls_discover_once_when_skipped(self):
        """First call after skip triggers discovery exactly once."""
        mcp_tool.mark_eager_discovery_skipped()
        with patch.object(mcp_tool, "discover_mcp_tools") as mock_discover:
            mcp_tool.ensure_mcp_discovered()
            mock_discover.assert_called_once()
        assert mcp_tool._lazy_discovery_done.is_set()

    def test_idempotent_after_done(self):
        """Second call after the done-Event is set never re-runs discovery."""
        mcp_tool.mark_eager_discovery_skipped()
        with patch.object(mcp_tool, "discover_mcp_tools") as mock_discover:
            mcp_tool.ensure_mcp_discovered()
            mcp_tool.ensure_mcp_discovered()
            mcp_tool.ensure_mcp_discovered()
            assert mock_discover.call_count == 1

    def test_thread_safe_single_discovery(self):
        """10 concurrent callers must trigger discovery exactly once."""
        mcp_tool.mark_eager_discovery_skipped()

        call_count = 0
        count_lock = threading.Lock()

        def slow_discover():
            nonlocal call_count
            with count_lock:
                call_count += 1
            # Hold the lazy lock long enough that all threads contend.
            time.sleep(0.05)
            return []

        start = threading.Event()
        threads = []

        def worker():
            start.wait()
            mcp_tool.ensure_mcp_discovered()

        with patch.object(mcp_tool, "discover_mcp_tools", side_effect=slow_discover):
            for _ in range(10):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            start.set()
            for t in threads:
                t.join(timeout=5)

        assert call_count == 1
        assert mcp_tool._lazy_discovery_done.is_set()

    def test_failure_does_not_propagate(self):
        """If discover raises, ensure_* swallows it, sets done, logs a warning."""
        mcp_tool.mark_eager_discovery_skipped()

        def boom():
            raise RuntimeError("connection refused")

        with patch.object(mcp_tool, "discover_mcp_tools", side_effect=boom), \
                patch.object(mcp_tool.logger, "warning") as mock_warning:
            # Must not raise.
            mcp_tool.ensure_mcp_discovered()

        # Done-Event set to prevent retry storms after a hard failure.
        assert mcp_tool._lazy_discovery_done.is_set()
        mock_warning.assert_called_once()

    def test_failure_prevents_retry(self):
        """After a failed discovery, subsequent calls do not retry."""
        mcp_tool.mark_eager_discovery_skipped()

        with patch.object(
            mcp_tool, "discover_mcp_tools", side_effect=RuntimeError("boom")
        ) as mock_discover, patch.object(mcp_tool.logger, "warning"):
            mcp_tool.ensure_mcp_discovered()
            mcp_tool.ensure_mcp_discovered()
            assert mock_discover.call_count == 1


class TestActivePlatformUsesNoMcp:
    def test_true_for_api_server_with_no_mcp(self):
        from gateway.run import _active_platform_uses_no_mcp

        config = {"platform_toolsets": {"api_server": ["delegation", "no_mcp"]}}
        with patch.dict("os.environ", {"HERMES_PLATFORM": "api_server"}):
            assert _active_platform_uses_no_mcp(config) is True

    def test_false_for_cli(self):
        from gateway.run import _active_platform_uses_no_mcp

        config = {
            "platform_toolsets": {
                "cli": ["delegation"],
                "api_server": ["delegation", "no_mcp"],
            }
        }
        with patch.dict("os.environ", {"HERMES_PLATFORM": "cli"}):
            assert _active_platform_uses_no_mcp(config) is False

    def test_false_when_platform_toolsets_missing(self):
        from gateway.run import _active_platform_uses_no_mcp

        config = {"some_other_key": {}}
        with patch.dict("os.environ", {"HERMES_PLATFORM": "api_server"}):
            assert _active_platform_uses_no_mcp(config) is False

    def test_false_for_unconfigured_platform(self):
        """Platform present in env but absent from platform_toolsets -> False."""
        from gateway.run import _active_platform_uses_no_mcp

        config = {"platform_toolsets": {"cli": ["delegation"]}}
        with patch.dict("os.environ", {"HERMES_PLATFORM": "api_server"}):
            assert _active_platform_uses_no_mcp(config) is False

    def test_defaults_to_cli_when_env_unset(self):
        """No HERMES_PLATFORM -> defaults to 'cli', which lacks no_mcp here."""
        from gateway.run import _active_platform_uses_no_mcp

        config = {"platform_toolsets": {"api_server": ["delegation", "no_mcp"]}}
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("HERMES_PLATFORM", None)
            os.environ.pop("HERMES_SESSION_PLATFORM", None)
            assert _active_platform_uses_no_mcp(config) is False
