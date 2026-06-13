"""Tests for delegate heartbeat stale threshold configuration."""

import pytest


class TestHeartbeatStaleThresholds:
    """Verify the heartbeat stale threshold constants are correct."""

    def test_idle_cycles_value(self):
        """IDLE stale cycles should be 15 (15 * 30s = 450s)."""
        from tools.delegate_tool import _HEARTBEAT_STALE_CYCLES_IDLE
        assert _HEARTBEAT_STALE_CYCLES_IDLE == 15

    def test_in_tool_cycles_value(self):
        """IN_TOOL stale cycles should be 40 (40 * 30s = 1200s)."""
        from tools.delegate_tool import _HEARTBEAT_STALE_CYCLES_IN_TOOL
        assert _HEARTBEAT_STALE_CYCLES_IN_TOOL == 40

    def test_idle_timeout_seconds(self):
        """Effective idle stale timeout: 15 * 30 = 450s (> typical LLM response time)."""
        from tools.delegate_tool import _HEARTBEAT_STALE_CYCLES_IDLE, _HEARTBEAT_INTERVAL
        effective = _HEARTBEAT_STALE_CYCLES_IDLE * _HEARTBEAT_INTERVAL
        assert effective == 450
        assert effective > 300  # Must be > 5 minutes for slow LLM responses

    def test_in_tool_timeout_seconds(self):
        """Effective in-tool stale timeout: 40 * 30 = 1200s (= 20 minutes)."""
        from tools.delegate_tool import _HEARTBEAT_STALE_CYCLES_IN_TOOL, _HEARTBEAT_INTERVAL
        effective = _HEARTBEAT_STALE_CYCLES_IN_TOOL * _HEARTBEAT_INTERVAL
        assert effective == 1200

    def test_interval_unchanged(self):
        """Heartbeat interval should remain 30s."""
        from tools.delegate_tool import _HEARTBEAT_INTERVAL
        assert _HEARTBEAT_INTERVAL == 30
