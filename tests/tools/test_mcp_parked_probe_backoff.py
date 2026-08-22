"""Tests for the parked self-probe backoff ladder.

A parked server self-probes on a timer (its tools are deregistered, so
nothing else can revive it). The original fixed 300 s cadence meant a
server parked on a permanent-looking failure — e.g. a stdio binary
dying at startup without its credentials — crash-looped forever: every
probe spawns the process, it exits before the MCP handshake, and 5
minutes later the cycle repeats. The interval must now double after
each failed probe (capped, jittered like the reconnect backoff so
restarted gateways do not probe in lockstep), reset when a session
proves healthy, and never delay an explicit reconnect.
"""

import asyncio

import pytest

from tools import mcp_tool
from tools.mcp_tool import MCPServerTask

# _parked_probe_interval() applies +/- _BACKOFF_JITTER (20%) around the
# ladder rung, mirroring the reconnect backoff. Assert on the band, not
# the point value.
_JITTER = mcp_tool._BACKOFF_JITTER


def _assert_in_jitter_band(value, expected):
    assert pytest.approx(expected, rel=_JITTER + 1e-9) == value


class TestParkedProbeInterval:
    """_parked_probe_interval ladder maths (jitter band per rung)."""

    def test_base_interval_initially(self):
        server = MCPServerTask("srv")
        assert server._parked_probe_streak == 0
        _assert_in_jitter_band(
            server._parked_probe_interval(), mcp_tool._PARKED_RETRY_INTERVAL
        )

    def test_interval_doubles_per_failed_probe(self):
        server = MCPServerTask("srv")
        server._parked_probe_streak = 3
        expected = mcp_tool._PARKED_RETRY_INTERVAL * (2 ** 3)
        _assert_in_jitter_band(server._parked_probe_interval(), expected)

    def test_interval_capped(self):
        server = MCPServerTask("srv")
        server._parked_probe_streak = 99
        _assert_in_jitter_band(
            server._parked_probe_interval(), mcp_tool._PARKED_RETRY_INTERVAL_MAX
        )

    def test_cap_boundary_exact_rung(self):
        """At streak == log2(cap/base) the interval equals the cap exactly.

        Pins the ladder's top rung against floating-point drift: with the
        current constants that rung is
        ceil(log2(3600 / 300)) == ceil(3.58) == 4, and 300 * 2**4 == 4800
        is clamped to 3600 inside min() BEFORE jitter — so the jitter band
        sits around the cap, never above it.
        """
        server = MCPServerTask("srv")
        server._parked_probe_streak = mcp_tool._PARKED_PROBE_STREAK_CAP
        interval = server._parked_probe_interval()
        # The jittered value must never exceed the cap by more than the
        # jitter band allows, and the pre-jitter rung is exactly the cap.
        assert interval <= mcp_tool._PARKED_RETRY_INTERVAL_MAX * (1 + _JITTER + 1e-9)
        assert interval >= mcp_tool._PARKED_RETRY_INTERVAL_MAX * (1 - _JITTER - 1e-9)
        # The ladder maths itself (pre-jitter) must hit the cap exactly at
        # the boundary rung with no floating-point drift.
        rung = mcp_tool._PARKED_RETRY_INTERVAL * (
            2 ** mcp_tool._PARKED_PROBE_STREAK_CAP
        )
        assert min(rung, mcp_tool._PARKED_RETRY_INTERVAL_MAX) == (
            mcp_tool._PARKED_RETRY_INTERVAL_MAX
        )

    def test_cap_constant_sane(self):
        # The cap must stay above the base, otherwise the ladder is a no-op.
        assert mcp_tool._PARKED_RETRY_INTERVAL_MAX > mcp_tool._PARKED_RETRY_INTERVAL

    def test_jitter_applied(self):
        """Repeated draws at the base rung must vary (jitter is live)."""
        server = MCPServerTask("srv")
        draws = {server._parked_probe_interval() for _ in range(24)}
        assert len(draws) > 1


class TestParkedProbeStreakLifecycle:
    """The streak grows on timed self-probe wakes and resets on health."""

    def test_timed_wake_increments_streak(self):
        """A self-probe (timeout elapsed, no explicit event) bumps the streak."""
        async def _scenario():
            server = MCPServerTask("srv")
            # Tiny timeout so the timed wake fires immediately.
            result = await server._wait_for_reconnect_or_shutdown(timeout=0.01)
            return server, result

        server, result = asyncio.run(_scenario())
        assert result == "reconnect"
        assert server._parked_probe_streak == 1

    def test_explicit_reconnect_leaves_streak_alone(self):
        """An explicit _reconnect_event wake must not grow the backoff."""
        async def _scenario():
            server = MCPServerTask("srv")
            server._parked_probe_streak = 2
            server._reconnect_event.set()
            # Long timeout — the explicit event must win the race.
            result = await server._wait_for_reconnect_or_shutdown(timeout=5)
            return server, result

        server, result = asyncio.run(_scenario())
        assert result == "reconnect"
        assert server._parked_probe_streak == 2

    def test_shutdown_wake_leaves_streak_alone(self):
        async def _scenario():
            server = MCPServerTask("srv")
            server._parked_probe_streak = 1
            server._shutdown_event.set()
            result = await server._wait_for_reconnect_or_shutdown(timeout=5)
            return server, result

        server, result = asyncio.run(_scenario())
        assert result == "shutdown"
        assert server._parked_probe_streak == 1

    def test_streak_resets_when_session_proves_healthy(self):
        server = MCPServerTask("srv")
        server._parked_probe_streak = 5
        server._mark_session_proven()
        assert server._parked_probe_streak == 0

    def test_streak_clamped_at_cap(self):
        """Timed wakes past the top rung never grow the counter further."""
        async def _scenario():
            server = MCPServerTask("srv")
            server._parked_probe_streak = mcp_tool._PARKED_PROBE_STREAK_CAP
            result = await server._wait_for_reconnect_or_shutdown(timeout=0.01)
            return server, result

        server, result = asyncio.run(_scenario())
        assert result == "reconnect"
        assert server._parked_probe_streak == mcp_tool._PARKED_PROBE_STREAK_CAP
