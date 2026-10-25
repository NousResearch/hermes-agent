"""Concurrent connector probes must not tear down each other's event loop.

Regression: probe servers are deliberately never registered in ``tools.mcp_tool._servers``, so
``_stop_mcp_loop(only_if_idle=True)`` saw an idle loop as soon as the FIRST of two concurrent
probes finished and stopped the loop under the sibling that was still connecting — surfacing as
one of two simultaneously refreshed Desktop/dashboard connector cards failing at random.
"""

from tools import mcp_tool_loop


def test_probe_lease_keeps_idle_stop_from_running(monkeypatch):
    monkeypatch.setattr(mcp_tool_loop, "_active_probes", 0, raising=False)

    with mcp_tool_loop._mcp_probe_lease():
        assert mcp_tool_loop._stop_mcp_loop(only_if_idle=True) is False
        with mcp_tool_loop._mcp_probe_lease():  # nested/concurrent probe
            assert mcp_tool_loop._stop_mcp_loop(only_if_idle=True) is False
        # Inner probe released; the outer one still holds the loop.
        assert mcp_tool_loop._stop_mcp_loop(only_if_idle=True) is False

    assert mcp_tool_loop._active_probes == 0
    # No lease and no loop running: the idle stop is now allowed to proceed.
    assert mcp_tool_loop._stop_mcp_loop(only_if_idle=True) is True


def test_probe_lease_releases_on_exception(monkeypatch):
    monkeypatch.setattr(mcp_tool_loop, "_active_probes", 0, raising=False)

    class Boom(Exception):
        pass

    try:
        with mcp_tool_loop._mcp_probe_lease():
            raise Boom
    except Boom:
        pass

    assert mcp_tool_loop._active_probes == 0
