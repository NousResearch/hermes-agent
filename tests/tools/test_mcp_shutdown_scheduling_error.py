"""A transport that could not be *scheduled* is an exit signal, not an MCP health signal.

Observed on a long-lived gateway (7 remote MCP servers): one restart tore down the interpreter /
the pool backing the reconnect, and every server charged its reconnect budget with
``RuntimeError: cannot schedule new futures after shutdown`` — 5 futile retries each, then a
``degraded → parked`` WARNING per server, which reads as a real outage in the error log (77 such
WARNINGs in 48h, all clustered inside gateway restarts).

``tools.interpreter_shutdown.interpreter_shutting_down`` already covers this class for cron
delivery, tool submission and the conversation retry (#55924/#58720); the MCP reconnect ladder is a
sibling call path that was missed.
"""

import asyncio

import pytest

from tools.mcp_tool import MCPServerTask


def _task(monkeypatch, retries: int = 0) -> MCPServerTask:
    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    task = MCPServerTask("t")
    task._ever_connected = True
    task._reconnect_retries = retries
    return task


@pytest.mark.no_isolate
@pytest.mark.parametrize("message", [
    "cannot schedule new futures after shutdown",
    "cannot schedule new futures after interpreter shutdown",
])
def test_shutdown_scheduling_error_exits_without_charging_the_ladder(monkeypatch, caplog, message):
    from tools.mcp_tool_server_run import _RetryBudget

    async def _scenario():
        task = _task(monkeypatch)
        with caplog.at_level("DEBUG", logger="tools.mcp_tool"):
            out = await task._on_transport_error(RuntimeError(message), _RetryBudget())
        return task, out

    task, out = asyncio.run(_scenario())

    assert out is False, "shutdown-race transport error must end the run loop, not retry"
    assert task._reconnect_retries == 0, "a shutdown race is not a reconnect charge"
    assert "parking" not in caplog.text, "shutdown must never park a server"
    assert "disconnected during shutdown" in caplog.text


@pytest.mark.no_isolate
def test_shutdown_scheduling_error_never_parks_an_exhausted_budget(monkeypatch, caplog):
    """Even with the budget exhausted (the live pattern), no park WARNING is emitted."""
    from tools import mcp_tool_server_run
    from tools.mcp_tool_server_run import _RetryBudget

    parked: list[str] = []

    async def _scenario():
        task = _task(monkeypatch, retries=mcp_tool_server_run._core._MAX_RECONNECT_RETRIES + 1)

        async def _fake_park(reason):
            parked.append(reason)
            return False

        monkeypatch.setattr(task, "_park", _fake_park, raising=False)
        with caplog.at_level("DEBUG", logger="tools.mcp_tool"):
            return await task._on_transport_error(
                RuntimeError("cannot schedule new futures after shutdown"), _RetryBudget())

    out = asyncio.run(_scenario())

    assert out is False and parked == []
    assert "parking" not in caplog.text


@pytest.mark.no_isolate
def test_genuine_transient_error_still_charges_the_ladder(monkeypatch):
    """Control: a real transport drop keeps the existing retry behaviour."""
    from tools import mcp_tool
    from tools.mcp_tool_server_run import _RetryBudget

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 3)

    async def _scenario():
        task = _task(monkeypatch)
        budget = _RetryBudget()
        out = await task._on_transport_error(ConnectionError("sse stream dropped"), budget)
        return task, out

    task, out = asyncio.run(_scenario())

    assert out is True
    assert task._reconnect_retries == 1
