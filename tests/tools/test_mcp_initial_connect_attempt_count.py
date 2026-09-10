import asyncio
import logging
from unittest.mock import AsyncMock

from tools import mcp_tool
from tools.mcp_tool_server_run import _RetryBudget


def test_initial_connect_exhaustion_logs_actual_attempt_count(monkeypatch, caplog):
    """The give-up warning reports attempts made, not the retry-only limit."""

    async def _run():
        server = mcp_tool.MCPServerTask("attempt-count")
        budget = _RetryBudget(initial_retries=mcp_tool._MAX_INITIAL_CONNECT_RETRIES)
        park = AsyncMock(return_value=False)
        monkeypatch.setattr(server, "_park_initial_failure", park)
        exc = ConnectionError("transient failure")

        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            result = await server._on_initial_connect_error(
                exc, exc, "transient", budget
            )

        assert result is False
        assert budget.initial_retries == mcp_tool._MAX_INITIAL_CONNECT_RETRIES + 1
        assert (
            f"failed initial connection after {budget.initial_retries} attempts"
            in caplog.text
        )
        park.assert_awaited_once_with(
            exc, "after initial connection failures", budget
        )

    asyncio.run(_run())
