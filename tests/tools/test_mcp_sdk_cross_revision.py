"""MCP SDK cross-revision response compatibility."""

from collections.abc import Mapping
from typing import Any

import anyio
import pytest
from anyio.abc import TaskStatus
from mcp import ClientSession
from mcp.shared.dispatcher import CallOptions, OnNotify, OnNotifyIntercept, OnRequest


class _ListToolsDispatcher:
    async def send_raw_request(
        self,
        method: str,
        params: Mapping[str, Any] | None,
        opts: CallOptions | None = None,
    ) -> dict[str, Any]:
        assert method == "tools/list"
        return {"tools": [], "cacheScope": ""}

    async def notify(
        self,
        method: str,
        params: Mapping[str, Any] | None,
        opts: CallOptions | None = None,
    ) -> None:
        raise AssertionError(f"unexpected notification: {method}")

    async def run(
        self,
        on_request: OnRequest,
        on_notify: OnNotify,
        on_notify_intercept: OnNotifyIntercept | None = None,
        *,
        task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started()
        await anyio.sleep_forever()


@pytest.mark.asyncio
async def test_future_cache_scope_is_ignored_for_older_negotiated_revision():
    """A future cache hint must not invalidate an older tools/list response."""
    session = ClientSession(dispatcher=_ListToolsDispatcher())
    session._negotiated_version = "2025-11-25"

    result = await session.list_tools()

    assert result.tools == []
    assert result.cache_scope == "private"
