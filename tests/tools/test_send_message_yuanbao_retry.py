"""Yuanbao-specific send_message delivery retries."""

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from tools.send_message_tool import _send_yuanbao


def test_send_yuanbao_waits_for_adapter_to_reappear(monkeypatch):
    adapter = object()
    adapters = [None, adapter]
    fake_mod = ModuleType("gateway.platforms.yuanbao")
    fake_mod.get_active_adapter = MagicMock(side_effect=lambda: adapters.pop(0))
    fake_mod.send_yuanbao_direct = AsyncMock(return_value={"success": True, "message_id": "m1"})
    monkeypatch.setitem(sys.modules, "gateway.platforms.yuanbao", fake_mod)

    async def run_test():
        with patch("tools.send_message_tool.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            result = await _send_yuanbao("direct:alice", "hello")
            return result, sleep_mock

    result, sleep_mock = asyncio.run(run_test())

    assert result == {"success": True, "message_id": "m1"}
    sleep_mock.assert_awaited_once_with(2.0)
    fake_mod.send_yuanbao_direct.assert_awaited_once_with(
        adapter,
        "direct:alice",
        "hello",
        media_files=None,
    )


def test_send_yuanbao_returns_not_running_after_retry_window(monkeypatch):
    fake_mod = ModuleType("gateway.platforms.yuanbao")
    fake_mod.get_active_adapter = MagicMock(return_value=None)
    fake_mod.send_yuanbao_direct = AsyncMock()
    monkeypatch.setitem(sys.modules, "gateway.platforms.yuanbao", fake_mod)

    async def run_test():
        with patch("tools.send_message_tool.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            result = await _send_yuanbao("direct:alice", "hello")
            return result, sleep_mock

    result, sleep_mock = asyncio.run(run_test())

    assert "error" in result
    assert "Yuanbao adapter is not running" in result["error"]
    assert sleep_mock.await_count == 2
    fake_mod.send_yuanbao_direct.assert_not_awaited()
