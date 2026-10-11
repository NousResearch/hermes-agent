"""Outbound-only sends must never borrow a live gateway adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


def test_force_standalone_bypasses_live_slack_adapter():
    calls = []

    async def standalone(pconfig, chat_id, message, **kwargs):
        calls.append((chat_id, message, kwargs))
        return {"success": True, "message_id": "standalone-1"}

    def live_adapter_must_not_be_resolved(_platform):
        raise AssertionError("force_standalone resolved a live adapter")

    pconfig = SimpleNamespace(enabled=False, token=None, extra={})
    with patch("tools.send_message_tool._live_adapter", live_adapter_must_not_be_resolved), \
         patch("tools.send_message_tool._plugin_standalone_sender", return_value=(standalone, None)):
        result = asyncio.run(_send_to_platform(
            Platform.SLACK, pconfig, "C1", "hello", force_standalone=True))

    assert result == {"success": True, "message_id": "standalone-1"}
    assert calls == [("C1", "hello", {"thread_id": None, "media_files": []})]
