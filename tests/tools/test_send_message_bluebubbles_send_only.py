"""Tool-layer wiring: _send_bluebubbles must use connect(send_only=True)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tools.send_message_senders import _send_bluebubbles


def test_send_bluebubbles_connects_send_only_then_sends_and_disconnects():
    """Regression for #51763 keep_open: tool path must pass send_only=True."""
    fake_adapter = MagicMock()
    fake_adapter.connect = AsyncMock(return_value=True)
    fake_adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="bb-msg-1", error=None)
    )
    fake_adapter.disconnect = AsyncMock()

    fake_mod = MagicMock()
    fake_mod.BlueBubblesAdapter = MagicMock(return_value=fake_adapter)

    with patch(
        "tools.send_message_senders._gateway_platform_module",
        return_value=(fake_mod, None),
    ):
        result = asyncio.run(
            _send_bluebubbles(
                {"server_url": "http://127.0.0.1:1234", "password": "x"},
                "chat-1",
                "hello",
            )
        )

    fake_mod.BlueBubblesAdapter.assert_called_once()
    fake_adapter.connect.assert_awaited_once_with(send_only=True)
    fake_adapter.send.assert_awaited_once_with("chat-1", "hello")
    fake_adapter.disconnect.assert_awaited_once()
    assert result.get("success") is True
    assert result.get("platform") == "bluebubbles"
    assert result.get("message_id") == "bb-msg-1"
