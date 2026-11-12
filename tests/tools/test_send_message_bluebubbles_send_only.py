"""Standalone BlueBubbles send must use connect(send_only=True)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from plugins.platforms.bluebubbles.adapter import _standalone_send


def test_standalone_send_connects_send_only_then_sends_and_disconnects(monkeypatch):
    """Regression: the send_message / cron path must not bind the gateway webhook."""
    fake_adapter = MagicMock()
    fake_adapter.connect = AsyncMock(return_value=True)
    fake_adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="bb-msg-1", error=None)
    )
    fake_adapter.disconnect = AsyncMock()

    monkeypatch.setattr(
        "plugins.platforms.bluebubbles.adapter.BlueBubblesAdapter",
        MagicMock(return_value=fake_adapter),
    )
    monkeypatch.setattr(
        "plugins.platforms.bluebubbles.adapter.check_bluebubbles_requirements",
        lambda: True,
    )

    result = asyncio.run(
        _standalone_send(
            SimpleNamespace(extra={"server_url": "http://127.0.0.1:1234", "password": "x"}),
            "chat-1",
            "hello",
        )
    )

    fake_adapter.connect.assert_awaited_once_with(send_only=True)
    fake_adapter.send.assert_awaited_once_with("chat-1", "hello")
    fake_adapter.disconnect.assert_awaited_once()
    assert result.get("success") is True
    assert result.get("platform") == "bluebubbles"
    assert result.get("message_id") == "bb-msg-1"
