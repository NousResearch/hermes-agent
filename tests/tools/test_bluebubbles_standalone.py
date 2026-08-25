import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from tools.send_message_tool import _send_bluebubbles


@pytest.mark.asyncio
async def test_standalone_bluebubbles_send_uses_outbound_only_lifecycle(monkeypatch):
    calls = []

    async def fail_full_connect(self, *, is_reconnect=False):
        raise AssertionError("standalone delivery must not bind or register a webhook")

    async def fail_full_disconnect(self):
        raise AssertionError("standalone delivery must not unregister the live webhook")

    async def connect_outbound(self):
        calls.append("connect_outbound")
        return True

    async def disconnect_outbound(self):
        calls.append("disconnect_outbound")

    async def send(self, chat_id, message, **kwargs):
        calls.append(("send", chat_id, message))
        return SendResult(success=True, message_id="message-guid")

    monkeypatch.setattr(BlueBubblesAdapter, "connect", fail_full_connect)
    monkeypatch.setattr(BlueBubblesAdapter, "disconnect", fail_full_disconnect)
    monkeypatch.setattr(BlueBubblesAdapter, "connect_outbound", connect_outbound, raising=False)
    monkeypatch.setattr(BlueBubblesAdapter, "disconnect_outbound", disconnect_outbound, raising=False)
    monkeypatch.setattr(BlueBubblesAdapter, "send", send)

    result = await _send_bluebubbles(
        {"server_url": "http://bluebubbles.test", "password": "secret"},
        "iMessage;-;user@example.com",
        "Reminder text",
    )

    assert result == {
        "success": True,
        "platform": "bluebubbles",
        "chat_id": "iMessage;-;user@example.com",
        "message_id": "message-guid",
    }
    assert calls == [
        "connect_outbound",
        ("send", "iMessage;-;user@example.com", "Reminder text"),
        "disconnect_outbound",
    ]
