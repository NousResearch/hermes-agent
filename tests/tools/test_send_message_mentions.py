"""send_message propagation and boundary checks for WhatsApp mentions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from gateway.config import Platform
from tools import send_message_tool as send_module


@pytest.mark.parametrize(
    "args",
    [
        {"action": "send", "target": "telegram:-100123", "message": "hello", "mentions": ["15551234567"]},
        {"action": "send", "target": "whatsapp:15551234567", "message": "hello", "mentions": ["15557654321"]},
        {"action": "send", "target": "whatsapp:120363408391911677@g.us", "message": "hello", "mentions": ["alice"]},
    ],
)
def test_tool_boundary_rejects_unsupported_or_malformed_mentions(args):
    result = json.loads(send_module.send_message_tool(args))

    assert "error" in result
    assert "mention" in result["error"].lower()


def test_whatsapp_mentions_route_to_adapter_delivery(monkeypatch):
    sender = AsyncMock(return_value={"success": True, "message_id": "m1"})
    monkeypatch.setattr(send_module, "_send_via_adapter", sender)

    result = asyncio.run(send_module._send_to_platform(
        Platform.WHATSAPP,
        SimpleNamespace(enabled=True, token=None, extra={}),
        "120363408391911677@g.us",
        "hello group",
        mentions=["15551234567@s.whatsapp.net", "149606612619433@lid"],
    ))

    assert result["success"] is True
    sender.assert_awaited_once_with(
        Platform.WHATSAPP,
        ANY,
        "120363408391911677@g.us",
        "hello group",
        thread_id=None,
        media_files=[],
        force_document=False,
        mentions=["15551234567@s.whatsapp.net", "149606612619433@lid"],
    )
