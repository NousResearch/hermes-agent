import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import discord_approval_box_tool as approval_tool


def test_first_resolution_wins_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(approval_tool, "get_hermes_home", lambda: Path(tmp_path))

    record = approval_tool.create_approval_record(
        title="Email draft",
        body="Draft for client review.",
        drive_url="https://drive.google.com/file/d/example/view",
        channel_id="123",
    )
    resolved = approval_tool.resolve_approval(record["id"], "approved", "Willie")

    assert resolved is not None
    assert resolved["status"] == "approved"
    assert resolved["resolved_by"] == "Willie"
    assert approval_tool.resolve_approval(record["id"], "rejected", "Other") is None
    assert approval_tool.get_approval_record(record["id"])["status"] == "approved"


@pytest.mark.asyncio
async def test_discord_deliverable_approval_has_exactly_three_review_controls():
    sent = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send_deliverable_approval(
        chat_id="555",
        title="Ron article-email draft",
        body="Prepared for review before release.",
        drive_url="https://drive.google.com/file/d/example/view",
        approval_id="abc123",
    )

    assert result.success is True
    assert "Approve" in sent["content"]
    assert "Needs Work" in sent["content"]
    assert "Reject" in sent["content"]
    assert sent["view"] is not None


def test_approval_tool_schedules_delivery_on_gateway_loop(tmp_path, monkeypatch):
    """Approval cards must use Discord's owning event loop, not _run_async."""
    monkeypatch.setattr(approval_tool, "get_hermes_home", lambda: Path(tmp_path))
    gateway_loop = object()

    class CompletedFuture:
        def result(self, timeout):
            assert timeout == 30
            return SimpleNamespace(success=True, message_id="1234")

    async def send_deliverable_approval(**_kwargs):
        return None

    adapter = SimpleNamespace(
        _event_loop=gateway_loop,
        send_deliverable_approval=send_deliverable_approval,
    )
    runner = SimpleNamespace(adapters={Platform.DISCORD: adapter})
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

    scheduled = {}

    def fake_schedule(coro, loop):
        scheduled["loop"] = loop
        coro.close()
        return CompletedFuture()

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", fake_schedule)
    payload = json.loads(approval_tool.discord_approval_box_tool({
        "title": "Review me",
        "body": "Draft is ready.",
        "channel_id": "discord",
    }))

    assert payload["success"] is True
    assert payload["message_id"] == "1234"
    assert scheduled["loop"] is gateway_loop
