import asyncio
from types import SimpleNamespace

import pytest

from tools.telegram_inbox_tool import (
    _archive_telegram_inbox_item_async,
    _parse_csv_ids,
    telegram_inbox_tool,
)


class FakeBot:
    def __init__(self):
        self.calls = []

    async def copy_message(self, **kwargs):
        self.calls.append(("copy_message", kwargs))
        return SimpleNamespace(message_id=9001)

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        return SimpleNamespace(message_id=9002)

    async def delete_message(self, **kwargs):
        self.calls.append(("delete_message", kwargs))
        return True

    async def set_message_reaction(self, **kwargs):
        self.calls.append(("set_message_reaction", kwargs))
        return True


def test_parse_csv_ids_ignores_empty_values():
    assert _parse_csv_ids(" 2,3,, 5 ") == {"2", "3", "5"}


@pytest.mark.asyncio
async def test_archive_copy_card_delete_only_after_success(monkeypatch):
    fake_bot = FakeBot()
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_DELETE_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_SOURCE_THREADS", "2")
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_TARGET_THREADS", "8,3,5,6,7")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-1001")

    result = await _archive_telegram_inbox_item_async(
        bot=fake_bot,
        source_chat_id="-1001",
        source_message_id="42",
        source_thread_id="2",
        target_chat_id="-1001",
        target_thread_id="8",
        archive_card="Captured → archived",
        mode="copy_original_card_then_delete",
    )

    assert result["success"] is True
    assert result["deleted_source"] is True
    assert [name for name, _ in fake_bot.calls] == [
        "copy_message",
        "send_message",
        "delete_message",
    ]
    copy_kwargs = fake_bot.calls[0][1]
    assert copy_kwargs == {
        "chat_id": -1001,
        "from_chat_id": -1001,
        "message_id": 42,
        "message_thread_id": 8,
    }
    delete_kwargs = fake_bot.calls[2][1]
    assert delete_kwargs == {"chat_id": -1001, "message_id": 42}


@pytest.mark.asyncio
async def test_archive_refuses_delete_when_not_enabled(monkeypatch):
    fake_bot = FakeBot()
    monkeypatch.delenv("TELEGRAM_ZERO_INBOX_DELETE_ENABLED", raising=False)
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_SOURCE_THREADS", "2")
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_TARGET_THREADS", "8")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-1001")

    result = await _archive_telegram_inbox_item_async(
        bot=fake_bot,
        source_chat_id="-1001",
        source_message_id="42",
        source_thread_id="2",
        target_chat_id="-1001",
        target_thread_id="8",
        archive_card="Captured → archived",
        mode="copy_original_card_then_delete",
    )

    assert result["success"] is False
    assert "delete mode is disabled" in result["error"]
    assert fake_bot.calls == []


@pytest.mark.asyncio
async def test_archive_refuses_unconfigured_source_thread(monkeypatch):
    fake_bot = FakeBot()
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_DELETE_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_SOURCE_THREADS", "2")
    monkeypatch.setenv("TELEGRAM_ZERO_INBOX_TARGET_THREADS", "8")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-1001")

    result = await _archive_telegram_inbox_item_async(
        bot=fake_bot,
        source_chat_id="-1001",
        source_message_id="42",
        source_thread_id="99",
        target_chat_id="-1001",
        target_thread_id="8",
        archive_card="Captured → archived",
        mode="copy_original_card_then_delete",
    )

    assert result["success"] is False
    assert "source_thread_id is not allowed" in result["error"]
    assert fake_bot.calls == []


def test_tool_validates_required_fields():
    payload = telegram_inbox_tool({"source_chat_id": "-1001"})
    assert "source_message_id" in payload
