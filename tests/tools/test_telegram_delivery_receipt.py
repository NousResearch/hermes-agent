"""Standalone receipts report Telegram's returned topic, never the requested topic."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools import send_message_senders as senders


@pytest.mark.parametrize("media", [False, True], ids=["text", "photo"])
@pytest.mark.parametrize("returned_topic", [67890, 78901], ids=["matching", "different"])
@pytest.mark.parametrize("topic_field", ["thread", "direct", "direct_with_null_thread"])
def test_receipt_reports_server_topic(monkeypatch, tmp_path, capsys, media, returned_topic, topic_field):
    fields = {"message_thread_id": returned_topic}
    if topic_field != "thread":
        fields = {"direct_messages_topic": SimpleNamespace(topic_id=returned_topic)}
        if topic_field == "direct_with_null_thread":
            fields["message_thread_id"] = None
    message = SimpleNamespace(message_id=42, **fields)
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=message),
        send_photo=AsyncMock(return_value=message),
    )
    monkeypatch.setattr(senders, "_telegram_bot", lambda _token: bot)
    monkeypatch.setattr(senders, "_telegram_format", lambda text: (text, None, False))
    attachments = None
    if media:
        photo = tmp_path / "fixture.png"
        photo.write_bytes(b"synthetic image")
        attachments = [(str(photo), False)]

    receipt = asyncio.run(senders._send_telegram(
        "fake-token", "12345", "synthetic delivery", media_files=attachments, thread_id="67890",
    ))

    assert receipt["success"] is True
    assert receipt["message_id"] == "42"
    assert receipt["thread_id"] == str(returned_topic)
    transport = bot.send_photo if media else bot.send_message
    transport.assert_awaited_once()
    assert transport.await_args.kwargs["message_thread_id"] == 67890
    from hermes_cli.send_cmd import _emit_result

    assert _emit_result(json.dumps(receipt), json_mode=True, quiet=False) == 0
    assert json.loads(capsys.readouterr().out) == receipt


@pytest.mark.parametrize("requested_topic, returned_fields, fallback", [
    (None, {}, False),
    ("67890", {}, False),
    (None, {"message_thread_id": None}, False),
    ("67890", {"message_thread_id": None}, False),
    ("67890", {"message_thread_id": None, "direct_messages_topic": None}, False),
    ("67890", {"direct_messages_topic": SimpleNamespace(topic_id=None)}, False),
    ("67890", {"message_thread_id": None}, True),
])
def test_missing_server_topic_is_not_invented(monkeypatch, requested_topic, returned_fields, fallback):
    message = SimpleNamespace(message_id=42, **returned_fields)
    transport = AsyncMock(return_value=message)
    if fallback:
        transport.side_effect = [Exception("Bad Request: message thread not found"), message]
    bot = SimpleNamespace(send_message=transport)
    monkeypatch.setattr(senders, "_telegram_bot", lambda _token: bot)
    monkeypatch.setattr(senders, "_telegram_format", lambda text: (text, None, False))

    receipt = asyncio.run(senders._send_telegram(
        "fake-token", "12345", "synthetic delivery", thread_id=requested_topic,
    ))

    assert receipt["success"] is True
    assert receipt["message_id"] == "42"
    assert "thread_id" not in receipt
    assert "message_thread_id" not in receipt
    if fallback:
        assert transport.await_count == 2
        assert transport.await_args_list[0].kwargs["message_thread_id"] == 67890
        assert "message_thread_id" not in transport.await_args.kwargs
    else:
        transport.assert_awaited_once()
