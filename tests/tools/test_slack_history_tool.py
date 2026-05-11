"""Tests for tools/slack_history_tool.py."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools.slack_history_tool import (
    _sanitize_message_text,
    _safe_actor_label,
    slack_history_tool,
)


class FakeSlackClient:
    def __init__(self, *, history_messages=None, thread_messages=None):
        self.history_messages = history_messages or []
        self.thread_messages = thread_messages or []
        self.history_calls = []
        self.thread_calls = []

    async def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        return {
            "ok": True,
            "messages": self.history_messages,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }

    async def conversations_replies(self, **kwargs):
        self.thread_calls.append(kwargs)
        return {
            "ok": True,
            "messages": self.thread_messages,
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }


@pytest.mark.asyncio
async def test_history_reads_channel_and_returns_redacted_messages(monkeypatch):
    client = FakeSlackClient(
        history_messages=[
            {
                "ts": "1778492155.804399",
                "user": "U123ALICE",
                "text": "Cloudflare 500 on https://example.com?token=secret-token email alex@example.com phone +33612345678",
            },
            {
                "ts": "1778492156.000000",
                "bot_id": "B999",
                "username": "Sentry",
                "subtype": "bot_message",
                "text": "TypeError: Cannot read properties of undefined",
            },
        ]
    )
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    with patch("tools.slack_history_tool._make_slack_client", return_value=client), \
         patch("gateway.channel_directory.resolve_channel_name", return_value="C0ALERTS"):
        result = json.loads(
            await slack_history_tool({
                "action": "history",
                "channel": "#alerts",
                "limit": 25,
            })
        )

    assert result["success"] is True
    assert result["channel"] == "C0ALERTS"
    assert result["count"] == 2
    assert client.history_calls == [{"channel": "C0ALERTS", "limit": 25}]
    assert result["messages"][0]["actor"].startswith("user:")
    assert result["messages"][1]["actor"].startswith("bot:")
    joined = json.dumps(result, ensure_ascii=False)
    assert "alex@example.com" not in joined
    assert "+33612345678" not in joined
    assert "secret-token" not in joined
    assert "[email]" in joined
    assert "[phone]" in joined
    assert "token=***" in joined


@pytest.mark.asyncio
async def test_thread_reads_conversations_replies_with_thread_ts(monkeypatch):
    client = FakeSlackClient(
        thread_messages=[
            {"ts": "1000.000001", "user": "U1", "text": "root alert"},
            {"ts": "1000.000002", "user": "U2", "text": "fix candidate"},
        ]
    )
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    with patch("tools.slack_history_tool._make_slack_client", return_value=client):
        result = json.loads(
            await slack_history_tool({
                "action": "thread",
                "channel": "C0ALERTS",
                "thread_ts": "1000.000001",
                "limit": 10,
                "oldest": "999.999999",
            })
        )

    assert result["success"] is True
    assert result["count"] == 2
    assert client.thread_calls == [
        {
            "channel": "C0ALERTS",
            "ts": "1000.000001",
            "limit": 10,
            "oldest": "999.999999",
        }
    ]


@pytest.mark.parametrize(
    "text,expected_absent,expected_present",
    [
        (
            "email ops@example.com and phone 06 12 34 56 78",
            ["ops@example.com", "06 12 34 56 78"],
            ["[email]", "[phone]"],
        ),
        (
            "https://x.test/path?signature=abc123&ok=1 token=shh",
            ["abc123", "shh"],
            ["signature=***", "token=***"],
        ),
    ],
)
def test_sanitize_message_text_redacts_pii_and_secrets(text, expected_absent, expected_present):
    sanitized = _sanitize_message_text(text)

    for value in expected_absent:
        assert value not in sanitized
    for value in expected_present:
        assert value in sanitized


def test_sanitize_message_text_truncates_long_transcripts():
    sanitized = _sanitize_message_text("x" * 250, max_chars=200)

    assert len(sanitized) <= 200
    assert sanitized.endswith("…")


def test_safe_actor_label_never_returns_raw_slack_user_ids():
    user_label = _safe_actor_label({"user": "U123456789"})
    bot_label = _safe_actor_label({"bot_id": "B123456789", "username": "Alert Bot"})

    assert user_label.startswith("user:")
    assert bot_label.startswith("bot:")
    assert "U123456789" not in user_label
    assert "Alert Bot" not in bot_label


@pytest.mark.asyncio
async def test_missing_thread_ts_is_error(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    result = json.loads(
        await slack_history_tool({"action": "thread", "channel": "C0ALERTS"})
    )

    assert result["error"] == "thread_ts is required when action='thread'"


@pytest.mark.asyncio
async def test_unknown_channel_name_returns_actionable_error(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        result = json.loads(
            await slack_history_tool({"action": "history", "channel": "#missing"})
        )

    assert result["error"].startswith("Could not resolve Slack channel")
