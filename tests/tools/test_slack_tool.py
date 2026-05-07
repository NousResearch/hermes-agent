"""Tests for the Slack read/search tool."""

import json
from unittest.mock import MagicMock

import pytest

from tools import slack_tool
from tools.slack_tool import (
    _parse_since,
    check_slack_tool_requirements,
    slack_handler,
)


class FakeSlackClient:
    def __init__(self, token="xoxb-test"):
        self.token = token
        self.history_calls = []

    def users_conversations(self, **kwargs):
        return {
            "ok": True,
            "channels": [
                {"id": "C123456789", "name": "sports_product", "is_private": False, "is_member": True},
                {"id": "G123456789", "name": "leadership", "is_private": True, "is_member": True},
            ],
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_info(self, channel):
        if channel == "CNOTMEMBER1":
            return {"ok": True, "channel": {"id": channel, "is_channel": True, "is_member": False}}
        return {"ok": True, "channel": {"id": channel, "is_channel": True, "is_member": True}}

    def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        channel = kwargs["channel"]
        messages = [
            {"ts": "1710000000.000001", "user": "U1", "text": f"Shipping update in {channel}"},
            {"ts": "1710000001.000001", "user": "U2", "text": "No blocker today", "reply_count": 2},
        ]
        return {"ok": True, "messages": messages, "response_metadata": {"next_cursor": ""}}

    def conversations_replies(self, **kwargs):
        return {
            "ok": True,
            "messages": [
                {"ts": kwargs["ts"], "user": "U2", "text": "No blocker today"},
                {"ts": "1710000002.000001", "user": "U1", "text": "Follow-up decision"},
            ],
        }

    def users_info(self, user):
        names = {"U1": "alice", "U2": "bob"}
        return {"ok": True, "user": {"name": names.get(user, user), "profile": {}}}


@pytest.fixture(autouse=True)
def slack_env(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(slack_tool, "SLACK_SDK_AVAILABLE", True)
    monkeypatch.setattr(slack_tool, "WebClient", FakeSlackClient)
    monkeypatch.setattr(slack_tool, "_directory_channels", lambda: [])
    monkeypatch.setattr(slack_tool, "_directory_lookup", lambda name: None)
    slack_tool._USER_CACHE.clear()


def test_check_requirements_requires_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    assert check_slack_tool_requirements() is False
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    assert check_slack_tool_requirements() is True


def test_list_channels_returns_joined_channels():
    result = json.loads(slack_handler(action="list_channels"))
    assert result["ok"] is True
    assert result["count"] == 2
    assert result["channels"][0]["name"] == "sports_product"
    assert result["channels"][1]["type"] == "private"


def test_read_channel_resolves_name_and_reads_history():
    result = json.loads(slack_handler(action="read_channel", channel="#sports_product", limit=10, since="7d"))
    assert result["ok"] is True
    assert result["channel_id"] == "C123456789"
    assert result["count"] == 2
    assert result["messages"][0]["user_name"] == "alice"
    assert "Shipping update" in result["messages"][0]["text"]
    assert result["since"] is not None


def test_read_channel_rejects_not_member():
    result = json.loads(slack_handler(action="read_channel", channel="CNOTMEMBER1"))
    assert result["ok"] is False
    assert "not a member" in result["error"]


def test_read_thread_returns_replies():
    result = json.loads(slack_handler(action="read_thread", channel="C123456789", thread_ts="1710000001.000001"))
    assert result["ok"] is True
    assert result["count"] == 2
    assert result["messages"][1]["text"] == "Follow-up decision"


def test_search_messages_scans_joined_channels():
    result = json.loads(slack_handler(action="search_messages", query="blocker", limit=5, max_channels=2))
    assert result["ok"] is True
    assert result["count"] == 2
    assert {m["channel_id"] for m in result["matches"]} == {"C123456789", "G123456789"}
    assert result["scope"].startswith("history scan")


def test_parse_since_accepts_relative_date_and_bad_input():
    assert _parse_since("24h") is not None
    assert _parse_since("2026-05-07") is not None
    assert _parse_since("1710000000.000001") == "1710000000.000001"
    assert _parse_since("last tuesday-ish") is None
