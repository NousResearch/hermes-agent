from __future__ import annotations

import json
from unittest.mock import patch

from tools.slack_api_tool import check_slack_api_requirements, slack_api_tool
from toolsets import resolve_toolset


def test_check_slack_api_requirements_requires_bot_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    assert check_slack_api_requirements() is False
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    assert check_slack_api_requirements() is True


def test_hermes_slack_toolset_includes_slack_api():
    assert "slack_api" in resolve_toolset("hermes-slack")
    assert "slack_api" in resolve_toolset("slack")


def test_replies_requires_valid_channel(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    payload = json.loads(slack_api_tool(action="replies", channel="bad", thread_ts="1783504790.412029"))
    assert payload["ok"] is False
    assert "channel must be" in payload["error"]


def test_history_returns_sanitized_messages(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    def fake_post_form(method, params):
        assert method == "conversations.history"
        assert params["channel"] == "C0BFGRSM3K8"
        return {
            "ok": True,
            "messages": [
                {
                    "type": "message",
                    "user": "U123",
                    "ts": "1783504790.412029",
                    "text": "parent",
                    "reply_count": 2,
                    "files": [
                        {
                            "id": "F1",
                            "name": "doc.pdf",
                            "title": "Doc",
                            "mimetype": "application/pdf",
                            "size": 123,
                            "url_private": "https://files.slack.com/files-pri/T/F/doc.pdf",
                        }
                    ],
                }
            ],
        }

    with patch("tools.slack_api_tool._post_form", side_effect=fake_post_form):
        payload = json.loads(slack_api_tool(action="history", channel="C0BFGRSM3K8"))

    assert payload["ok"] is True
    assert payload["messages"][0]["reply_count"] == 2
    assert payload["messages"][0]["files"][0]["id"] == "F1"


def test_send_posts_thread_reply(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    def fake_post_json(method, payload):
        assert method == "chat.postMessage"
        assert payload == {
            "channel": "C0BFGRSM3K8",
            "thread_ts": "1783504790.412029",
            "text": "hello",
        }
        return {"ok": True, "channel": "C0BFGRSM3K8", "ts": "1783504791.000000"}

    with patch("tools.slack_api_tool._post_json", side_effect=fake_post_json):
        payload = json.loads(
            slack_api_tool(
                action="send",
                channel="C0BFGRSM3K8",
                thread_ts="1783504790.412029",
                text="hello",
            )
        )

    assert payload == {
        "ok": True,
        "channel": "C0BFGRSM3K8",
        "ts": "1783504791.000000",
        "thread_ts": "1783504790.412029",
        "error": None,
    }
