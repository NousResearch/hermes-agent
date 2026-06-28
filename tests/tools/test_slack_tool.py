import json
import os
from pathlib import Path
from typing import Any

from tools import slack_tool


CHANNEL_ID = "C123456789"


def test_slack_find_messages_returns_newest_x_links(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        assert method == "GET"
        assert token == "xoxb-test"
        assert endpoint == "conversations.history"
        assert params is not None
        assert params["channel"] == CHANNEL_ID
        return {
            "ok": True,
            "messages": [
                {
                    "ts": "222.0002",
                    "user": "U2",
                    "text": "processed <https://x.com/example/status/222|x post>",
                },
                {
                    "ts": "111.0001",
                    "user": "U1",
                    "text": "plain message without a link",
                },
            ],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="find_messages", channel=CHANNEL_ID))

    assert result["channel"] == CHANNEL_ID
    assert result["count"] == 1
    assert result["matches"][0]["ts"] == "222.0002"
    assert result["matches"][0]["urls"] == ["https://x.com/example/status/222"]


def test_slack_fetch_history_accepts_channel_name(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    calls = []

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        calls.append((endpoint, dict(params or {})))
        if endpoint == "conversations.list":
            return {
                "ok": True,
                "channels": [{"id": CHANNEL_ID, "name": "quotes", "is_member": True}],
                "response_metadata": {"next_cursor": ""},
            }
        if endpoint == "conversations.history":
            assert params is not None
            assert params["channel"] == CHANNEL_ID
            return {
                "ok": True,
                "messages": [{"ts": "333.0003", "user": "U3", "text": "hello"}],
                "has_more": False,
                "response_metadata": {"next_cursor": ""},
            }
        raise AssertionError(endpoint)

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="fetch_history", channel="#quotes"))

    assert [c[0] for c in calls] == ["conversations.list", "conversations.history"]
    assert result["messages"][0]["text"] == "hello"


def test_slack_tool_is_read_only(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    result = json.loads(slack_tool.slack(action="delete_message", channel=CHANNEL_ID))

    assert result["error"] == "Unknown action: delete_message"
    assert result["available_actions"] == ["list_channels", "fetch_history", "find_messages"]


def test_slack_find_messages_caps_returned_matches(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        assert endpoint == "conversations.history"
        return {
            "ok": True,
            "messages": [
                {"ts": f"{idx}.0000", "user": "U", "text": f"https://x.com/example/status/{idx}"}
                for idx in range(250)
            ],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="find_messages", channel=CHANNEL_ID, limit=10_000))

    assert result["count"] == 200
    assert result["matches"][199]["ts"] == "199.0000"


def test_slack_history_invalid_limit_uses_safe_default(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    seen_params = []

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        seen_params.append(dict(params or {}))
        return {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    bad_limit: Any = "not-a-number"
    result = json.loads(slack_tool.slack(action="fetch_history", channel=CHANNEL_ID, limit=bad_limit))

    assert result["count"] == 0
    assert seen_params[0]["limit"] == 50


def test_slack_history_string_false_stays_false(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    seen_params = []

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        seen_params.append(dict(params or {}))
        return {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    inclusive: Any = "false"
    result = json.loads(slack_tool.slack(action="fetch_history", channel=CHANNEL_ID, inclusive=inclusive))

    assert result["count"] == 0
    assert seen_params[0]["inclusive"] == "false"


def test_slack_uses_gateway_config_token_when_env_missing(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.setattr(slack_tool, "_configured_bot_tokens", lambda: ["xoxb-config"])
    seen_tokens = []

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        seen_tokens.append(token)
        return {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="fetch_history", channel=CHANNEL_ID))

    assert result["count"] == 0
    assert seen_tokens == ["xoxb-config"]


def test_slack_multi_token_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-one,xoxb-two")

    result = json.loads(slack_tool.slack(action="fetch_history", channel=CHANNEL_ID))

    assert result["error"] == "Slack history tool requires exactly one bot token; multi-workspace token selection is not supported yet."
    assert not slack_tool.check_slack_tool_requirements()


def test_slack_message_text_is_truncated_but_urls_are_preserved(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    long_prefix = "x" * 2500

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        return {
            "ok": True,
            "messages": [{"ts": "444.0004", "user": "U4", "text": f"{long_prefix} https://x.com/example/status/444"}],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="fetch_history", channel=CHANNEL_ID))
    message = result["messages"][0]

    assert message["text_truncated"] is True
    assert len(message["text"]) == 2000
    assert message["urls"] == ["https://x.com/example/status/444"]


def test_slack_find_messages_query_uses_raw_text_after_truncation(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    long_prefix = "x" * 2500

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        return {
            "ok": True,
            "messages": [
                {
                    "ts": "445.0005",
                    "user": "U5",
                    "text": f"{long_prefix} needle https://x.com/example/status/445",
                }
            ],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="find_messages", channel=CHANNEL_ID, query="needle"))

    assert result["count"] == 1
    assert result["matches"][0]["ts"] == "445.0005"
    assert result["matches"][0]["text_truncated"] is True


def test_slack_find_messages_supports_custom_link_domains(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        return {
            "ok": True,
            "messages": [{"ts": "446.0006", "user": "U6", "text": "see https://example.com/path"}],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(
        slack_tool.slack(action="find_messages", channel=CHANNEL_ID, link_domains="example.com")
    )

    assert result["count"] == 1
    assert result["matches"][0]["urls"] == ["https://example.com/path"]


def test_slack_fetch_history_blocks_non_allowed_channel(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", CHANNEL_ID)

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        raise AssertionError("Slack API should not be called for blocked channels")

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="fetch_history", channel="C999999999"))

    assert result == {
        "error": "Slack channel is not in configured allowed_channels.",
        "channel": "C999999999",
    }


def test_slack_fetch_history_blocks_config_only_allowed_channel_on_first_call(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    hermes_home = Path(os.environ["HERMES_HOME"])
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  slack:\n"
        "    allowed_channels:\n"
        f"      - {CHANNEL_ID}\n",
        encoding="utf-8",
    )

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        raise AssertionError("Slack API should not be called for blocked channels")

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="fetch_history", channel="C999999999"))

    assert result == {
        "error": "Slack channel is not in configured allowed_channels.",
        "channel": "C999999999",
    }
    assert os.environ["SLACK_ALLOWED_CHANNELS"] == CHANNEL_ID


def test_slack_find_messages_blocks_non_allowed_channel(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", CHANNEL_ID)

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        raise AssertionError("Slack API should not be called for blocked channels")

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="find_messages", channel="C999999999"))

    assert result["error"] == "Slack channel is not in configured allowed_channels."
    assert result["channel"] == "C999999999"


def test_slack_allowed_channels_do_not_block_dms(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", CHANNEL_ID)
    seen_channels = []

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        assert params is not None
        seen_channels.append(params["channel"])
        return {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="fetch_history", channel="D123456789"))

    assert result["count"] == 0
    assert seen_channels == ["D123456789"]


def test_slack_list_channels_filters_allowed_channels(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", CHANNEL_ID)

    def fake_request(method, endpoint, token, *, params=None, body=None, timeout=20):
        assert endpoint == "conversations.list"
        return {
            "ok": True,
            "channels": [
                {"id": CHANNEL_ID, "name": "allowed", "is_member": True},
                {"id": "C999999999", "name": "blocked", "is_member": True},
            ],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_request", fake_request)

    result = json.loads(slack_tool.slack(action="list_channels"))

    assert result["allowed_channels_applied"] is True
    assert result["count"] == 1
    assert result["channels"][0]["id"] == CHANNEL_ID


def test_slack_tool_is_in_slack_platform_bundle(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    from toolsets import resolve_toolset

    assert "slack" in resolve_toolset("slack")
    assert "slack" in resolve_toolset("hermes-slack")


def test_slack_schema_exposes_action_parameters(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    import model_tools

    schema = next(
        tool for tool in model_tools.get_tool_definitions(enabled_toolsets=["hermes-slack"], quiet_mode=True)
        if tool["function"]["name"] == "slack"
    )
    properties = schema["function"]["parameters"]["properties"]

    assert properties["action"]["enum"] == ["list_channels", "fetch_history", "find_messages"]
    assert "delete_message" not in properties["action"]["enum"]
