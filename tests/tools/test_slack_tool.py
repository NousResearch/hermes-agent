import pytest

from tools import slack_tool


class FakeSlackClient:
    def __init__(self, *, list_response=None, history_response=None, replies_response=None):
        self.list_response = list_response or {"ok": True, "channels": []}
        self.history_response = history_response or {"ok": True, "messages": []}
        self.replies_response = replies_response or {"ok": True, "messages": []}
        self.calls = []

    async def conversations_list(self, **kwargs):
        self.calls.append(("conversations_list", kwargs))
        return self.list_response

    async def conversations_history(self, **kwargs):
        self.calls.append(("conversations_history", kwargs))
        return self.history_response

    async def conversations_replies(self, **kwargs):
        self.calls.append(("conversations_replies", kwargs))
        return self.replies_response


class FakeSlackApiError(Exception):
    def __init__(self, response):
        super().__init__("slack api error")
        self.response = response


class RaisingSlackClient:
    async def conversations_list(self, **_kwargs):
        raise FakeSlackApiError(
            {
                "ok": False,
                "error": "missing_scope",
                "needed": "groups:read",
                "provided": "channels:read",
            }
        )


@pytest.mark.asyncio
async def test_list_channels_reports_missing_scope_with_fix(monkeypatch):
    client = FakeSlackClient(
        list_response={
            "ok": False,
            "error": "missing_scope",
            "needed": "groups:read",
            "provided": "channels:read",
        },
    )
    monkeypatch.setattr(slack_tool, "_get_slack_client", lambda: client)

    result = await slack_tool.slack_list_channels()

    assert result["ok"] is False
    assert result["error"] == "missing_scope"
    assert result["needed"] == "groups:read"
    assert "reinstall" in result["fix"]


@pytest.mark.asyncio
async def test_list_channels_handles_slack_sdk_raised_missing_scope(monkeypatch):
    monkeypatch.setattr(slack_tool, "_get_slack_client", lambda: RaisingSlackClient())

    result = await slack_tool.slack_list_channels()

    assert result["ok"] is False
    assert result["error"] == "missing_scope"
    assert result["needed"] == "groups:read"
    assert "reinstall" in result["fix"]


@pytest.mark.asyncio
async def test_get_messages_resolves_channel_name_and_sorts_oldest_first(monkeypatch):
    client = FakeSlackClient(
        list_response={
            "ok": True,
            "channels": [{"id": "C123", "name": "vucar-staging-aws"}],
        },
        history_response={
            "ok": True,
            "messages": [
                {"ts": "1710000002.000000", "user": "U2", "text": "second"},
                {"ts": "1710000001.000000", "user": "U1", "text": "first"},
            ],
        },
    )
    monkeypatch.setattr(slack_tool, "_get_slack_client", lambda: client)

    result = await slack_tool.slack_get_messages("#vucar-staging-aws")

    assert result["ok"] is True
    assert result["channel"] == "C123"
    assert [m["text"] for m in result["messages"]] == ["first", "second"]
    assert client.calls[0][0] == "conversations_list"
    assert client.calls[1] == ("conversations_history", {"channel": "C123", "limit": 30})


@pytest.mark.asyncio
async def test_get_thread_fetches_replies_by_channel_id(monkeypatch):
    client = FakeSlackClient(
        replies_response={
            "ok": True,
            "messages": [
                {"ts": "1710000001.000000", "user": "U1", "text": "root"},
                {"ts": "1710000002.000000", "user": "U2", "text": "reply"},
            ],
        },
    )
    monkeypatch.setattr(slack_tool, "_get_slack_client", lambda: client)

    result = await slack_tool.slack_get_thread("C123", "1710000001.000000")

    assert result["ok"] is True
    assert result["channel"] == "C123"
    assert result["thread_ts"] == "1710000001.000000"
    assert [m["text"] for m in result["messages"]] == ["root", "reply"]
    assert client.calls == [
        (
            "conversations_replies",
            {"channel": "C123", "ts": "1710000001.000000", "limit": 50},
        )
    ]


@pytest.mark.asyncio
async def test_get_thread_requires_thread_ts(monkeypatch):
    monkeypatch.setattr(slack_tool, "_get_slack_client", lambda: FakeSlackClient())

    result = await slack_tool.slack_get_thread("C123", "")

    assert result == {"ok": False, "error": "thread_ts is required"}


def test_hermes_slack_toolset_exposes_slack_context_tools(monkeypatch):
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(slack_tool, "_get_slack_client", lambda: object())
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()

    names = {
        item["function"]["name"]
        for item in get_tool_definitions(enabled_toolsets=["hermes-slack"], quiet_mode=True)
    }

    assert {"slack_list_channels", "slack_get_messages", "slack_get_thread"} <= names
