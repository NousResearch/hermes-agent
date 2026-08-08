import asyncio
import json
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    get_session_transport_adapter,
    set_session_vars,
)
from tools import slack_tool


CHANNEL_ID = "C12345678"
THREAD_TS = "1712345678.000100"
TEAM_ID = "T12345678"
OTHER_TEAM_ID = "T87654321"
BOT_ID = "B12345678"


@pytest.fixture(autouse=True)
def active_slack_session(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    tokens = set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        thread_id="",
        profile="default",
        scope_id=TEAM_ID,
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _reader(monkeypatch, payload=None):
    calls: list[dict[str, Any]] = []

    def fake(channel_id, **kwargs):
        calls.append({"channel_id": channel_id, **kwargs})
        return payload or {
            "ok": True,
            "messages": [],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(slack_tool, "_read_from_live_adapter", fake)
    return calls


def test_defaults_to_active_channel(monkeypatch):
    calls = _reader(
        monkeypatch,
        {
            "ok": True,
            "messages": [
                {
                    "ts": "1712345679.000200",
                    "user": "U12345678",
                    "text": "see <https://example.com/old|old link>",
                }
            ],
            "has_more": False,
            "response_metadata": {"next_cursor": "next-page"},
        },
    )

    result = json.loads(slack_tool.slack_history(limit=10))

    assert result["success"] is True
    assert result["channel"] == CHANNEL_ID
    assert result["content_is_untrusted"] is True
    assert "https://example.com/old" in result["messages"][0]["text"]
    assert result["next_cursor"] == "next-page"
    assert calls == [
        {
            "channel_id": CHANNEL_ID,
            "scope_id": TEAM_ID,
            "thread_ts": "",
            "limit": 10,
            "before": "",
            "after": "",
            "cursor": "",
        }
    ]


def test_defaults_to_active_thread(monkeypatch):
    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        thread_id=THREAD_TS,
        profile="default",
        scope_id=TEAM_ID,
    )
    calls = _reader(monkeypatch)

    result = json.loads(slack_tool.slack_history())

    assert result["success"] is True
    assert result["thread_ts"] == THREAD_TS
    assert calls[0]["thread_ts"] == THREAD_TS


def test_active_thread_cannot_switch_to_another_thread(monkeypatch):
    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        thread_id=THREAD_TS,
        profile="default",
        scope_id=TEAM_ID,
    )
    monkeypatch.setattr(
        slack_tool,
        "_read_from_live_adapter",
        lambda *_args, **_kwargs: pytest.fail("thread override reached Slack"),
    )

    result = json.loads(slack_tool.slack_history(thread_ts="1712345678.999999"))

    assert result["success"] is False
    assert "cannot switch away from the active thread" in result["error"]


def test_channel_context_cannot_select_a_thread(monkeypatch):
    monkeypatch.setattr(
        slack_tool,
        "_read_from_live_adapter",
        lambda *_args, **_kwargs: pytest.fail("unrelated thread reached Slack"),
    )

    result = json.loads(slack_tool.slack_history(thread_ts=THREAD_TS))

    assert result["success"] is False
    assert "cannot select a thread" in result["error"]


def test_other_channel_is_rejected_before_io(monkeypatch):
    monkeypatch.setattr(
        slack_tool,
        "_read_from_live_adapter",
        lambda *_args, **_kwargs: pytest.fail("blocked target reached Slack"),
    )

    result = json.loads(slack_tool.slack_history(channel="C99999999"))

    assert result["success"] is False
    assert "active conversation" in result["error"]


def test_non_slack_context_is_rejected_before_io(monkeypatch):
    clear_session_vars([])
    set_session_vars(platform="telegram", chat_id=CHANNEL_ID)
    monkeypatch.setattr(
        slack_tool,
        "_read_from_live_adapter",
        lambda *_args, **_kwargs: pytest.fail("non-Slack turn reached Slack"),
    )

    result = json.loads(slack_tool.slack_history())

    assert result["success"] is False
    assert "active Slack conversation" in result["error"]


def test_bounds_and_pagination_are_forwarded(monkeypatch):
    calls = _reader(monkeypatch)

    result = json.loads(
        slack_tool.slack_history(
            limit=999,
            before="1712345680.000300",
            after="1712345600.000000",
            cursor="cursor-2",
        )
    )

    assert result["success"] is True
    assert calls[0] == {
        "channel_id": CHANNEL_ID,
        "scope_id": TEAM_ID,
        "thread_ts": "",
        "limit": 50,
        "before": "1712345680.000300",
        "after": "1712345600.000000",
        "cursor": "cursor-2",
    }


def test_invalid_timestamp_is_rejected_before_io(monkeypatch):
    monkeypatch.setattr(
        slack_tool,
        "_read_from_live_adapter",
        lambda *_args, **_kwargs: pytest.fail("invalid input reached Slack"),
    )

    result = json.loads(slack_tool.slack_history(before="yesterday"))

    assert result["success"] is False
    assert "before must be a Slack timestamp" in result["error"]


def test_result_is_bounded(monkeypatch):
    payload = {
        "ok": True,
        "messages": [
            {
                "ts": f"17123456{index:02d}.000100",
                "user": "U12345678",
                "text": "x" * 2_000,
            }
            for index in range(50)
        ],
        "has_more": True,
        "response_metadata": {"next_cursor": "more"},
    }
    _reader(monkeypatch, payload)

    raw = slack_tool.slack_history(limit=50)
    result = json.loads(raw)

    assert len(raw) <= slack_tool._MAX_RESULT_CHARS
    assert result["result_truncated"] is True
    assert result["count"] == 50
    assert len(result["messages"]) == 50
    assert all(message["text_truncated"] for message in result["messages"])
    assert result["has_more"] is True
    assert result["next_cursor"] == "more"


def test_live_reader_uses_transport_adapter_and_forwards_thread_bounds(monkeypatch):
    from gateway.config import Platform
    import gateway.run as gateway_run

    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        async def conversations_replies(self, **kwargs):
            calls.append(("replies", kwargs))
            return {
                "ok": True,
                "messages": [],
                "response_metadata": {"next_cursor": ""},
            }

    class Adapter:
        def __init__(self):
            self._team_clients = {TEAM_ID: Client()}
            self._team_bot_ids = {TEAM_ID: BOT_ID}

    class Loop:
        def is_running(self):
            return True

    adapter = Adapter()
    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        profile="secondary",
        scope_id=TEAM_ID,
        transport_adapter=adapter,
    )
    runner = SimpleNamespace(
        adapters={Platform.SLACK: adapter},
        _profile_adapters={"secondary": {Platform.SLACK: object()}},
        _gateway_loop=Loop(),
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    def run_now(coro, _loop, **_kwargs):
        future: Future = Future()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(slack_tool, "safe_schedule_threadsafe", run_now)

    result = slack_tool._read_from_live_adapter(
        CHANNEL_ID,
        scope_id=TEAM_ID,
        thread_ts=THREAD_TS,
        limit=20,
        before="1712345680.000300",
        after="1712345600.000000",
        cursor="",
    )

    assert result["ok"] is True
    assert calls == [
        (
            "replies",
            {
                "channel": CHANNEL_ID,
                "ts": THREAD_TS,
                "limit": 20,
                "latest": "1712345680.000300",
                "oldest": "1712345600.000000",
                "cursor": None,
            },
        ),
    ]


def test_live_reader_forwards_channel_bounds_and_cursor(monkeypatch):
    from gateway.config import Platform
    import gateway.run as gateway_run

    calls: list[dict[str, Any]] = []

    class Client:
        async def conversations_history(self, **kwargs):
            calls.append(kwargs)
            return {
                "ok": True,
                "messages": [],
                "response_metadata": {"next_cursor": "next"},
            }

    class Adapter:
        def __init__(self):
            self._team_clients = {TEAM_ID: Client()}
            self._team_bot_ids = {TEAM_ID: BOT_ID}

    class Loop:
        def is_running(self):
            return True

    adapter = Adapter()
    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        scope_id=TEAM_ID,
        transport_adapter=adapter,
    )
    runner = SimpleNamespace(
        adapters={Platform.SLACK: adapter},
        _profile_adapters={},
        _gateway_loop=Loop(),
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    def run_now(coro, _loop, **_kwargs):
        future: Future = Future()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(slack_tool, "safe_schedule_threadsafe", run_now)

    result = slack_tool._read_from_live_adapter(
        CHANNEL_ID,
        scope_id=TEAM_ID,
        thread_ts="",
        limit=25,
        before="1712345680.000300",
        after="1712345600.000000",
        cursor="cursor-2",
    )

    assert result["ok"] is True
    assert calls == [
        {
            "channel": CHANNEL_ID,
            "limit": 25,
            "latest": "1712345680.000300",
            "oldest": "1712345600.000000",
            "cursor": "cursor-2",
        }
    ]


def test_live_reader_rejects_unknown_workspace_before_slack_io(monkeypatch):
    from gateway.config import Platform
    import gateway.run as gateway_run

    class Client:
        async def conversations_history(self, **_kwargs):
            pytest.fail("unknown workspace reached Slack")

    class Adapter:
        def __init__(self):
            self._team_clients = {TEAM_ID: Client()}
            self._team_bot_ids = {TEAM_ID: BOT_ID}

    class Loop:
        def is_running(self):
            return True

    adapter = Adapter()
    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        scope_id=OTHER_TEAM_ID,
        transport_adapter=adapter,
    )
    runner = SimpleNamespace(
        adapters={Platform.SLACK: adapter},
        _profile_adapters={},
        _gateway_loop=Loop(),
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    def run_now(coro, _loop, **_kwargs):
        future: Future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(slack_tool, "safe_schedule_threadsafe", run_now)

    with pytest.raises(slack_tool.SlackHistoryError, match="workspace is unavailable"):
        slack_tool._read_from_live_adapter(
            CHANNEL_ID,
            scope_id=OTHER_TEAM_ID,
            thread_ts="",
            limit=20,
            before="",
            after="",
            cursor="",
        )


def test_live_reader_rejects_user_token_principal_before_slack_io(monkeypatch):
    from gateway.config import Platform
    import gateway.run as gateway_run

    class Client:
        async def conversations_history(self, **_kwargs):
            pytest.fail("user-token principal reached Slack")

    class Adapter:
        def __init__(self):
            self._team_clients = {TEAM_ID: Client()}
            self._team_bot_ids = {}

    class Loop:
        def is_running(self):
            return True

    adapter = Adapter()
    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        scope_id=TEAM_ID,
        transport_adapter=adapter,
    )
    runner = SimpleNamespace(
        adapters={Platform.SLACK: adapter},
        _profile_adapters={},
        _gateway_loop=Loop(),
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    def run_now(coro, _loop, **_kwargs):
        future: Future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(slack_tool, "safe_schedule_threadsafe", run_now)

    with pytest.raises(slack_tool.SlackHistoryError, match="bot is unavailable"):
        slack_tool._read_from_live_adapter(
            CHANNEL_ID,
            scope_id=TEAM_ID,
            thread_ts="",
            limit=20,
            before="",
            after="",
            cursor="",
        )


def test_selected_secondary_profile_never_falls_back_to_primary_adapter(monkeypatch):
    from gateway.config import Platform
    import gateway.run as gateway_run

    clear_session_vars([])
    set_session_vars(
        platform="slack",
        chat_id=CHANNEL_ID,
        profile="secondary",
        scope_id=TEAM_ID,
    )

    class Loop:
        def is_running(self):
            return True

    runner = SimpleNamespace(
        adapters={Platform.SLACK: object()},
        _profile_adapters={},
        _gateway_loop=Loop(),
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    with pytest.raises(slack_tool.SlackHistoryError, match="unavailable"):
        slack_tool._live_adapter_and_loop()


def test_gateway_propagates_slack_workspace_scope_to_tools():
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = object()
    cast(Any, runner).adapters = {Platform.SLACK: adapter}
    cast(Any, runner)._profile_adapters = {}
    context = SimpleNamespace(
        source=SimpleNamespace(
            platform=Platform.SLACK,
            chat_id=CHANNEL_ID,
            chat_type="channel",
            chat_name="general",
            thread_id=THREAD_TS,
            scope_id=TEAM_ID,
            user_id="U12345678",
            user_name="user",
            message_id="1712345680.000300",
            profile="default",
            _transport_adapter_ref=lambda: adapter,
        ),
        session_key="slack:default:T12345678:C12345678",
    )

    tokens = runner._set_session_env(cast(Any, context))
    try:
        assert get_session_env("HERMES_SESSION_SCOPE_ID") == TEAM_ID
        assert get_session_transport_adapter() is adapter
    finally:
        runner._clear_session_env(tokens)


def test_tool_is_registered_in_slack_platform_bundle_without_global_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    import model_tools
    from toolsets import resolve_toolset

    model_tools._clear_tool_defs_cache()
    assert "slack_history" in resolve_toolset("slack")
    assert "slack_history" in resolve_toolset("hermes-slack")

    schema = next(
        tool
        for tool in model_tools.get_tool_definitions(
            enabled_toolsets=["hermes-slack"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        if tool["function"]["name"] == "slack_history"
    )
    properties = schema["function"]["parameters"]["properties"]
    assert set(properties) == {
        "channel",
        "thread_ts",
        "limit",
        "before",
        "after",
        "cursor",
    }
