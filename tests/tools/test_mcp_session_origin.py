"""Opt-in ``session_origin`` injection for MCP tools."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.mcp_tool_handlers as handlers
from tools.mcp_tool_session_origin import parse_session_origin_config

_CHAT = {
    "HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "-1001", "HERMES_SESSION_CHAT_TYPE": "group",
    "HERMES_SESSION_THREAD_ID": "42", "HERMES_SESSION_USER_ID": "u1", "HERMES_SESSION_ID": "session-1",
    "HERMES_SESSION_KEY": "agent:main:telegram:group:-1001:42",
}
_CONFIG = {"session_origin": {"tools": ["watch_chat", "send_and_watch"]}}


def _server(config=_CONFIG):
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=SimpleNamespace(isError=False, content=[], structuredContent=None))
    # No ``_watch_stdio_children``: the call races no child watcher, so the plain await path runs.
    return SimpleNamespace(session=session, _config=config)


def _call(server, tool_name, args, env=_CHAT):
    with patch("gateway.session_context.get_session_env", side_effect=lambda k, d="": env.get(k, d)):
        asyncio.run(handlers._call_tool_racing_stdio_death(server, "srv", tool_name, args))
    return server.session.call_tool.await_args.kwargs["arguments"]


def test_listed_tool_receives_origin_and_empty_fields_are_dropped():
    sent = _call(_server(), "watch_chat", {"chatId": "1@c.us"})
    assert sent["_hermesOrigin"] == {
        "platform": "telegram", "chatId": "-1001", "chatType": "group", "threadId": "42", "userId": "u1",
        "sessionId": "session-1", "sessionKey": "agent:main:telegram:group:-1001:42"}


def test_unlisted_tool_receives_nothing():
    assert "_hermesOrigin" not in _call(_server(), "send_text", {"chatId": "1@c.us"})


@pytest.mark.parametrize("config", [None, {}, {"session_origin": None}])
def test_server_without_opt_in_receives_nothing(config):
    assert "_hermesOrigin" not in _call(_server(config), "watch_chat", {})


def test_session_without_chat_channel_receives_nothing():
    assert "_hermesOrigin" not in _call(_server(), "watch_chat", {}, env={})


def test_model_supplied_origin_is_overwritten():
    sent = _call(_server(), "watch_chat", {"_hermesOrigin": {"platform": "telegram", "chatId": "victim"}})
    assert sent["_hermesOrigin"]["chatId"] == "-1001"


def test_callers_dict_is_not_mutated():
    original = {"chatId": "1@c.us"}
    _call(_server(), "watch_chat", original)
    assert original == {"chatId": "1@c.us"}


def test_custom_argument_name():
    server = _server({"session_origin": {"tools": ["watch_chat"], "argument": "_origin"}})
    sent = _call(server, "watch_chat", {})
    assert "_origin" in sent and "_hermesOrigin" not in sent


@pytest.mark.parametrize("block", [
    {"tools": "watch_chat"}, {"tools": []}, {"tools": ["*"]}, {"tools": [""]}, {"tools": [1]},
    {"tools": ["a"], "argument": ""}, {}, "watch_chat",
])
def test_malformed_block_opts_nothing_in(block):
    assert parse_session_origin_config("srv", {"session_origin": block}) is None
    assert "_hermesOrigin" not in _call(_server({"session_origin": block}), "watch_chat", {})
