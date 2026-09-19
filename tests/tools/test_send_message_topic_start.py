"""Behavior tests for the Telegram topic kickoff tool."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from plugins.platforms.telegram import topic_tool


def test_topic_tool_is_registered_only_in_the_telegram_bundle():
    ctx = SimpleNamespace(register_tool=MagicMock())

    topic_tool.register_topic_tool(ctx)

    ctx.register_tool.assert_called_once_with(
        name="telegram_topic_start",
        toolset="telegram",
        schema=topic_tool.TELEGRAM_TOPIC_START_SCHEMA,
        handler=topic_tool.telegram_topic_start,
        emoji="🧵",
    )
    from toolsets import TOOLSETS

    assert "telegram_topic_start" in TOOLSETS["hermes-telegram"]["tools"]
    assert "telegram_topic_start" not in TOOLSETS["hermes-cli"]["tools"]


def test_topic_start_creates_current_chat_topic_and_wakes_it(monkeypatch):
    create_thread = AsyncMock(return_value="444")
    adapter = SimpleNamespace(create_handoff_thread=create_thread)
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    runner = SimpleNamespace(adapters={Platform.TELEGRAM: adapter}, _gateway_loop=loop)
    wake = AsyncMock()
    values = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "-100123",
        "HERMES_SESSION_USER_ID": "42",
        "HERMES_SESSION_PROFILE": "dev",
    }

    import gateway.run as gateway_run
    import gateway.session_context as session_context

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": values.get(key, default))

    try:
        with patch("gateway.wake.deliver_wake", wake):
            result = json.loads(
                topic_tool.telegram_topic_start(
                    {"topic_name": "Research", "prompt": "Research the launch."}
                )
            )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()

    assert result == {
        "success": True,
        "platform": "telegram",
        "chat_id": "-100123",
        "thread_id": "444",
        "topic_name": "Research",
        "started": True,
    }
    create_thread.assert_awaited_once_with("-100123", "Research")
    wake.assert_awaited_once()
    source = wake.await_args.kwargs["source"]
    assert wake.await_args.args == (adapter,)
    assert wake.await_args.kwargs["text"] == "Research the launch."
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100123"
    assert source.chat_type == "forum"
    assert source.thread_id == "444"
    assert source.user_id == "42"
    assert source.profile == "dev"


def test_topic_start_refuses_non_telegram_session(monkeypatch):
    import gateway.session_context as session_context

    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": "slack" if key == "HERMES_SESSION_PLATFORM" else default,
    )

    result = json.loads(
        topic_tool.telegram_topic_start(
            {"topic_name": "Research", "prompt": "Research the launch."}
        )
    )

    assert result == {
        "success": False,
        "error": "telegram_topic_start is available only from a Telegram session",
    }
