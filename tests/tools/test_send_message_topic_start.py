"""Behavior tests for the Telegram topic kickoff tool."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    admit = AsyncMock()
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
        with patch("gateway.wake.admit_gateway_event", admit):
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
    admit.assert_awaited_once()
    await_args = admit.await_args
    assert await_args is not None
    assert await_args.args[0] is adapter
    event = await_args.args[1]
    assert event.text == "Research the launch."
    assert event.internal is False
    assert event.delegated_continuation is True
    assert event.allow_gateway_control is False
    source = event.source
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100123"
    assert source.chat_type == "forum"
    assert source.thread_id == "444"
    assert source.user_id == "42"
    assert source.profile == "dev"


@pytest.mark.asyncio
async def test_topic_start_uses_normal_admission_and_respects_pause():
    from agent import estop
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    authorized = []

    def authorize(source, *, allow_adapter_delegation=True):
        authorized.append(source)
        return True

    runner._is_user_authorized = authorize
    observed = {}

    class AdmissionAdapter:
        async def create_handoff_thread(self, chat_id, topic_name):
            return "444"

        async def handle_message(self, event):
            event._gateway_accepted = True
            observed["event"] = event
            observed["reply"] = await runner._handle_message(event)

    estop.engage(reason="maintenance")
    try:
        result = await topic_tool._create_topic_and_start(
            AdmissionAdapter(),
            chat_id="-100123",
            topic_name="Research",
            prompt="Research the launch.",
            user_id="42",
            profile="dev",
        )
    finally:
        estop.disengage()

    event = observed["event"]
    assert result["success"] is True
    assert authorized == [event.source]
    assert event.internal is False
    assert event.delegated_continuation is True
    assert event.allow_gateway_control is False
    assert "paused" in observed["reply"].lower()
    assert "maintenance" in observed["reply"]
    persisted = runner._hmwa_user_transcript_entry(
        event,
        SimpleNamespace(
            persist_user_message=event.text,
            message_text=event.text,
            persist_user_timestamp=None,
            persist_user_display_kind=None,
            persistence_owner=None,
        ),
        1.0,
    )
    assert persisted["display_metadata"] == {"input_origin": "agent_delegated_continuation"}



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
