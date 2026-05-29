"""Unit tests for MCP passthrough interception and gateway delivery."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.memory_provider import MemoryProvider
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.mcp_passthrough import forward_mcp_passthrough_notification
from gateway.session import SessionSource
from tools.mcp_tool import MCPServerTask, _PassthroughInterceptingReadStream


class _FakeReadStream:
    def __init__(self, messages):
        self._messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def aclose(self):
        return None

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


class _RecordingMemoryProvider(MemoryProvider):
    def __init__(self):
        self.initialized = []
        self.events = []
        self.shutdown_called = False

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self.initialized.append((session_id, kwargs))

    def get_tool_schemas(self):
        return []

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        self.events.append((user_content, assistant_content, session_id))

    def sync_passive_event(
        self,
        content: str,
        *,
        session_id: str = "",
        source_label: str = "",
        metadata=None,
    ) -> None:
        self.events.append((content, session_id, source_label, dict(metadata or {})))

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.mark.asyncio
async def test_handle_passthrough_notification_prefers_markdown_field(caplog):
    server = MCPServerTask("notif_srv")
    server._passthrough_targets = ["wecom"]
    message = SimpleNamespace(
        message=SimpleNamespace(
            root=SimpleNamespace(
                method="notifications/message",
                params={
                    "type": "alert",
                    "markdown": "# 告警研判主动推送\n\n- 告警 ID: 181953",
                    "summary": "ignored",
                },
            )
        )
    )

    with caplog.at_level(logging.INFO):
        with patch(
            "gateway.mcp_passthrough.forward_mcp_passthrough_notification",
            new=AsyncMock(),
        ) as forward:
            handled = await server._handle_passthrough_notification(message)

    assert handled is True
    forward.assert_awaited_once_with(
        server_name="notif_srv",
        payload_json="# 告警研判主动推送\n\n- 告警 ID: 181953",
        targets=["wecom"],
    )
    assert "payload_source=markdown" in caplog.text


@pytest.mark.asyncio
async def test_handle_passthrough_notification_falls_back_to_raw_json_without_markdown():
    server = MCPServerTask("notif_srv")
    server._passthrough_targets = ["wecom"]
    message = SimpleNamespace(
        message=SimpleNamespace(
            root=SimpleNamespace(
                method="notifications/message",
                params={"text": "hello", "count": 2},
            )
        )
    )

    with patch(
        "gateway.mcp_passthrough.forward_mcp_passthrough_notification",
        new=AsyncMock(),
    ) as forward:
        handled = await server._handle_passthrough_notification(message)

    assert handled is True
    forward.assert_awaited_once_with(
        server_name="notif_srv",
        payload_json='{"text": "hello", "count": 2}',
        targets=["wecom"],
    )


@pytest.mark.asyncio
async def test_handle_passthrough_notification_logs_custom_method_mismatch(caplog):
    server = MCPServerTask("notif_srv")
    server._passthrough_targets = ["wecom"]
    message = SimpleNamespace(
        message=SimpleNamespace(
            root=SimpleNamespace(
                method="notifications/alert",
                params={"text": "hello"},
            )
        )
    )

    with caplog.at_level(logging.INFO):
        handled = await server._handle_passthrough_notification(message)

    assert handled is False
    assert "received notification 'notifications/alert' but passthrough only forwards 'notifications/message'" in caplog.text


@pytest.mark.asyncio
async def test_handle_passthrough_notification_logs_missing_params(caplog):
    server = MCPServerTask("notif_srv")
    server._passthrough_targets = ["wecom"]
    message = SimpleNamespace(
        message=SimpleNamespace(
            root=SimpleNamespace(
                method="notifications/message",
            )
        )
    )

    with caplog.at_level(logging.WARNING):
        handled = await server._handle_passthrough_notification(message)

    assert handled is True
    assert "received notifications/message without params; not forwarding" in caplog.text


@pytest.mark.asyncio
async def test_intercepting_read_stream_swallows_handled_notifications():
    passthrough_message = SimpleNamespace(
        message=SimpleNamespace(
            root=SimpleNamespace(method="notifications/message", params={"event": "alert"})
        )
    )
    normal_message = SimpleNamespace(message=SimpleNamespace(root=SimpleNamespace(method="notifications/tools/list_changed")))
    handler = AsyncMock(side_effect=[True, False])
    stream = _PassthroughInterceptingReadStream(
        _FakeReadStream([passthrough_message, normal_message]),
        handler,
        "notif_srv",
    )

    seen = []
    async for item in stream:
        seen.append(item)

    assert seen == [normal_message]
    assert handler.await_count == 2


@pytest.mark.asyncio
async def test_forward_passthrough_retries_then_persists_to_live_sessions(caplog):
    home = HomeChannel(platform=Platform.WECOM, chat_id="home-chat", name="Ops")
    config = GatewayConfig(
        platforms={
            Platform.WECOM: PlatformConfig(enabled=True, home_channel=home),
        }
    )
    session_entry = SimpleNamespace(
        session_id="sess-1",
        session_key="agent:main:wecom:dm:home-chat",
        origin=SessionSource(
            platform=Platform.WECOM,
            chat_id="home-chat",
            chat_type="dm",
            user_id="home-chat",
            user_name="Ops",
        ),
        platform=Platform.WECOM,
    )
    adapter = SimpleNamespace(
        send=AsyncMock(
            side_effect=[
                SimpleNamespace(success=False, error="first failure"),
                SimpleNamespace(success=False, error="second failure"),
                SimpleNamespace(success=True, message_id="msg-1"),
            ]
        )
    )
    session_store = SimpleNamespace(
        list_sessions=MagicMock(return_value=[session_entry]),
        append_to_transcript=MagicMock(),
        update_session=MagicMock(),
        get_or_create_session=MagicMock(return_value=session_entry),
    )
    runner = SimpleNamespace(
        config=config,
        adapters={Platform.WECOM: adapter},
        session_store=session_store,
    )

    with caplog.at_level(logging.INFO):
        with patch("gateway.mcp_passthrough._get_live_runner", return_value=runner), patch(
            "gateway.mcp_passthrough.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            await forward_mcp_passthrough_notification(
                server_name="notif_srv",
                payload_json='{"event": "ping"}',
                targets=["wecom"],
            )

    assert adapter.send.await_count == 3
    assert sleep_mock.await_count == 2
    session_store.append_to_transcript.assert_called_once_with(
        "sess-1",
        {
            "role": "assistant",
            "content": '[MCP passthrough from notif_srv] {"event": "ping"}',
        },
    )
    session_store.update_session.assert_called_once_with("agent:main:wecom:dm:home-chat")
    assert "passthrough delivered to wecom on attempt 3/3" in caplog.text
    assert "passthrough target=wecom persisted=1 memory_synced=0" in caplog.text


@pytest.mark.asyncio
async def test_forward_passthrough_syncs_active_memory_provider_for_target_session():
    home = HomeChannel(platform=Platform.WECOM, chat_id="home-chat", name="Ops")
    config = GatewayConfig(
        platforms={
            Platform.WECOM: PlatformConfig(enabled=True, home_channel=home),
        }
    )
    session_entry = SimpleNamespace(
        session_id="sess-1",
        session_key="agent:main:wecom:dm:home-chat",
        origin=SessionSource(
            platform=Platform.WECOM,
            chat_id="home-chat",
            chat_type="dm",
            user_id="user-1",
            user_name="Ops User",
        ),
        platform=Platform.WECOM,
    )
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-1"))
    )
    session_store = SimpleNamespace(
        list_sessions=MagicMock(return_value=[session_entry]),
        append_to_transcript=MagicMock(),
        update_session=MagicMock(),
        get_or_create_session=MagicMock(return_value=session_entry),
    )
    runner = SimpleNamespace(
        config=config,
        adapters={Platform.WECOM: adapter},
        session_store=session_store,
    )
    provider = _RecordingMemoryProvider()

    with patch("gateway.mcp_passthrough._get_live_runner", return_value=runner), patch(
        "hermes_cli.config.load_config",
        return_value={"memory": {"provider": "recording"}},
    ), patch(
        "plugins.memory.load_memory_provider",
        return_value=provider,
    ), patch(
        "hermes_constants.get_hermes_home",
        return_value="/tmp/hermes-home",
    ), patch(
        "hermes_cli.profiles.get_active_profile_name",
        return_value="coder",
    ):
        await forward_mcp_passthrough_notification(
            server_name="notif_srv",
            payload_json=json.dumps({"event": "ping"}, ensure_ascii=False),
            targets=["wecom"],
        )

    assert provider.initialized[0][0] == "sess-1"
    init_kwargs = provider.initialized[0][1]
    assert init_kwargs["platform"] == "wecom"
    assert init_kwargs["user_id"] == "user-1"
    assert init_kwargs["chat_id"] == "home-chat"
    assert init_kwargs["gateway_session_key"] == "agent:main:wecom:dm:home-chat"
    assert provider.events == [
        (
            '{"event": "ping"}',
            "sess-1",
            "mcp:notif_srv",
            {
                "kind": "mcp_passthrough",
                "server_name": "notif_srv",
                "platform": "wecom",
            },
        )
    ]
    assert provider.shutdown_called is True