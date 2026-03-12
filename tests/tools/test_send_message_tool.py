"""Tests for tools/send_message_tool.py."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import _send_qq, send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    telegram_cfg = SimpleNamespace(enabled=True, token="fake-token", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    ), telegram_cfg


class TestSendMessageTool:
    def test_sends_to_explicit_telegram_topic_target(self):
        config, telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:-1001:17585",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(Platform.TELEGRAM, telegram_cfg, "-1001", "hello", thread_id="17585")
        mirror_mock.assert_called_once_with("telegram", "-1001", "hello", source_label="cli", thread_id="17585")


class TestQQStandaloneSend:
    @pytest.mark.asyncio
    async def test_send_qq_imports_httpx_and_sends_message(self, monkeypatch):
        sent_payloads = []

        class _Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                if url.endswith("/getAppAccessToken"):
                    return _Response({"access_token": "qq-token"})
                sent_payloads.append((url, json, headers))
                return _Response({"id": f"msg-{len(sent_payloads)}"})

        fake_httpx = SimpleNamespace(AsyncClient=lambda timeout=30.0: _Client())
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        result = await _send_qq(
            SimpleNamespace(token="1024", api_key="secret", extra={}),
            "user:user-openid",
            "hello qq",
        )

        assert result["success"] is True
        assert sent_payloads == [
            (
                "https://api.sgroup.qq.com/v2/users/user-openid/messages",
                {"msg_type": 0, "content": "hello qq"},
                {"Authorization": "QQBot qq-token", "X-Union-Appid": "1024"},
            )
        ]

    @pytest.mark.asyncio
    async def test_send_qq_chunks_long_messages(self, monkeypatch):
        sent_payloads = []

        class _Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                if url.endswith("/getAppAccessToken"):
                    return _Response({"access_token": "qq-token"})
                sent_payloads.append(json["content"])
                return _Response({"id": f"msg-{len(sent_payloads)}"})

        fake_httpx = SimpleNamespace(AsyncClient=lambda timeout=30.0: _Client())
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        result = await _send_qq(
            SimpleNamespace(token="1024", api_key="secret", extra={}),
            "group:group-openid",
            "x" * 2001,
        )

        assert result["success"] is True
        assert sent_payloads == ["x" * 2000, "x"]
        assert result["message_ids"] == ["msg-1", "msg-2"]

    def test_resolved_telegram_topic_name_preserves_thread_id(self):
        config, telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("gateway.channel_directory.resolve_channel_name", return_value="-1001:17585"), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:Coaching Chat / topic 17585",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(Platform.TELEGRAM, telegram_cfg, "-1001", "hello", thread_id="17585")
