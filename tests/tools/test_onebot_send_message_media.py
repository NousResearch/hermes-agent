"""OneBot media delivery for send_message (#94695 consolidation).

Covers the tools/send_message_tool._send_to_platform onebot branch:
standalone_sender_fn routing for cron out-of-process delivery
(media rides on the last text chunk), and the missing-sender error path.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform, PlatformConfig
from tools.send_message_tool import _send_to_platform


def _onebot_entry(sender_mock):
    return SimpleNamespace(
        standalone_sender_fn=sender_mock,
        send_message_handler=None,
    )


def test_onebot_media_routes_text_then_media_to_standalone_sender() -> None:
    """文本分块 + 媒体只在最后一块传给 standalone sender。"""
    sender = AsyncMock(
        return_value={"success": True, "message_id": "123"}
    )
    pconfig = PlatformConfig(enabled=True, extra={})

    async def run():
        with (
            patch("hermes_cli.plugins.discover_plugins", return_value=None),
            patch(
                "gateway.platform_registry.platform_registry.get",
                return_value=_onebot_entry(sender),
            ),
        ):
            return await _send_to_platform(
                Platform("onebot"),
                pconfig,
                "private:123456789",
                "长文本" * 500,
                media_files=[("/data/audio/remind.silk", True)],
            )

    result = asyncio.run(run())
    assert result["success"] is True
    assert sender.await_count >= 1, "standalone sender must be called"
    calls = sender.await_args_list
    # media 只出现在最后一次调用（is_last chunk）
    assert all(c.kwargs.get("media_files") is None for c in calls[:-1])
    assert calls[-1].kwargs["media_files"] == [("/data/audio/remind.silk", True)]


def test_onebot_missing_standalone_sender_returns_error() -> None:
    """插件未注册 standalone_sender_fn 时返回明确错误。"""
    pconfig = PlatformConfig(enabled=True, extra={})

    async def run():
        with (
            patch("hermes_cli.plugins.discover_plugins", return_value=None),
            patch(
                "gateway.platform_registry.platform_registry.get",
                return_value=None,
            ),
        ):
            return await _send_to_platform(
                Platform("onebot"),
                pconfig,
                "private:123456789",
                "hi",
                media_files=[("/tmp/a.png", False)],
            )

    result = asyncio.run(run())
    assert "missing standalone_sender_fn" in result["error"]


def test_onebot_without_media_does_not_touch_standalone_sender() -> None:
    """纯文本消息不触发媒体分支：standalone sender 被调用但不携带 media_files。"""
    sender = AsyncMock(return_value={"success": True, "message_id": "1"})
    pconfig = PlatformConfig(enabled=True, extra={})

    async def run():
        with (
            patch("hermes_cli.plugins.discover_plugins", return_value=None),
            patch(
                "gateway.platform_registry.platform_registry.get",
                return_value=_onebot_entry(sender),
            ),
        ):
            # 无 media_files → onebot 媒体分支被跳过；文本仍可经 standalone
            # sender 离进程发送，但 media_files 必须为 None。
            return await _send_to_platform(
                Platform("onebot"),
                pconfig,
                "private:123456789",
                "纯文本",
            )

    asyncio.run(run())
    assert sender.await_count == 1
    assert not sender.await_args_list[0].kwargs.get("media_files")