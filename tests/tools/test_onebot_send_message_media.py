"""OneBot media delivery for send_message (#94695 consolidation).

Covers the tools/send_message_tool._send_to_platform onebot branch:
standalone_sender_fn routing for cron out-of-process delivery
(media rides on the last text chunk), and the missing-sender error path.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# 回归（DEVLOG 已知限制#3 复核）：cron 离进程投递的媒体不再被白名单丢弃。
# commit b1465df01d 已把 onebot 纳入 _PLUGIN_STANDALONE_MEDIA 与
# _MEDIA_PLATFORMS_NOTE；以下用例实证「图片 + 文本」经
# _send_to_platform → _send_plugin_standalone → 插件 _standalone_send
# （registry.standalone_sender_fn）链路真实投递到 OneBot HTTP API。
# ---------------------------------------------------------------------------


def _tmp_image() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".png", dir=tempfile.gettempdir(), delete=False)
    f.write(b"\x89PNG\r\n\x1a\n")
    f.close()
    return f.name


def _resp(retcode: int = 0):
    r = AsyncMock()
    r.json = AsyncMock(return_value={"retcode": retcode, "data": {"message_id": 1}})
    return r


def _onebot_http_session():
    """构造被 mock 的 aiohttp.ClientSession，按序记录每次 POST (url, json payload)。"""
    calls = []

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_resp())
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, calls


def test_onebot_cron_image_with_text_reaches_standalone_sender() -> None:
    """cron 媒体白名单回归：图片 + 文本经 _send_plugin_standalone 投递，
    插件 standalone sender 必须同时收到文本块与 media_files（图片）。"""
    sender = AsyncMock(return_value={"success": True, "message_id": "1"})
    image = _tmp_image()
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
                "group:123456789",
                "每日早报",
                media_files=[(image, False)],
            )

    result = asyncio.run(run())
    assert result["success"] is True, result
    # 单 chunk：文本与媒体同一次调用，二者都不能被白名单丢掉
    assert sender.await_count == 1
    call = sender.await_args
    assert call.args[2] == "每日早报", "文本必须原样到达 standalone sender"
    assert call.kwargs["media_files"] == [(image, False)], "图片必须原样到达 standalone sender"
    assert call.kwargs["force_document"] is False


def test_onebot_cron_standalone_chain_delivers_text_then_image_to_onebot_http() -> None:
    """端到端回归：_send_to_platform → _send_plugin_standalone → 插件真实
    _standalone_send → OneBot HTTP API。文本与图片各成一条 CQ 消息，
    均不被丢弃（mock 边界：aiohttp.ClientSession）。"""
    from plugins.platforms.onebot import adapter as onebot_adapter

    image = _tmp_image()
    session_ctx, calls = _onebot_http_session()
    pconfig = PlatformConfig(
        enabled=True,
        extra={
            "http_url": "http://127.0.0.1:5700",
            # 提供 voice_mount 以免触发 config.yaml 回退读取；前缀与临时目录
            # 不重叠，路径不会被重映射。
            "voice_mount": {"host": "/host/media", "container": "/cont/media"},
        },
    )
    entry = SimpleNamespace(
        standalone_sender_fn=onebot_adapter._standalone_send,
        send_message_handler=None,
    )

    async def run():
        with (
            patch("hermes_cli.plugins.discover_plugins", return_value=None),
            patch("gateway.platform_registry.platform_registry.get", return_value=entry),
            patch("plugins.platforms.onebot.adapter.aiohttp.ClientSession", return_value=session_ctx),
            patch.dict(os.environ, {"ONEBOT_HTTP_URL": "", "ONEBOT_ACCESS_TOKEN": ""}),
        ):
            return await _send_to_platform(
                Platform("onebot"),
                pconfig,
                "group:123456789",
                "每日早报",
                media_files=[(image, False)],
            )

    result = asyncio.run(run())
    assert result["success"] is True, result
    assert len(calls) >= 2, f"文本与图片应至少各一次 POST，实际 {calls}"
    posts = [payload for _, payload in calls]
    assert posts[0]["message"] == "每日早报", "文本必须先于图片投递且不被丢弃"
    assert any(p["message"] == f"[CQ:image,file={image}]" for p in posts[1:]), "图片必须以 CQ:image 段投递且不被丢弃"
    assert all(u.endswith("/send_group_msg") for u, _ in calls), "group 目标应走 send_group_msg"


def test_onebot_register_wires_standalone_sender_fn_for_cron() -> None:
    """注册链路：register(ctx) 必须把模块级 _standalone_send 挂到
    PlatformEntry.standalone_sender_fn（cron 离进程投递的取用点），
    并声明 cron 投递环境变量。"""
    from plugins.platforms.onebot import adapter as onebot_adapter

    ctx = MagicMock()
    onebot_adapter.register(ctx)
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["standalone_sender_fn"] is onebot_adapter._standalone_send
    assert kwargs["cron_deliver_env_var"] == "ONEBOT_HOME_CHANNEL"