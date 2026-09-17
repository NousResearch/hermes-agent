"""Outbound send bodies must not carry lone surrogate code points (#113799).

``hermes send MESSAGE`` passes argv through surrogateescape decoding on macOS
shells; a lone surrogate then crashes the UTF-8 marshal inside platform SDK
request bodies (feishu/lark) and the message is lost after the retries.
``_handle_send`` scrubs before media extraction and the session mirror consume
the text; ``_send_to_platform`` re-scrubs as the chokepoint for direct callers
(cron standalone delivery), and ``_handle_react`` scrubs the emoji the same way.
"""

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import send_message_tool

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    )


def _send(message):
    send_mock = AsyncMock(return_value={"success": True})
    with patch("gateway.config.load_gateway_config", return_value=_make_config()), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=send_mock), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool({"action": "send", "target": "telegram:12345", "message": message})
        )
    return result, send_mock


def test_lone_surrogates_are_scrubbed_before_the_platform_send():
    result, send_mock = _send("警报 \ud800\udfff done")

    assert result["success"] is True
    delivered = send_mock.await_args.args[3]
    assert not _SURROGATE_RE.search(delivered)
    assert delivered == "警报 \ufffd\ufffd done"


def test_surrogate_free_message_is_sent_unchanged():
    result, send_mock = _send("all clean: 你好 \U0001f44d")

    assert result["success"] is True
    assert send_mock.await_args.args[3] == "all clean: 你好 \U0001f44d"


def test_cron_style_direct_send_to_platform_is_scrubbed():
    # cron/scheduler_delivery.py::_standalone_send awaits _send_to_platform
    # directly (bypassing _handle_send), so the chokepoint must scrub there.
    from tools.send_message_tool import _send_to_platform

    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    send_mock = AsyncMock(return_value={"success": True})
    with patch("tools.send_message_tool._send_telegram", new=send_mock):
        result = asyncio.run(
            _send_to_platform(Platform.TELEGRAM, telegram_cfg, "12345", "cron \ud83d end"))

    assert result["success"] is True
    delivered = send_mock.await_args.args[2]
    assert not _SURROGATE_RE.search(delivered)
    assert delivered == "cron \ufffd end"


def test_react_emoji_is_scrubbed_before_the_adapter():
    react_mock = AsyncMock(return_value={"success": True})
    with patch("tools.send_message_tool._resolve_tool_target",
               return_value=("telegram", "12345", None, None)), \
         patch("tools.send_message_tool._platform_enum",
               return_value=(Platform.TELEGRAM, None)), \
         patch("tools.send_message_tool._authorize_relay_target", return_value=None), \
         patch("tools.send_message_tool._live_adapter",
               return_value=(True, SimpleNamespace(add_reaction=react_mock))), \
         patch("model_tools._run_async", side_effect=_run_async_immediately):
        result = json.loads(send_message_tool({
            "action": "react", "target": "telegram:12345",
            "message_id": "m1", "emoji": "\u2764\ufe0f\ud83d"}))

    assert result["success"] is True
    assert react_mock.await_args.kwargs["emoji"] == "\u2764\ufe0f\ufffd"
