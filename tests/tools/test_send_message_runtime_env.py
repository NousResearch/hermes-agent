"""Runtime env loading for send_message tool calls."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from hermes_cli.config import get_hermes_home, invalidate_env_cache
from tools.send_message_tool import send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def test_bare_telegram_target_loads_home_channel_from_runtime_files(monkeypatch):
    hermes_home = get_hermes_home()
    (hermes_home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=123:abc\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "TELEGRAM_HOME_CHANNEL: '-2002'\n",
        encoding="utf-8",
    )
    invalidate_env_cache()

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    with patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "telegram",
                    "message": "hello",
                }
            )
        )

    assert result["success"] is True
    assert result["note"] == "Sent to telegram home channel (chat_id: -2002)"
    send_mock.assert_awaited_once()
    args, kwargs = send_mock.await_args
    assert args[0] is Platform.TELEGRAM
    assert args[1].token == "123:abc"
    assert args[2] == "-2002"
    assert args[3] == "hello"
    assert kwargs == {
        "thread_id": None,
        "media_files": [],
        "force_document": False,
    }
