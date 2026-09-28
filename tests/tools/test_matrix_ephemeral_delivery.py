"""Ephemeral delivery uses native authenticated alias resolution."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.matrix import adapter as matrix
from tools import send_message_senders as senders


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["access-token", ""])
async def test_alias_reaches_connected_native_adapter(monkeypatch, token):
    alias = "#scheduled:remote.test"
    adapter = SimpleNamespace(
        connect=AsyncMock(return_value=True),
        disconnect=AsyncMock(),
        send=AsyncMock(return_value=SendResult(success=True, message_id="$sent")),
    )
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (None, None))
    monkeypatch.setattr(matrix, "MatrixAdapter", lambda config: adapter)
    config = SimpleNamespace(
        token=token, extra={"homeserver": "https://remote.test", "password": "password"}
    )

    result = await senders._send_matrix_via_adapter(
        config, alias, "report", thread_id="$root"
    )

    assert result == {
        "success": True,
        "platform": "matrix",
        "chat_id": alias,
        "message_id": "$sent",
    }
    adapter.send.assert_awaited_once_with(
        alias, "report", metadata={"thread_id": "$root"}
    )
    adapter.disconnect.assert_awaited_once()


@pytest.mark.parametrize("separator", ["/", ":"])
@pytest.mark.parametrize(
    "destination", ["!scheduled:remote.test", "#scheduled:remote.test"]
)
@pytest.mark.parametrize("rejected", [False, True])
def test_bare_platform_send_uses_configured_home_thread(
    tmp_path, monkeypatch, separator, destination, rejected
):
    from gateway.config import PlatformConfig
    from tools.send_message_tool import send_message_tool
    from tools import send_message_tool as messaging
    import model_tools

    target = f"{destination}{separator}$root"
    (tmp_path / "config.yaml").write_text(
        json.dumps({
            "platforms": {
                "matrix": {
                    "enabled": True,
                    "token": "token",
                    "home_channel": {"platform": "matrix", "chat_id": target},
                    "extra": {"homeserver": "https://remote.test", "e2ee_mode": "off"},
                }
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = matrix.MatrixAdapter(
        PlatformConfig(enabled=True, extra={"e2ee_mode": "off"})
    )
    adapter._client = SimpleNamespace(
        resolve_room_alias=AsyncMock(
            return_value=SimpleNamespace(
                room_id="!scheduled:remote.test", servers=["route.test"]
            )
        ),
        join_room=AsyncMock(return_value="!scheduled:remote.test"),
        send_message_event=AsyncMock(return_value="$sent"),
        get_state_event=AsyncMock(side_effect=_MissingEncryption()),
        crypto=None,
    )
    adapter._refresh_dm_cache = AsyncMock()
    if rejected:
        adapter._client.get_state_event.side_effect = ValueError("state forbidden")
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (None, adapter))
    monkeypatch.setattr(model_tools, "_run_async", asyncio.run)
    monkeypatch.setattr(messaging, "_mirror_sent_message", lambda *a: False)

    result = json.loads(send_message_tool({"target": "matrix", "message": "report"}))

    if rejected:
        assert target in result["error"]
        adapter._client.send_message_event.assert_not_awaited()
        return
    assert result.get("success"), result
    args = adapter._client.send_message_event.await_args.args
    assert (str(args[0]), args[2].get("m.relates_to")) == (
        "!scheduled:remote.test",
        {
            "rel_type": "m.thread",
            "event_id": "$root",
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": "$root"},
        },
    )


class _MissingEncryption(Exception):
    errcode = "M_NOT_FOUND"
