"""Ephemeral delivery uses native authenticated alias resolution."""

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
