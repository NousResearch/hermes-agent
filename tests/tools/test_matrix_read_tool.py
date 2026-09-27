"""Matrix reads use the live session's receiving adapter and its policy."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import clear_session_vars, get_session_transport, set_session_vars
from hermes_cli.tools_config import _get_platform_tools
from tools.registry import registry

importlib.import_module("tools.matrix_read_tool")


def test_matrix_read_uses_session_owner_and_rejects_other_rooms():
    adapter = SimpleNamespace(
        read_matrix_context=AsyncMock(return_value={"events": [{"event_id": "$one", "body": "hello"}]})
    )
    tokens = set_session_vars(
        platform="matrix", chat_id="!room:server", user_id="@alice:server",
        thread_id="$root", session_key="matrix-session", transport_adapter=adapter,
    )
    try:
        result = json.loads(registry.dispatch("matrix_read", {"kind": "room", "limit": 5}))
        wrong_room = json.loads(registry.dispatch(
            "matrix_read", {"kind": "room", "room_id": "!other:server", "limit": 5}
        ))
    finally:
        clear_session_vars(tokens)

    assert result == {"events": [{"event_id": "$one", "body": "hello"}]}
    assert wrong_room == {"error": "Matrix reads are limited to the current room"}
    assert get_session_transport() == (None, None)
    adapter.read_matrix_context.assert_awaited_once_with(
        "room", "!room:server", None, 5, requester="@alice:server",
    )


def test_matrix_read_requires_live_matrix_session():
    tokens = set_session_vars(platform="cli", chat_id="!room:server")
    try:
        result = json.loads(registry.dispatch("matrix_read", {"kind": "room"}))
    finally:
        clear_session_vars(tokens)

    assert result == {"error": "Matrix reads require a live Matrix session"}


def test_matrix_read_is_only_in_matrix_default_toolset():
    assert ("matrix_read" in _get_platform_tools({}, "matrix"),
            "matrix_read" in _get_platform_tools({}, "telegram")) == (True, False)


@pytest.mark.asyncio
async def test_matrix_read_runs_on_owning_gateway_loop():
    owner_loop = asyncio.get_running_loop()

    async def read(*args, **kwargs):
        return {"on_owner_loop": asyncio.get_running_loop() is owner_loop}

    adapter = SimpleNamespace(read_matrix_context=read)
    tokens = set_session_vars(
        platform="matrix", chat_id="!room:server", user_id="@alice:server",
        transport_adapter=adapter,
    )
    try:
        result = await asyncio.to_thread(registry.dispatch, "matrix_read", {"kind": "room"})
    finally:
        clear_session_vars(tokens)

    assert json.loads(result) == {"on_owner_loop": True}
