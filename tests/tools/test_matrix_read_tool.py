"""Matrix reads use the live session's receiving adapter and its policy."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import TypedDict, Unpack
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import (
    clear_session_vars,
    get_session_transport,
    set_session_vars,
)
from hermes_cli.tools_config import _get_platform_tools
from tools.registry import registry

importlib.import_module("tools.matrix_read_tool")


class _MatrixSessionOverrides(TypedDict, total=False):
    platform: str
    chat_id: str
    user_id: str
    thread_id: str
    session_key: str
    transport_adapter: object | None
    transport_loop: asyncio.AbstractEventLoop | None


def _bind_matrix_session(
    adapter: object | None, **overrides: Unpack[_MatrixSessionOverrides]
):
    values: _MatrixSessionOverrides = dict(
        platform="matrix",
        chat_id="!room:server",
        user_id="@alice:server",
        transport_adapter=adapter,
        transport_loop=asyncio.get_running_loop(),
    )
    values.update(overrides)
    return set_session_vars(**values)


@pytest.mark.asyncio
async def test_matrix_read_uses_session_owner_and_room():
    adapter = SimpleNamespace(
        read_matrix_context=AsyncMock(
            return_value={"events": [{"event_id": "$one", "body": "hello"}]}
        )
    )
    tokens = _bind_matrix_session(
        adapter, thread_id="$root", session_key="matrix-session"
    )
    try:
        raw_result = await asyncio.to_thread(
            registry.dispatch,
            "matrix_read",
            {"kind": "room", "limit": 5},
        )
        assert isinstance(raw_result, str)
        result = json.loads(raw_result)
    finally:
        clear_session_vars(tokens)

    assert result == {"events": [{"event_id": "$one", "body": "hello"}]}
    assert get_session_transport() == (None, None)
    adapter.read_matrix_context.assert_awaited_once_with(
        "room",
        "!room:server",
        None,
        5,
        requester="@alice:server",
    )


def test_matrix_read_requires_live_matrix_session():
    tokens = set_session_vars(platform="cli", chat_id="!room:server")
    try:
        raw_result = registry.dispatch("matrix_read", {"kind": "room"})
        assert isinstance(raw_result, str)
        result = json.loads(raw_result)
    finally:
        clear_session_vars(tokens)

    assert result == {"error": "Matrix reads require a live Matrix session"}


def test_matrix_read_is_only_in_matrix_default_toolset():
    assert (
        "matrix_read" in _get_platform_tools({}, "matrix"),
        "matrix_read" in _get_platform_tools({}, "telegram"),
    ) == (True, False)


@pytest.mark.asyncio
async def test_matrix_read_runs_on_owning_gateway_loop():
    owner_loop = asyncio.get_running_loop()

    async def read(*args, **kwargs):
        return {"on_owner_loop": asyncio.get_running_loop() is owner_loop}

    tokens = _bind_matrix_session(SimpleNamespace(read_matrix_context=read))
    try:
        result = await asyncio.to_thread(
            registry.dispatch, "matrix_read", {"kind": "room"}
        )
    finally:
        clear_session_vars(tokens)

    assert isinstance(result, str)
    assert json.loads(result) == {"on_owner_loop": True}


@pytest.mark.asyncio
async def test_matrix_read_refuses_a_stopped_owner_loop():
    adapter = SimpleNamespace(
        read_matrix_context=AsyncMock(return_value={"events": []})
    )
    stopped_loop = asyncio.new_event_loop()
    tokens = _bind_matrix_session(adapter, transport_loop=stopped_loop)
    try:
        result = await asyncio.to_thread(
            registry.dispatch, "matrix_read", {"kind": "room"}
        )
    finally:
        clear_session_vars(tokens)
        stopped_loop.close()

    assert isinstance(result, str)
    assert json.loads(result) == {"error": "Matrix gateway loop is unavailable"}
    adapter.read_matrix_context.assert_not_awaited()
