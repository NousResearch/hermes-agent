"""Matrix reaction actions use the current gateway session and its receiving adapter."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.tools_config import _get_platform_tools
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.read_context import SessionAccess
from tools.registry import registry

ROOM = "!room:server"
REQUESTER = "@alice:server"


def _matrix_adapter(*, allowed_rooms=(), authorized=True) -> MatrixAdapter:
    adapter = object.__new__(MatrixAdapter)
    adapter._reactions_enabled = False
    adapter._pending_reactions = {}
    adapter._agent_reactions = {(ROOM, "$current"): ["$earlier"]}
    adapter._joined_rooms = {ROOM}
    adapter._allowed_room_ids = set(allowed_rooms)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._authorization_check = lambda *_args, **_kwargs: authorized
    adapter._send_reaction = AsyncMock(return_value="$reaction")
    adapter._redact_reaction = AsyncMock(return_value=True)
    return adapter


async def _dispatch_in_session(adapter, args, *, user_id=REQUESTER) -> dict:
    importlib.import_module("tools.matrix_reaction_tool")
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=user_id,
        message_id="$current",
        transport_adapter=adapter,
    )
    try:
        return json.loads(
            await asyncio.to_thread(registry.dispatch, "matrix_reaction", args)
        )
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_matrix_reaction_uses_current_message_and_receiving_adapter():
    importlib.import_module("tools.matrix_reaction_tool")
    adapter = SimpleNamespace(
        check_session_access=AsyncMock(return_value=SessionAccess(chat_type="group")),
        add_reaction=AsyncMock(
            return_value={"success": True, "message_id": "$current"}
        ),
        remove_reaction=AsyncMock(
            return_value={"success": True, "message_id": "$current"}
        ),
    )
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!room:server",
        user_id="@alice:server",
        message_id="$current",
        transport_adapter=adapter,
    )
    try:
        react = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "matrix_reaction",
                {"action": "react", "emoji": "👍"},
            )
        )
        unreact = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "matrix_reaction",
                {"action": "unreact"},
            )
        )
    finally:
        clear_session_vars(tokens)

    assert (react, unreact) == (
        {"success": True, "message_id": "$current"},
        {"success": True, "message_id": "$current"},
    )
    adapter.add_reaction.assert_awaited_once_with(
        chat_id="!room:server",
        emoji="👍",
        message_id="$current",
    )
    adapter.remove_reaction.assert_awaited_once_with(
        chat_id="!room:server",
        message_id="$current",
    )


def test_matrix_reaction_is_only_in_matrix_default_toolset():
    assert (
        "matrix_reaction" in _get_platform_tools({}, "matrix"),
        "matrix_reaction" in _get_platform_tools({}, "telegram"),
    ) == (True, False)


@pytest.mark.asyncio
async def test_matrix_reaction_requires_a_live_session_and_an_event():
    importlib.import_module("tools.matrix_reaction_tool")
    adapter = SimpleNamespace(add_reaction=AsyncMock())

    wrong_platform = set_session_vars(
        platform="cli",
        chat_id="!room:server",
        message_id="$current",
        transport_adapter=adapter,
    )
    try:
        platform_result = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "matrix_reaction",
                {"action": "react", "emoji": "👍"},
            )
        )
    finally:
        clear_session_vars(wrong_platform)

    no_event = set_session_vars(
        platform="matrix",
        chat_id="!room:server",
        user_id="@alice:server",
        transport_adapter=adapter,
    )
    try:
        event_result = json.loads(
            await asyncio.to_thread(
                registry.dispatch,
                "matrix_reaction",
                {"action": "react", "emoji": "👍"},
            )
        )
    finally:
        clear_session_vars(no_event)

    assert (platform_result, event_result) == (
        {"error": "Matrix reactions require a live Matrix session"},
        {
            "error": "message_id is required when the session has no current Matrix event"
        },
    )
    adapter.add_reaction.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args", [{"action": "react", "emoji": "👍"}, {"action": "unreact"}]
)
@pytest.mark.parametrize(
    ("policy", "user_id", "error"),
    [
        (
            {"allowed_rooms": {"!other:server"}},
            REQUESTER,
            "Matrix room is not allowed or joined",
        ),
        (
            {"authorized": False},
            REQUESTER,
            "Matrix requester is not authorized for this room",
        ),
        ({}, "", "Matrix reactions require a live Matrix session"),
    ],
)
async def test_matrix_reaction_applies_room_and_requester_policy(args, policy, user_id, error):
    adapter = _matrix_adapter(**policy)

    result = await _dispatch_in_session(adapter, args, user_id=user_id)

    assert (
        result,
        adapter._send_reaction.await_count,
        adapter._redact_reaction.await_count,
        adapter._agent_reactions,
    ) == (
        {"error": error},
        0,
        0,
        {(ROOM, "$current"): ["$earlier"]},
    )


@pytest.mark.asyncio
async def test_timed_out_reaction_is_recorded_when_the_send_completes(monkeypatch):
    tool = importlib.import_module("tools.matrix_reaction_tool")
    adapter = _matrix_adapter()
    adapter._agent_reactions = {}
    send_started = asyncio.Event()
    finish_send = asyncio.Event()

    async def send(*_args):
        send_started.set()
        await finish_send.wait()
        return "$late"

    adapter._send_reaction = AsyncMock(side_effect=send)

    with monkeypatch.context() as patch:
        patch.setattr(tool, "_REACTION_TIMEOUT_SECONDS", 0)
        react = await _dispatch_in_session(adapter, {"action": "react", "emoji": "👍"})
    await asyncio.wait_for(send_started.wait(), timeout=2)
    finish_send.set()
    unreact = await _dispatch_in_session(adapter, {"action": "unreact"})

    assert (
        react,
        unreact,
        [call.args[:2] for call in adapter._redact_reaction.await_args_list],
        adapter._agent_reactions,
    ) == (
        {"error": "Matrix reaction timed out and may still complete"},
        {"success": True, "message_id": "$current"},
        [(ROOM, "$late")],
        {},
    )
