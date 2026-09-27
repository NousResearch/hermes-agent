"""Matrix reaction actions use the current gateway session and its receiving adapter."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.tools_config import _get_platform_tools
from tools.registry import registry


@pytest.mark.asyncio
async def test_matrix_reaction_uses_current_message_and_receiving_adapter():
    importlib.import_module("tools.matrix_reaction_tool")
    adapter = SimpleNamespace(
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
