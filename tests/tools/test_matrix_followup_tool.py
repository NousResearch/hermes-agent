"""The Matrix follow-up action updates only the current live turn."""

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
async def test_followup_action_validates_filter_and_updates_live_turn():
    importlib.import_module("tools.matrix_followup_tool")
    adapter = SimpleNamespace(configure_reaction_followups=AsyncMock(return_value=True))
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!room:test",
        user_id="@alice:test",
        session_key="agent:matrix:room",
        session_id="sid",
        transport_adapter=adapter,
        transport_loop=asyncio.get_running_loop(),
    )
    try:
        invalid = json.loads(
            await asyncio.to_thread(
                registry.dispatch, "matrix_followup", {"enabled": True, "emoji": []}
            )
        )
        enabled = json.loads(
            await asyncio.to_thread(
                registry.dispatch, "matrix_followup", {"enabled": True, "emoji": ["👍"]}
            )
        )
        disabled = json.loads(
            await asyncio.to_thread(
                registry.dispatch, "matrix_followup", {"enabled": False}
            )
        )
    finally:
        clear_session_vars(tokens)

    assert (invalid, enabled, disabled) == (
        {"error": "emoji must be a non-empty list of non-empty strings"},
        {"success": True, "enabled": True, "emoji": ["👍"]},
        {"success": True, "enabled": False, "emoji": []},
    )
    assert [
        (call.args, call.kwargs)
        for call in adapter.configure_reaction_followups.await_args_list
    ] == [
        (("agent:matrix:room", True, ("👍",)), {
            "room_id": "!room:test", "requester": "@alice:test",
            "thread_id": "", "profile": "", "session_id": "sid",
        }),
        (("agent:matrix:room", False, ()), {
            "room_id": "!room:test", "requester": "@alice:test",
            "thread_id": "", "profile": "", "session_id": "sid",
        }),
    ]
    assert (
        "matrix_followup" in _get_platform_tools({}, "matrix"),
        "matrix_followup" in _get_platform_tools({}, "telegram"),
    ) == (True, False)
