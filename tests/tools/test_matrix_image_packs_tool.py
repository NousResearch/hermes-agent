"""Image packs resolve from the current Matrix session without process gates."""

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.tools_config import _get_platform_tools
from tools.registry import registry

importlib.import_module("tools.matrix_image_packs_tool")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,args", [("list", {}), ("send", {"selection_id": "selected"})]
)
async def test_registry_dispatch_preserves_owner_loop_and_current_route(action, args):
    loop = asyncio.get_running_loop()
    calls = []

    async def packs(*positional, **kwargs):
        calls.append((positional, kwargs, asyncio.get_running_loop() is loop))
        return {"success": True}

    adapter = SimpleNamespace(matrix_image_packs=packs)
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!room:server",
        user_id="@alice:server",
        thread_id="$root",
        message_id="$reply",
        transport_adapter=adapter,
        transport_loop=loop,
    )
    try:
        result = await asyncio.to_thread(
            registry.dispatch, "matrix_image_packs", {"action": action, **args}
        )
    finally:
        clear_session_vars(tokens)
    assert isinstance(result, str)
    assert json.loads(result) == {"success": True}
    assert calls == [
        (
            (action, "!room:server"),
            {
                "requester": "@alice:server",
                "selection_id": args.get("selection_id"),
                "thread_id": "$root",
                "reply_to": "$reply",
            },
            True,
        )
    ]
    assert (
        "matrix_image_packs" in _get_platform_tools({}, "matrix"),
        "matrix_image_packs" in _get_platform_tools({}, "telegram"),
    ) == (True, False)


@pytest.mark.parametrize("platform,action", [("cli", "list"), ("matrix", "send")])
def test_dispatch_requires_live_session_and_valid_selection(platform, action):
    import model_tools

    original_tool_loop = model_tools._tool_loop
    tokens = set_session_vars(
        platform=platform, chat_id="!room:server", user_id="@alice:server"
    )
    try:
        raw = registry.dispatch("matrix_image_packs", {"action": action})
        assert isinstance(raw, str)
        result = json.loads(raw)
    finally:
        clear_session_vars(tokens)
        tool_loop = model_tools._tool_loop
        if tool_loop is not None and tool_loop is not original_tool_loop:
            tool_loop.close()
        model_tools._tool_loop = original_tool_loop
    assert result == {"error": "Image packs require a live Matrix session"}
