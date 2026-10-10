"""Matrix discovery is available per session and uses its live transport."""

import asyncio
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.tools_config import _get_platform_tools
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.registry import registry

importlib.import_module("tools.matrix_read_tool")


def test_builtin_discovery_registers_matrix_reads_without_loading_the_adapter():
    code = """
import json
import sys
from tools.registry import discover_builtin_tools, registry

modules = discover_builtin_tools()
print(json.dumps({
    "discovered": "tools.matrix_read_tool" in modules,
    "registered": registry.get_entry("matrix_read") is not None,
    "matrix_modules": sorted(
        module for module in sys.modules
        if module == "plugins.platforms.matrix"
        or module.startswith("plugins.platforms.matrix.")
    ),
    "result": json.loads(registry.dispatch("matrix_read", {"kind": "joined_rooms"})),
}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(process.stdout) == {
        "discovered": True,
        "registered": True,
        "matrix_modules": [],
        "result": {"error": "Matrix reads require a live Matrix session"},
    }


async def _dispatch(arguments: dict[str, object]) -> dict[str, object]:
    raw = await asyncio.to_thread(registry.dispatch, "matrix_read", arguments)
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert isinstance(result, dict)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "list joined Matrix rooms",
        "list joined Matrix Spaces",
        "search Matrix user directory",
    ],
)
async def test_matrix_discovery_schema_and_dispatch_need_no_process_matrix_flags(
    monkeypatch,
    query,
):
    from model_tools import get_tool_definitions, handle_function_call
    from toolsets import _HERMES_CORE_TOOLS

    for key in (
        "HERMES_PLATFORM",
        "MATRIX_HOMESERVER",
        "MATRIX_ACCESS_TOKEN",
        "MATRIX_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    definitions = get_tool_definitions(
        enabled_toolsets=sorted(_get_platform_tools({}, "matrix")),
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    schema = next(
        item["function"]
        for item in definitions
        if item["function"]["name"] == "matrix_read"
    )
    search = json.loads(
        await asyncio.to_thread(
            handle_function_call,
            "tool_search",
            {"queries": [query]},
            enabled_toolsets=sorted(_get_platform_tools({}, "matrix")),
        )
    )
    assert "matrix_read" in search["results"][0]["matches"]
    assert {"joined_rooms", "joined_spaces", "users"} <= set(
        schema["parameters"]["properties"]["kind"]["enum"]
    )
    assert "matrix_read" not in _HERMES_CORE_TOOLS
    assert "matrix_read" not in _get_platform_tools({}, "telegram")
    adapter = SimpleNamespace(
        discover_matrix=AsyncMock(
            return_value={"rooms": [], "truncated": False, "errors": []}
        ),
        read_matrix_context=AsyncMock(),
    )
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!origin:server",
        user_id="@alice:server",
        transport_adapter=adapter,
        transport_loop=asyncio.get_running_loop(),
    )
    try:
        result = await _dispatch({"kind": "joined_spaces", "limit": 3})
    finally:
        clear_session_vars(tokens)
    assert result == {"rooms": [], "truncated": False, "errors": []}
    adapter.discover_matrix.assert_awaited_once_with(
        "joined_spaces",
        "!origin:server",
        3,
        requester="@alice:server",
        search_term=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"kind": "users"},
        {"kind": "users", "search_term": " "},
        {"kind": "users", "search_term": "x" * 201},
        {"kind": "joined_rooms", "limit": 51},
        {"kind": "joined_rooms", "limit": True},
    ],
)
async def test_discovery_rejects_invalid_arguments_before_transport(args):
    adapter = SimpleNamespace(discover_matrix=AsyncMock(), read_matrix_context=AsyncMock())
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!origin:server",
        user_id="@alice:server",
        transport_adapter=adapter,
        transport_loop=asyncio.get_running_loop(),
    )
    try:
        result = await _dispatch(args)
    finally:
        clear_session_vars(tokens)
    assert "error" in result
    adapter.discover_matrix.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,user", [("cli", "@alice:server"), ("matrix", "")])
async def test_discovery_rejects_unknown_sessions(platform, user):
    adapter = SimpleNamespace(discover_matrix=AsyncMock(), read_matrix_context=AsyncMock())
    tokens = set_session_vars(
        platform=platform,
        chat_id="!origin:server",
        user_id=user,
        transport_adapter=adapter,
        transport_loop=asyncio.get_running_loop(),
    )
    try:
        result = await _dispatch({"kind": "joined_rooms"})
    finally:
        clear_session_vars(tokens)
    assert result == {"error": "Matrix reads require a live Matrix session"}
    adapter.discover_matrix.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_uses_live_client_and_profile_a_b_a(tmp_path):
    from agent import secret_scope
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    loop = asyncio.get_running_loop()
    adapters = {}
    cryptos = {}
    homes = {key: tmp_path / key for key in ("a", "b")}
    secret_scope.set_multiplex_active(True)
    try:
        for key, home in homes.items():
            home.mkdir()
            instance = MatrixAdapter(
                PlatformConfig(
                    enabled=True,
                    token=key,
                    extra={"homeserver": "https://server", "user_id": "@bot:server"},
                )
            )
            instance._user_id = "@bot:server"
            instance._joined_rooms = {"!origin:server", "!candidate:server"}
            instance._allowed_room_ids = {"!origin:server"}
            instance.set_authorization_check(lambda *args: True)

            def check_scope(_key=key):
                assert asyncio.get_running_loop() is loop
                assert get_hermes_home() == homes[_key]
                assert secret_scope.get_secret("MATRIX_ACCESS_TOKEN") == _key
                assert adapters[_key]._client.crypto is cryptos[_key]

            async def directory(*args, _key=key, _check=check_scope, **kwargs):
                _check()
                return {"results": [{"user_id": f"@{_key}:server"}], "limited": False}

            async def state(room, event_type, _key=key, _check=check_scope):
                _check()
                return {"name": f"{_key} room"} if event_type == "m.room.name" else {}

            async def members(room, _key=key, _check=check_scope):
                _check()
                profiles = {"@bot:server": {}, "@alice:server": {}, f"@{_key}:server": {}}
                if room == "!candidate:server":
                    if _key == "a":
                        raise TimeoutError("!candidate:server access_token=secret")
                    profiles["@bob:server"] = {}
                return profiles

            cryptos[key] = object()
            instance._client = SimpleNamespace(
                api=SimpleNamespace(request=AsyncMock(side_effect=directory)),
                crypto=cryptos[key],
                get_state_event=AsyncMock(side_effect=state),
                get_joined_members=AsyncMock(side_effect=members),
                get_joined_rooms=AsyncMock(return_value=sorted(instance._joined_rooms)),
            )
            adapters[key] = instance

        results = []
        for key in ("a", "b", "a"):
            home_token = set_hermes_home_override(str(homes[key]))
            secret_token = secret_scope.set_secret_scope({"MATRIX_ACCESS_TOKEN": key})
            tokens = set_session_vars(
                platform="matrix",
                chat_id="!origin:server",
                user_id="@alice:server",
                profile=key,
                transport_adapter=adapters[key],
                transport_loop=loop,
            )
            try:
                results.append((
                    await _dispatch({"kind": "users", "search_term": key}),
                    await _dispatch({"kind": "joined_rooms"}),
                ))
            finally:
                clear_session_vars(tokens)
                secret_scope.reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        secret_scope.set_multiplex_active(False)
    assert results == [
        (
            {
                "users": [
                    {
                        "user_id": f"@{key}:server",
                        "display_name": None,
                        "avatar_url": None,
                    }
                ],
                "truncated": False,
                "errors": [],
            },
            {
                "rooms": [
                    {
                        "room_id": "!origin:server",
                        "room_type": None,
                        "name": f"{key} room",
                        "topic": None,
                        "canonical_alias": None,
                    }
                ],
                "truncated": key == "a",
                "errors": [{"error": "Matrix room classification is unavailable"}]
                if key == "a"
                else [],
            },
        )
        for key in ("a", "b", "a")
    ]
