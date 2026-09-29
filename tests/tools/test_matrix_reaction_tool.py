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
        transport_loop=asyncio.get_running_loop(),
    )
    try:
        return json.loads(
            await asyncio.to_thread(registry.dispatch, "matrix_reaction", args)
        )
    finally:
        clear_session_vars(tokens)


class _LoopRecordingAdapter:
    """Records each adapter call and whether it ran on the session's gateway loop."""

    def __init__(self, owner_loop: asyncio.AbstractEventLoop):
        self.owner_loop = owner_loop
        self.calls: list[tuple] = []

    def _record(self, *call) -> None:
        self.calls.append((*call, asyncio.get_running_loop() is self.owner_loop))

    async def check_session_access(self, room_id, requester):
        self._record("check_session_access", room_id, requester)
        return SessionAccess(chat_type="group")

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self._record("add_reaction", chat_id, emoji, message_id)
        return {"success": True, "message_id": message_id}

    async def remove_reaction(self, chat_id, message_id=None):
        self._record("remove_reaction", chat_id, message_id)
        return {"success": True, "message_id": message_id}


@pytest.mark.asyncio
async def test_matrix_reaction_runs_on_the_receiving_adapters_gateway_loop():
    adapter = _LoopRecordingAdapter(asyncio.get_running_loop())

    react = await _dispatch_in_session(adapter, {"action": "react", "emoji": "👍"})
    unreact = await _dispatch_in_session(adapter, {"action": "unreact"})

    assert (react, unreact, adapter.calls) == (
        {"success": True, "message_id": "$current"},
        {"success": True, "message_id": "$current"},
        [
            ("check_session_access", ROOM, REQUESTER, True),
            ("add_reaction", ROOM, "👍", "$current", True),
            ("check_session_access", ROOM, REQUESTER, True),
            ("remove_reaction", ROOM, "$current", True),
        ],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["react", "unreact", "read"])
@pytest.mark.parametrize("policy", ["membership", "room", "requester"])
async def test_session_policy_is_current_after_identity_resolution(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    policy: str,
):
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from plugins.platforms.matrix.read_context import read_matrix_context

    room = "!room:server"
    requester = "@alice:server"
    adapter = object.__new__(MatrixAdapter)
    adapter._client = None
    adapter._reactions_enabled = False
    adapter._pending_reactions = {}
    adapter._agent_reactions = {(room, "$current"): ["$earlier"]}
    adapter._joined_rooms = {room}
    adapter._allowed_room_ids = {room}
    adapter._authorization_check = lambda *_args, **_kwargs: True
    send = AsyncMock(return_value="$reaction")
    redact = AsyncMock(return_value=True)
    monkeypatch.setattr(adapter, "_send_reaction", send)
    monkeypatch.setattr(adapter, "_redact_reaction", redact)

    async def resolve_identity(_room: str) -> bool:
        if policy == "membership":
            adapter._joined_rooms.clear()
        elif policy == "room":
            adapter._allowed_room_ids = {"!other:server"}
        else:
            adapter._authorization_check = lambda *_args, **_kwargs: False
        return False

    identity = AsyncMock(side_effect=resolve_identity)
    monkeypatch.setattr(adapter, "_is_dm_room", identity)
    if action == "read":
        result = await read_matrix_context(
            adapter,
            "room",
            room,
            None,
            1,
            requester=requester,
        )
    else:
        importlib.import_module("tools.matrix_reaction_tool")
        tokens = set_session_vars(
            platform="matrix",
            chat_id=room,
            user_id=requester,
            message_id="$current",
            transport_adapter=adapter,
            transport_loop=asyncio.get_running_loop(),
        )
        try:
            raw = await asyncio.to_thread(
                registry.dispatch,
                "matrix_reaction",
                {"action": action, "emoji": "👍"},
            )
            assert isinstance(raw, str)
            result = json.loads(raw)
        finally:
            clear_session_vars(tokens)

    error = (
        "Matrix requester is not authorized for this room"
        if policy == "requester"
        else "Matrix room is not allowed or joined"
    )
    assert (
        result,
        send.await_count,
        redact.await_count,
        adapter._agent_reactions,
        identity.await_count,
    ) == (
        {"error": error},
        0,
        0,
        {(room, "$current"): ["$earlier"]},
        1,
    )
