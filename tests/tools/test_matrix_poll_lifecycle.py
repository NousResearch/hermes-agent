"""Poll operations retain their client and recheck access after network waits."""

from __future__ import annotations

import asyncio
from functools import partial
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from plugins.platforms.matrix.poll_actions import matrix_poll_action
from plugins.platforms.matrix.polls import UNSTABLE
from plugins.platforms.matrix.reply_context import MatrixEventContextCache
from tools.registry import registry

importlib.import_module("tools.matrix_poll_tool")

ROOM = "!poll:server"
BOT = "@bot:server"


def _dispatch_in_worker(tool, args):
    from model_tools import _get_worker_loop

    loop = _get_worker_loop()
    try:
        return registry.dispatch(tool, args)
    finally:
        loop.close()



@pytest.mark.asyncio
@pytest.mark.parametrize("action,wait_at", [
    ("create", "encryption"), ("results", "relations"),
    ("vote", "relations"), ("vote", "encryption"),
    ("close", "relations"), ("close", "encryption"),
])
@pytest.mark.parametrize("change", ["client", "actor", "closing", "room", "requester"])
async def test_access_changes_during_io_refuse_results_and_mutations(action, wait_at, change):
    from mautrix.errors import MNotFound

    entered = asyncio.Event()
    resume = asyncio.Event()
    start = {
        "room_id": ROOM, "event_id": "$poll", "sender": BOT,
        "type": f"{UNSTABLE}start", "origin_server_ts": 50,
        "content": {f"{UNSTABLE}start": {
            "question": {"org.matrix.msc1767.text": "Which?"},
            "kind": f"{UNSTABLE}disclosed",
            "answers": [{"id": "a", "org.matrix.msc1767.text": "A"},
                        {"id": "b", "org.matrix.msc1767.text": "B"}],
        }},
    }

    async def pause():
        entered.set()
        await resume.wait()

    async def request(method, path, **kwargs):
        if "/event/" in path:
            return start
        if wait_at == "relations":
            await pause()
        return {"chunk": []}

    async def state(room_id, event_type):
        if wait_at == "encryption":
            await pause()
        raise MNotFound(404, "Room is not encrypted")

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)),
                             get_state_event=state, send_message_event=AsyncMock(return_value="$sent"))
    replacement = SimpleNamespace(api=SimpleNamespace(request=AsyncMock()),
                                  get_state_event=state, send_message_event=AsyncMock(return_value="$other"))
    adapter = SimpleNamespace(
        _client=client, _user_id=BOT, _joined_rooms={ROOM}, _closing=False,
        _event_context_cache=MatrixEventContextCache(),
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kwargs: True,
    )
    adapter.matrix_poll_action = partial(matrix_poll_action, adapter)
    args = {"question": "Which?", "answers": ["A", "B"]} if action == "create" else {
        "poll_id": "$poll", "answers": ["a"],
    }
    tokens = set_session_vars(platform="matrix", chat_id=ROOM, user_id="@alice:server", transport_adapter=adapter,
                              transport_loop=asyncio.get_running_loop())
    try:
        pending = asyncio.create_task(asyncio.to_thread(_dispatch_in_worker, f"matrix_poll_{action}", args))
        await entered.wait()
        if change == "client":
            adapter._client = replacement
        elif change == "actor":
            adapter._user_id = "@replacement:server"
        elif change == "closing":
            adapter._closing = True
        elif change == "room":
            adapter._joined_rooms.clear()
        else:
            adapter._is_sender_authorized = lambda user, **kwargs: False
        resume.set()
        output = await pending
        assert isinstance(output, str)
        result = json.loads(output)
    finally:
        resume.set()
        clear_session_vars(tokens)

    expected = {
        "client": "Matrix client changed during the poll operation",
        "actor": "Matrix client changed during the poll operation",
        "closing": "Matrix client is disconnected",
        "room": "Matrix room is not allowed or joined",
        "requester": "Matrix requester is not authorized for this room",
    }
    assert result == {"error": expected[change]}
    client.send_message_event.assert_not_awaited()
    replacement.send_message_event.assert_not_awaited()
    replacement.api.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_checks_the_owning_profile_policy_across_thread_and_loop_hops(monkeypatch, tmp_path):
    from agent import secret_scope
    from gateway.config import PlatformConfig
    from gateway.platforms._shared import get_scoped_secret
    from gateway.run import _profile_runtime_scope
    from mautrix.errors import MNotFound
    from plugins.platforms.matrix.adapter import MatrixAdapter

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    adapters = {}
    for label, allowed in (("A", "@alice:server"), ("B", "@bob:server")):
        home = tmp_path / label
        home.mkdir()
        (home / ".env").write_text(f"MATRIX_ALLOWED_USERS={allowed}\n", encoding="utf-8")
        with _profile_runtime_scope(home):
            adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"user_id": BOT}))
            adapter._joined_rooms.add(ROOM)
            adapter._is_dm_room = AsyncMock(return_value=False)
            adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
            adapter.set_authorization_check(
                lambda user, *_args, **_kwargs: user in get_scoped_secret("MATRIX_ALLOWED_USERS", "").split(",")
            )

            async def state(room_id, event_type):
                raise MNotFound(404, "Room is not encrypted")

            adapter._client = SimpleNamespace(
                get_state_event=state, send_message_event=AsyncMock(return_value="$created"),
            )
            adapters[label] = adapter

    observed = []
    for label in ("A", "B", "A"):
        with _profile_runtime_scope(tmp_path / label):
            tokens = set_session_vars(
                platform="matrix", chat_id=ROOM, user_id="@alice:server", transport_adapter=adapters[label],
                transport_loop=asyncio.get_running_loop(),
            )
            try:
                output = await asyncio.to_thread(_dispatch_in_worker, "matrix_poll_create", {
                    "question": "Which?", "answers": ["A", "B"],
                })
                assert isinstance(output, str)
                result = json.loads(output)
                if "answers" in result:
                    result = {**result, "answers": [{**answer, "id": None} for answer in result["answers"]]}
                observed.append(result)
            finally:
                clear_session_vars(tokens)

    success = {"poll_id": "$created", "actor": BOT, "answers": [{"id": None, "text": "A"}, {"id": None, "text": "B"}]}
    assert observed == [success, {"error": "Matrix requester is not authorized for this room"}, success]
    assert adapters["A"]._client.send_message_event.await_count == 2
    adapters["B"]._client.send_message_event.assert_not_awaited()
