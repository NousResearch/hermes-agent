"""Poll tools use the receiving Matrix client and preserve room and actor gates."""

from __future__ import annotations

import asyncio
from functools import partial
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.tools_config import _checklist_toolset_keys, _get_platform_tools, _save_platform_tools
from plugins.platforms.matrix.poll_actions import matrix_poll_action
from plugins.platforms.matrix.polls import UNSTABLE
from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.reply_context import MatrixEventContextCache
from plugins.platforms.matrix.thread_context import history_entry
from tools.registry import registry
from toolsets import resolve_toolset

importlib.import_module("tools.matrix_poll_tool")

ROOM = "!poll:server"


def _dispatch_in_worker(tool, args):
    from model_tools import _get_worker_loop

    loop = _get_worker_loop()
    try:
        return registry.dispatch(tool, args)
    finally:
        loop.close()



async def dispatch(tool, args):
    result = await asyncio.to_thread(_dispatch_in_worker, tool, args)
    assert isinstance(result, str)
    return json.loads(result)


def poll_start(sender="@bot:server"):
    return {
        "room_id": ROOM, "event_id": "$poll", "sender": sender,
        "type": f"{UNSTABLE}start", "origin_server_ts": 50,
        "content": {f"{UNSTABLE}start": {
            "question": {"org.matrix.msc1767.text": "Which?"}, "kind": f"{UNSTABLE}disclosed",
            "answers": [{"id": "a", "org.matrix.msc1767.text": "A"}, {"id": "b", "org.matrix.msc1767.text": "B"}],
        }},
    }


def adapter_for(client, actor):
    from types import MethodType
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = SimpleNamespace(
        _allowed_room_ids=set(),
        _client=client, _user_id=actor, _joined_rooms={ROOM},
        _event_context_cache=MatrixEventContextCache(),
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: user == "@alice:server",
    )
    adapter._is_allowed_matrix_room = MethodType(
        MatrixAdapter._is_allowed_matrix_room, adapter,
    )
    adapter.matrix_poll_action = partial(matrix_poll_action, adapter)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("action,args", [
    ("create", {"question": "Which?", "answers": ["A", "B"]}),
    ("vote", {"poll_id": "$poll", "answers": ["b"]}),
    ("close", {"poll_id": "$poll"}),
    ("results", {"poll_id": "$poll"}),
])
async def test_registry_uses_each_receiving_adapter_and_native_sdk_types(action, args):
    from mautrix.errors import MNotFound
    from mautrix.types import EventType
    from mautrix.types.event.base import GenericEvent

    owning_loop = asyncio.get_running_loop()
    all_calls = []
    for actor in ("@bot:server", "@second:server", "@bot:server"):
        async def request(method, path, **kwargs):
            assert asyncio.get_running_loop() is owning_loop
            return poll_start(actor) if "/event/" in path else {"chunk": []}

        async def state(room_id, event_type):
            raise MNotFound(404, "Room is not encrypted")

        async def send(room_id, event_type, content):
            assert asyncio.get_running_loop() is owning_loop
            assert event_type.t_class is EventType.Class.MESSAGE
            raw: Any = {**poll_start(actor), "type": str(event_type), "content": content}
            serialized = GenericEvent.deserialize(raw).serialize()
            assert isinstance(serialized, dict)
            assert serialized["content"] == content
            all_calls.append((actor, str(event_type), content))
            return "$sent"

        client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)),
                                 get_state_event=state, send_message_event=send)
        adapter = adapter_for(client, actor)
        tokens = set_session_vars(platform="matrix", chat_id=ROOM, user_id="@alice:server", transport_adapter=adapter)
        try:
            result = await dispatch(f"matrix_poll_{action}", args)
            denied = await dispatch(f"matrix_poll_{action}", {**args, "room_id": "!other:server"})
        finally:
            clear_session_vars(tokens)
        assert "error" not in result, result
        assert denied == {"error": "Matrix polls are limited to the current room"}
        if action in {"vote", "close"}:
            assert result == {"poll_id": "$poll", "event_id": "$sent", "actor": actor, "action": action}
        if action == "create":
            assert {**result, "answers": [{**answer, "id": None} for answer in result["answers"]]} == {
                "poll_id": "$sent", "actor": actor, "answers": [{"id": None, "text": "A"}, {"id": None, "text": "B"}],
            }
    if action != "results":
        for actor, event_type, content in all_calls:
            assert event_type == f"{UNSTABLE}{'response' if action == 'vote' else 'end' if action == 'close' else 'start'}"
            if action != "create":
                assert content["m.relates_to"] == {"rel_type": "m.reference", "event_id": "$poll"}
    tool = f"matrix_poll_{action}"
    assert (any(tool in resolve_toolset(name) for name in _get_platform_tools({}, "matrix")),
            any(tool in resolve_toolset(name) for name in _get_platform_tools({}, "telegram"))) == (True, False)
    assert await dispatch(tool, args) == {"error": "Matrix polls require a live Matrix session"}


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing_keys", "truncated", "unauthorized", "invalid_answers", "open_undisclosed", "passive"])
async def test_results_and_writes_fail_closed_and_polls_remain_passive(problem):
    start = poll_start()
    if problem == "open_undisclosed":
        start["content"][f"{UNSTABLE}start"]["kind"] = f"{UNSTABLE}undisclosed"
    response = {
        "room_id": ROOM, "event_id": "$vote", "sender": "@alice:server", "origin_server_ts": 100,
        "type": f"{UNSTABLE}response", "content": {f"{UNSTABLE}response": {"answers": ["a"]},
        "m.relates_to": {"rel_type": "m.reference", "event_id": "$poll"}},
    }
    if problem == "missing_keys":
        response.update(type="m.room.encrypted", content={})

    async def request(method, path, **kwargs):
        if "/event/" in str(path):
            return start
        if "/m.annotation" in path:
            return {"chunk": []}
        return {"chunk": [response], **({"next_batch": "more"} if problem == "truncated" else {})}

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None,
                             send_message_event=AsyncMock())
    adapter = adapter_for(client, "@bot:server")
    tokens = set_session_vars(platform="matrix", chat_id=ROOM, user_id="@intruder:server" if problem == "unauthorized" else "@alice:server", transport_adapter=adapter)
    try:
        result = await dispatch("matrix_poll_results", {"poll_id": "$poll", "limit": 1})
        vote = await dispatch("matrix_poll_vote", {"poll_id": "$poll", "limit": 1, "answers": ["unknown"]})
    finally:
        clear_session_vars(tokens)
    client.send_message_event.assert_not_awaited()
    if problem == "unauthorized":
        assert result == vote == {"error": "Matrix requester is not authorized for this room"}
        client.api.request.assert_not_awaited()
        return
    if problem in {"missing_keys", "truncated", "open_undisclosed"}:
        assert ([answer["votes"] for answer in result["answers"]], result["voters"]) == ([None, None], None)
        assert result["complete"] is (problem == "open_undisclosed")
    assert "error" in vote
    if problem == "passive":
        visible = await read_matrix_context(adapter, "event", ROOM, "$poll", 1, requester="@alice:server")
        parsed = await history_entry(client, start, adapter._event_context_cache, ROOM)
        assert parsed is not None
        entry, _ = parsed
        assert visible == {"events": [{
            "event_id": "$poll", "sender": "@bot:server", "body": "[poll: Which?; answers: a: A; b: B]",
            "msgtype": None, "thread_id": None, "timestamp": 50, "sender_authorized": True,
        }], "errors": []}
        assert entry.text == visible["events"][0]["body"]


@pytest.mark.asyncio
async def test_encrypted_stable_poll_end_appears_in_reads_and_history():
    from mautrix.types.event.base import GenericEvent

    content = {"m.relates_to": {"rel_type": "m.reference", "event_id": "$poll"},
               "m.text": [{"body": "The poll has closed."}]}
    clear: Any = {"room_id": ROOM, "event_id": "$end", "sender": "@alice:server",
                  "origin_server_ts": 70, "type": "m.poll.end", "content": content}
    decrypted = GenericEvent.deserialize(clear)
    encrypted = {"room_id": ROOM, "event_id": "$end", "sender": "@alice:server", "origin_server_ts": 70,
                 "type": "m.room.encrypted", "content": {
                     "algorithm": "m.megolm.v1.aes-sha2", "ciphertext": "x", "session_id": "s",
                     "m.relates_to": content["m.relates_to"]}}

    async def request(method, path, **kwargs):
        return encrypted if "/event/" in str(path) else {"chunk": []}

    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)),
                             crypto=SimpleNamespace(decrypt_megolm_event=AsyncMock(return_value=decrypted)))
    adapter = adapter_for(client, "@bot:server")

    visible = await read_matrix_context(adapter, "event", ROOM, "$end", 1, requester="@alice:server")
    parsed = await history_entry(client, encrypted, MatrixEventContextCache(), ROOM)

    closure = "[poll end event; closure authority must be checked]"
    assert visible == {"events": [{
        "event_id": "$end", "sender": "@alice:server", "body": closure,
        "msgtype": None, "thread_id": None, "timestamp": 70, "sender_authorized": True,
    }], "errors": []}
    assert parsed is not None and parsed[0].text == closure


def test_tool_discovery_does_not_load_the_matrix_adapter():
    probe = ("import sys; from tools.registry import discover_builtin_tools; discover_builtin_tools(); "
             "print('plugins.platforms.matrix.adapter' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True,
                            env=env, cwd=Path(__file__).resolve().parents[2])
    assert result.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.parametrize("selection,expected", [({"web"}, False), ({"web", "matrix_polls"}, True)])
def test_hermes_tools_selection_controls_matrix_polls_on_matrix_only(selection, expected):
    config: dict[str, Any] = {}
    with patch("hermes_cli.tools_config.save_config"):
        for platform in ("matrix", "telegram"):
            _save_platform_tools(config, platform, selection)

    assert (
        ("matrix_polls" in _checklist_toolset_keys("matrix"), "matrix_polls" in _checklist_toolset_keys("telegram")),
        "matrix_polls" in _get_platform_tools({}, "matrix"),
        ("matrix_polls" in _get_platform_tools(config, "matrix"), "matrix_polls" in _get_platform_tools(config, "telegram")),
    ) == ((True, False), True, (expected, False))
