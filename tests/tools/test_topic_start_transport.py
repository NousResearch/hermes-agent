"""Topic kickoff keeps the originating bot across multiplexed runtime routes."""
import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway import session_context
from gateway.session_identity import identity_of
from plugins.platforms.telegram import topic_tool
from tests.gateway.test_multiplex_transport_matrix import _live

pytest_plugins = ("tests.gateway.test_multiplex_transport_matrix",)


@pytest.mark.parametrize(
    "receiver,chat_id,offline",
    [("team_b", "1", False), ("primary", "555", False),
     ("team_b", "999", False), ("team_b", "1", True)],
)
def test_topic_kickoff_preserves_origin_transport(mux, monkeypatch, receiver, chat_id, offline):
    runner = mux.runner
    adapter = getattr(mux, receiver)
    source, identity = _live(
        mux, adapter, chat_id,
        **({"transport_profile": "team_b"} if receiver == "team_b" else {}),
    )
    runner.session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _entries={"current": SimpleNamespace(origin=source, transport_profile=identity.transport_profile)},
    )
    runner._gateway_loop = None
    admitted = []

    async def accept(event):
        event._gateway_accepted = True
        admitted.append(event)

    for bot in (mux.primary, mux.team_b):
        bot.create_handoff_thread = AsyncMock(return_value="444")
        bot.handle_message = accept
    if offline:
        runner._profile_adapters["team_b"] = {}
        runner._profile_failed_platforms = {"team_b": {source.platform: {}}}
        runner.session_store._entries["current"].origin = replace(source)
    values = {
        "HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_KEY": "current",
        "HERMES_SESSION_CHAT_ID": chat_id, "HERMES_SESSION_USER_ID": source.user_id,
        "HERMES_SESSION_PROFILE": identity.runtime_profile,
    }
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": values.get(key, default))
    monkeypatch.setattr(topic_tool, "_run_on_gateway_loop", lambda coro, loop: asyncio.run(coro))

    result = json.loads(topic_tool.telegram_topic_start({"topic_name": "Continued task", "prompt": "Continue"}))

    if offline:
        assert result["success"] is False
        mux.primary.create_handoff_thread.assert_not_awaited()
        mux.team_b.create_handoff_thread.assert_not_awaited()
        assert not admitted
    else:
        assert result["success"] is True
        adapter.create_handoff_thread.assert_awaited_once_with(chat_id, "Continued task")
        other = mux.team_b if adapter is mux.primary else mux.primary
        other.create_handoff_thread.assert_not_awaited()
        assert len(admitted) == 1
        carried = identity_of(admitted[0].source)
        assert carried is not None
        assert (carried.transport_profile, carried.runtime_profile) == (
            identity.transport_profile, identity.runtime_profile,
        )
        assert admitted[0].source.thread_id == "444"
        assert admitted[0].delegated_continuation is True
        assert admitted[0].internal is False
