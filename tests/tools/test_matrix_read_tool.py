"""Matrix reads use the live session's receiving adapter and its policy."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.relay.adapter import RelayAdapter
from gateway.session_context import clear_session_vars, get_session_transport, set_session_vars
from hermes_cli.tools_config import _get_platform_tools
from tools.registry import registry

matrix_read_tool = importlib.import_module("tools.matrix_read_tool")


def _bind_matrix_session(adapter, **overrides):
    values = dict(platform="matrix", chat_id="!room:server", user_id="@alice:server",
                  transport_adapter=adapter, transport_loop=asyncio.get_running_loop())
    values.update(overrides)
    return set_session_vars(**values)


@pytest.mark.asyncio
async def test_matrix_read_uses_session_owner_and_room():
    adapter = SimpleNamespace(
        read_matrix_context=AsyncMock(return_value={"events": [{"event_id": "$one", "body": "hello"}]})
    )
    tokens = _bind_matrix_session(adapter, thread_id="$root", session_key="matrix-session")
    try:
        result = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_read", {"kind": "room", "limit": 5},
        ))
    finally:
        clear_session_vars(tokens)

    assert result == {"events": [{"event_id": "$one", "body": "hello"}]}
    assert get_session_transport() == (None, None)
    adapter.read_matrix_context.assert_awaited_once_with(
        "room", "!room:server", None, 5, requester="@alice:server",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("args", "error"), [
    ({}, "kind must be room, thread, or event"),
    ({"kind": "search"}, "kind must be room, thread, or event"),
    ({"kind": "thread"}, "event_id is required for thread and event reads"),
    ({"kind": "event"}, "event_id is required for thread and event reads"),
    ({"kind": "event", "event_id": "not-an-event"}, "event_id is required for thread and event reads"),
    ({"kind": "room", "limit": 0}, "limit must be between 1 and 50"),
    ({"kind": "room", "limit": 51}, "limit must be between 1 and 50"),
    ({"kind": "room", "limit": True}, "limit must be between 1 and 50"),
    ({"kind": "room", "limit": "5"}, "limit must be between 1 and 50"),
], ids=["no-kind", "unknown-kind", "thread-without-root", "event-without-id", "event-id-without-sigil",
        "limit-zero", "limit-over-maximum", "limit-boolean", "limit-string"])
async def test_matrix_read_rejects_invalid_arguments_before_reading(args, error):
    adapter = SimpleNamespace(read_matrix_context=AsyncMock(return_value={"events": []}))
    tokens = _bind_matrix_session(adapter)
    try:
        result = json.loads(await asyncio.to_thread(registry.dispatch, "matrix_read", args))
    finally:
        clear_session_vars(tokens)

    assert result == {"error": error}
    adapter.read_matrix_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("session", [
    {"platform": "cli", "transport_adapter": None},
    {"transport_adapter": object.__new__(RelayAdapter)},
], ids=["cli", "relay-fronted-matrix"])
async def test_matrix_read_requires_live_matrix_session(session):
    tokens = _bind_matrix_session(None, **session)
    try:
        result = json.loads(await asyncio.to_thread(registry.dispatch, "matrix_read", {"kind": "room"}))
    finally:
        clear_session_vars(tokens)

    assert result == {"error": "Matrix reads require a live Matrix session"}


def test_matrix_read_is_only_in_matrix_default_toolset():
    assert ("matrix_read" in _get_platform_tools({}, "matrix"),
            "matrix_read" in _get_platform_tools({}, "telegram")) == (True, False)


@pytest.mark.asyncio
async def test_matrix_read_runs_on_owning_gateway_loop():
    owner_loop = asyncio.get_running_loop()

    async def read(*args, **kwargs):
        return {"on_owner_loop": asyncio.get_running_loop() is owner_loop}

    tokens = _bind_matrix_session(SimpleNamespace(read_matrix_context=read))
    try:
        result = await asyncio.to_thread(registry.dispatch, "matrix_read", {"kind": "room"})
    finally:
        clear_session_vars(tokens)

    assert json.loads(result) == {"on_owner_loop": True}


@pytest.mark.asyncio
async def test_matrix_read_refuses_a_stopped_owner_loop():
    adapter = SimpleNamespace(read_matrix_context=AsyncMock(return_value={"events": []}))
    stopped_loop = asyncio.new_event_loop()
    tokens = _bind_matrix_session(adapter, transport_loop=stopped_loop)
    try:
        result = await asyncio.to_thread(registry.dispatch, "matrix_read", {"kind": "room"})
    finally:
        clear_session_vars(tokens)
        stopped_loop.close()

    assert json.loads(result) == {"error": "Matrix gateway loop is unavailable"}
    adapter.read_matrix_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_matrix_read_deadline_cancels_a_stalled_read(monkeypatch):
    monkeypatch.setattr(matrix_read_tool, "_READ_DEADLINE_SECONDS", 0)
    owner_loop = asyncio.get_running_loop()
    reading = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()
    outcome = []
    schedule = matrix_read_tool.safe_schedule_threadsafe

    # A read cancelled before the gateway loop starts it never runs, so the
    # deadline must not begin until the read is waiting.
    def schedule_and_wait_until_reading(*args, **kwargs):
        future = schedule(*args, **kwargs)
        asyncio.run_coroutine_threadsafe(reading.wait(), owner_loop).result()
        return future

    monkeypatch.setattr(matrix_read_tool, "safe_schedule_threadsafe", schedule_and_wait_until_reading)

    async def stalled_read(*args, **kwargs):
        reading.set()
        try:
            await release.wait()
            outcome.append("completed")
            return {"events": []}
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        finally:
            settled.set()

    tokens = _bind_matrix_session(SimpleNamespace(read_matrix_context=stalled_read))
    try:
        dispatch = asyncio.create_task(asyncio.to_thread(registry.dispatch, "matrix_read", {"kind": "room"}))
        started = asyncio.create_task(reading.wait())
        await asyncio.wait({dispatch, started}, return_when=asyncio.FIRST_COMPLETED)
        started.cancel()
        await asyncio.wait({dispatch}, timeout=10)
        release.set()
        result = json.loads(await dispatch)
        if reading.is_set():
            await settled.wait()
    finally:
        clear_session_vars(tokens)

    assert (result, outcome) == ({"error": "Matrix read timed out"}, ["cancelled"])
