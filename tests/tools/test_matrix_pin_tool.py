"""Matrix pin actions require an opted-in Matrix session."""

import asyncio
import importlib
import json
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from gateway.session_identity import RoutingIdentity
from hermes_cli.tools_config import _get_platform_tools
from tools.interrupt import is_thread_interrupted, set_interrupt
from tools.registry import registry

importlib.import_module("tools.matrix_pin_tool")
matrix_tool_runtime = importlib.import_module("tools.matrix_tool_runtime")


def _identity(adapter, home):
    return RoutingIdentity("default", "default", home, home, multiplexed=False, transport=weakref.ref(adapter))


def _admin_config(home):
    from hermes_cli.config import atomic_config_write
    home.mkdir(parents=True, exist_ok=True)
    atomic_config_write(home / "config.yaml", {"platform_toolsets": {"matrix": ["matrix_admin"]}})


def _matrix_adapter(pins, send):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    class MissingState(RuntimeError):
        errcode = "M_NOT_FOUND"

    async def request(_method, path, *, query_params=None, **_kwargs):
        room_id, state = unquote(str(path)).removeprefix("/_matrix/client/v3/rooms/").split("/state/")
        event_type = state.partition("/")[0]
        if event_type == "m.room.pinned_events":
            return await pins(room_id, event_type)
        if event_type == "m.room.member":
            return {"membership": "join"}
        if event_type == "m.room.power_levels":
            return {"users": {"@alice:server": 50}}
        if event_type == "m.room.create" and query_params == {"format": "event"}:
            return {"type": "m.room.create", "sender": "@creator:server", "content": {"room_version": "10"}}
        raise MissingState()

    adapter = object.__new__(MatrixAdapter)
    adapter._pin_state_lock = asyncio.Lock()
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=request), send_state_event=send)
    adapter._joined_rooms = {"!room:server"}
    adapter._allowed_room_ids = set()
    adapter._user_id = "@bot:server"
    adapter._owner_profile = None
    adapter._allowed_room_ids = set()
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = lambda user, **kw: user == "@alice:server"
    return adapter


# Only a hung test reaches this bound; every wait below ends on an event.
_HANG_TIMEOUT = 30


def _observe_change(adapter, worker, stopped):
    """Record the worker's loop and task when it creates the pin change, and set *stopped* when the change ends."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    def change(*args, **kwargs):
        worker.loop, worker.task = asyncio.get_running_loop(), asyncio.current_task()

        async def run():
            try:
                return await MatrixAdapter.change_matrix_pin(adapter, *args, **kwargs)
            finally:
                stopped.set()

        return run()

    adapter.change_matrix_pin = change


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["pin", "unpin"])
@pytest.mark.parametrize("abort", ["interrupt", "timeout", "cancel"])
@pytest.mark.parametrize("stage", ["reading", "dispatch"])
async def test_aborted_pin_waits_until_owner_cannot_begin_a_write(
    action, abort, stage, tmp_path, monkeypatch,
):
    from agent.tool_executor import _registered_tool_worker
    from hermes_constants import (
        get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )

    owner_loop = asyncio.get_running_loop()
    owner_paused = asyncio.Event()
    owner_released = asyncio.Event()
    operation_finished = asyncio.Event()
    now = [0.0]
    monkeypatch.setattr(matrix_tool_runtime, "_monotonic", lambda: now[0])
    worker = SimpleNamespace()
    agent = SimpleNamespace(
        _tool_worker_threads=set(), _tool_worker_threads_lock=threading.Lock(),
    )
    observations = []
    writes = []
    profile_home = tmp_path / "served-profile"

    def release_if_worker_waits():
        # The worker loop runs this after the step in which the worker asked the
        # owner to stop, so the worker has either returned or is waiting.
        if not worker.task.done():
            owner_loop.call_soon_threadsafe(owner_released.set)

    async def hold_owner(on_release=lambda: None):
        owner_paused.set()
        try:
            await owner_loop.create_future()
        except asyncio.CancelledError:
            worker.loop.call_soon_threadsafe(release_if_worker_waits)
            await owner_released.wait()
            on_release()
            raise

    async def state(room_id, event_type):
        observations.append((asyncio.get_running_loop(), get_hermes_home(), room_id, event_type))
        if stage == "reading":
            await hold_owner()
        return {"pinned": ["$event"] if action == "unpin" else []}

    async def write(room_id, event_type, content):
        await hold_owner(lambda: writes.append((
            worker.task.done(), asyncio.get_running_loop(), get_hermes_home(), room_id, event_type, content,
        )))

    adapter = _matrix_adapter(state, write)
    _observe_change(adapter, worker, operation_finished)

    def dispatch():
        try:
            with _registered_tool_worker(agent) as worker.tid:
                return json.loads(registry.dispatch(
                    "matrix_pin", {"action": action, "event_id": "$event"},
                ))
        except asyncio.CancelledError:
            return "cancelled"
        finally:
            worker.owner_stopped_before_report = operation_finished.is_set()

    _admin_config(profile_home)
    home_token = set_hermes_home_override(profile_home)
    tokens = set_session_vars(
        platform="matrix", chat_id="!room:server", user_id="@alice:server",
        transport_adapter=adapter, transport_loop=asyncio.get_running_loop(),
        routing_identity=_identity(adapter, profile_home),
    )
    task = asyncio.create_task(asyncio.to_thread(dispatch))
    try:
        await asyncio.wait_for(owner_paused.wait(), timeout=_HANG_TIMEOUT)
        if abort == "cancel":
            worker.loop.call_soon_threadsafe(worker.task.cancel)
        if abort == "interrupt":
            set_interrupt(True, worker.tid)
        if abort == "timeout":
            now[0] += 31.0
        result = await asyncio.wait_for(task, timeout=_HANG_TIMEOUT)
    finally:
        owner_released.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        clear_session_vars(tokens)
        reset_hermes_home_override(home_token)

    expected = {
        "interrupt": {"error": "Matrix pin update interrupted"},
        "timeout": {"error": "Matrix pin update timed out"},
        "cancel": "cancelled",
    }
    if stage == "dispatch" and abort != "cancel":
        expected[abort] = {
            "error": f"{expected[abort]['error']} after the change was sent to the homeserver",
            "outcome": "unknown",
            "next_step": "Read the current pins with matrix_read kind=pins before retrying",
        }
    assert (result, worker.owner_stopped_before_report, is_thread_interrupted(worker.tid),
            agent._tool_worker_threads, observations, writes) == (
        expected[abort], True, False, set(),
        [(owner_loop, profile_home, "!room:server", "m.room.pinned_events")],
        [(False, owner_loop, profile_home, "!room:server", "m.room.pinned_events",
          {"pinned": [] if action == "unpin" else ["$event"]})] if stage == "dispatch" else [],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["pin", "unpin"])
@pytest.mark.parametrize("same_loop", [False, True])
async def test_repeated_cancellation_waits_for_owning_task_cleanup(action, same_loop):
    owner_loop = asyncio.get_running_loop()
    read_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    owner_stopped = asyncio.Event()
    worker = SimpleNamespace(before_release=None)

    async def state(room_id, event_type):
        read_started.set()
        try:
            await owner_loop.create_future()
        finally:
            cleanup_started.set()
            await cleanup_release.wait()

    adapter = _matrix_adapter(state, AsyncMock(return_value="$state"))
    _observe_change(adapter, worker, owner_stopped)

    def release_cleanup():
        worker.before_release = worker.task.done(), owner_stopped.is_set()
        owner_loop.call_soon_threadsafe(cleanup_release.set)

    def cancel_again_then_release_cleanup():
        # The cancellation wakes the worker before release_cleanup runs, so the
        # worker handles it while the owner is still cleaning up.
        worker.task.cancel()
        worker.loop.call_soon(release_cleanup)

    async def dispatch():
        args = {"action": action, "event_id": "$event"}
        if same_loop:
            return await registry.get_entry("matrix_pin").handler(args)
        return await asyncio.to_thread(registry.dispatch, "matrix_pin", args)

    from hermes_constants import get_hermes_home
    _admin_config(get_hermes_home())
    tokens = set_session_vars(
        platform="matrix", chat_id="!room:server", user_id="@alice:server",
        transport_adapter=adapter, transport_loop=asyncio.get_running_loop(),
        routing_identity=_identity(adapter, get_hermes_home()),
    )
    task = asyncio.create_task(dispatch())
    try:
        await asyncio.wait_for(read_started.wait(), timeout=_HANG_TIMEOUT)
        worker.loop.call_soon_threadsafe(worker.task.cancel)
        await asyncio.wait_for(cleanup_started.wait(), timeout=_HANG_TIMEOUT)
        before_second_cancel = worker.task.done(), owner_stopped.is_set()
        worker.loop.call_soon_threadsafe(cancel_again_then_release_cleanup)
        result, = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=_HANG_TIMEOUT)
    finally:
        cleanup_release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        clear_session_vars(tokens)

    assert (before_second_cancel, worker.before_release, type(result).__name__, owner_stopped.is_set()) == (
        (False, False), (False, False), "CancelledError", True,
    )
    adapter._client.send_state_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["pin", "unpin"])
@pytest.mark.parametrize("stage", ["queued", "reading", "cancel_queued"])
@pytest.mark.parametrize("same_loop", [False, True])
async def test_interrupted_pin_never_writes_after_pending_work_resumes(
    action, stage, same_loop, tmp_path,
):
    from hermes_constants import (
        get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )
    from plugins.platforms.matrix.adapter import MatrixAdapter

    owner_loop = asyncio.get_running_loop()
    pending = asyncio.Event()
    resume = asyncio.Event()
    cancelled = asyncio.Event()
    worker_ids = []
    observations = []
    profile_home = tmp_path / "served-profile"

    class PinLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                pending.set()
            try:
                return await super().acquire()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async def state(room_id, event_type):
        observations.append((asyncio.get_running_loop(), get_hermes_home(), room_id, event_type))
        if stage == "reading":
            pending.set()
            await resume.wait()
        return {"pinned": ["$event"] if action == "unpin" else []}

    adapter = _matrix_adapter(state, AsyncMock(return_value="$state"))
    adapter._pin_state_lock = PinLock()

    def dispatch():
        worker_ids.append(threading.get_ident())
        return registry.dispatch("matrix_pin", {"action": action, "event_id": "$event"})

    async def dispatch_on_owner_loop():
        worker_ids.append(threading.get_ident())
        return await registry.get_entry("matrix_pin").handler({"action": action, "event_id": "$event"})

    _admin_config(profile_home)
    home_token = set_hermes_home_override(profile_home)
    tokens = set_session_vars(
        platform="matrix", chat_id="!room:server", user_id="@alice:server",
        transport_adapter=adapter, transport_loop=asyncio.get_running_loop(),
        routing_identity=_identity(adapter, profile_home),
    )
    if stage != "reading":
        await adapter._pin_state_lock.acquire()
    task = asyncio.create_task(dispatch_on_owner_loop() if same_loop else asyncio.to_thread(dispatch))
    try:
        await pending.wait()
        set_interrupt(True, worker_ids[0])
        if stage == "cancel_queued":
            result = json.loads(await asyncio.wait_for(task, timeout=_HANG_TIMEOUT))
            await asyncio.wait_for(cancelled.wait(), timeout=_HANG_TIMEOUT)
        else:
            resume.set()
            if stage == "queued":
                adapter._pin_state_lock.release()
            result = json.loads(await task)
    finally:
        resume.set()
        if adapter._pin_state_lock.locked():
            adapter._pin_state_lock.release()
        await task
        for worker_id in worker_ids:
            set_interrupt(False, worker_id)
        clear_session_vars(tokens)
        reset_hermes_home_override(home_token)

    assert result == {"error": "Matrix pin update interrupted"}
    adapter._client.send_state_event.assert_not_awaited()
    expected_observations = [(owner_loop, profile_home, "!room:server", "m.room.pinned_events")]
    assert observations == (expected_observations if stage == "reading" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", ["event", "$"])
async def test_matrix_pin_uses_current_session_owner_and_validates_event_id(invalid_id):
    adapter = SimpleNamespace(change_matrix_pin=AsyncMock(return_value={
        "pinned": ["$event"], "state_event_id": "$state",
    }))
    tokens = set_session_vars(
        platform="matrix", chat_id="!room:server", user_id="@alice:server",
        transport_adapter=adapter, transport_loop=asyncio.get_running_loop(),
    )
    try:
        accepted = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_pin", {"action": "pin", "event_id": "$event"},
        ))
        invalid = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_pin", {"action": "pin", "event_id": invalid_id},
        ))
    finally:
        clear_session_vars(tokens)

    assert (accepted, invalid) == (
        {"pinned": ["$event"], "state_event_id": "$state"},
        {"error": "event_id must be a Matrix event ID"},
    )
    call = adapter.change_matrix_pin.await_args
    assert (call.args, call.kwargs["requester"], call.kwargs["interrupt_check"]()) == (
        ("pin", "!room:server", "$event"), "@alice:server", False,
    )
    assert adapter.change_matrix_pin.await_count == 1


def test_matrix_pin_is_restricted_to_explicit_matrix_admin_toolset():
    config = {"platform_toolsets": {
        "matrix": ["hermes-matrix", "matrix_admin"],
        "telegram": ["hermes-telegram", "matrix_admin"],
    }}
    assert (
        "matrix_admin" in _get_platform_tools({}, "matrix"),
        "matrix_admin" in _get_platform_tools(config, "matrix"),
        "matrix_admin" in _get_platform_tools(config, "telegram"),
    ) == (False, True, False)

    tokens = set_session_vars(platform="cli", chat_id="!room:server")
    try:
        result = json.loads(registry.dispatch("matrix_pin", {"action": "pin", "event_id": "$event"}))
    finally:
        clear_session_vars(tokens)

    assert result == {"error": "Matrix pin actions require a live Matrix session"}
