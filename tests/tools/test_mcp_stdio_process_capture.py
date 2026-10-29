"""SDK spawn observations stay with their task and preserve native cleanup."""
import asyncio
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp.client import stdio

from tools import mcp_tool_process_capture as capture


@pytest.fixture
def sdk_factory(monkeypatch):
    calls = []

    async def spawn(*args, **kwargs):
        process = object()
        calls.append((args, kwargs, process))
        return process

    stop = AsyncMock()
    monkeypatch.setattr(stdio, "_create_platform_compatible_process", spawn)
    monkeypatch.setattr(stdio, "_stop_server_process", stop)
    return calls, stop


def test_factory_installs_once_and_leaves_unscoped_spawns_unobserved(sdk_factory):
    calls, stop = sdk_factory
    seen = []

    async def run():
        with capture.capture_stdio_process(seen.append):
            installed = stdio._create_platform_compatible_process
            with capture.capture_stdio_process(seen.append):
                assert stdio._create_platform_compatible_process is installed
                owned = await installed("synthetic-command", cwd="synthetic-directory")
        outside = await installed("another-synthetic-command")
        assert seen == [owned]
        assert outside is not owned

    asyncio.run(run())
    assert len(calls) == 2
    assert calls[0][:2] == (("synthetic-command",), {"cwd": "synthetic-directory"})
    stop.assert_not_awaited()


def test_simultaneous_tasks_observe_only_their_own_returned_process(monkeypatch, sdk_factory):
    entered = 0
    observed = {"a": [], "b": []}
    returned = {}

    async def run():
        both_inside = asyncio.Event()

        async def spawn(command):
            nonlocal entered
            process = object()
            entered += 1
            if entered == 2:
                both_inside.set()
            await both_inside.wait()
            return process

        monkeypatch.setattr(stdio, "_create_platform_compatible_process", spawn)

        async def launch(name):
            with capture.capture_stdio_process(observed[name].append):
                returned[name] = await stdio._create_platform_compatible_process(name)

        await asyncio.wait_for(asyncio.gather(launch("a"), launch("b")), 5)

    asyncio.run(run())
    assert observed == {name: [process] for name, process in returned.items()}
    assert returned["a"] is not returned["b"]
    sdk_factory[1].assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_nested_scope_restores_outer_callback_after_failure(sdk_factory, failure):
    outer, inner = [], []

    async def run():
        with capture.capture_stdio_process(outer.append):
            first = await stdio._create_platform_compatible_process()
            with pytest.raises(failure):
                with capture.capture_stdio_process(inner.append):
                    second = await stdio._create_platform_compatible_process()
                    raise failure()
            third = await stdio._create_platform_compatible_process()
        await stdio._create_platform_compatible_process()
        assert outer == [first, third]
        assert inner == [second]

    asyncio.run(run())
    sdk_factory[1].assert_not_awaited()


def test_factory_failure_is_not_retried_or_cleaned_as_a_returned_process(monkeypatch, sdk_factory):
    attempts, observed = [], []

    async def spawn():
        attempts.append(True)
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", spawn)

    async def run():
        with pytest.raises(OSError, match="synthetic spawn failure"):
            with capture.capture_stdio_process(observed.append):
                await stdio._create_platform_compatible_process()
        assert capture._on_spawn.get() is None

    asyncio.run(run())
    assert attempts == [True]
    assert observed == []
    sdk_factory[1].assert_not_awaited()


def test_cancellation_during_factory_restores_context_without_claiming_a_process(monkeypatch, sdk_factory):
    observed, restored, attempts = [], [], []

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def spawn():
            attempts.append(True)
            entered.set()
            await release.wait()
            raise AssertionError("Cancelled factory must not finish")

        monkeypatch.setattr(stdio, "_create_platform_compatible_process", spawn)

        async def launch():
            try:
                with capture.capture_stdio_process(observed.append):
                    await stdio._create_platform_compatible_process()
            finally:
                restored.append(capture._on_spawn.get())

        task = asyncio.create_task(launch())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert attempts == [True]
    assert observed == []
    assert restored == [None]
    sdk_factory[1].assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_callback_failure_stops_exact_process_in_shield_without_retry(monkeypatch, sdk_factory, failure):
    calls, _ = sdk_factory
    stopped, callback_processes = [], []

    async def stop(process):
        assert anyio.current_effective_deadline() == float("inf")
        await anyio.lowlevel.checkpoint()
        stopped.append(process)

    monkeypatch.setattr(stdio, "_stop_server_process", stop)

    async def run():
        with anyio.CancelScope() as scope:
            def callback(process):
                callback_processes.append(process)
                scope.cancel()
                raise failure()

            with pytest.raises(failure):
                with capture.capture_stdio_process(callback):
                    await stdio._create_platform_compatible_process()
        assert capture._on_spawn.get() is None

    asyncio.run(run())
    assert len(calls) == 1
    assert callback_processes == stopped == [calls[0][2]]
