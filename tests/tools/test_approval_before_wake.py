"""Pre-wake UI state changes must not lose or prematurely release approvals."""

import asyncio
from unittest.mock import Mock

import pytest

from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.asyncio
@pytest.mark.parametrize("same_surface", [False, True])
async def test_before_wake_groups_waits_across_sessions(same_surface):
    loop = asyncio.get_running_loop()
    ready = [asyncio.Event(), asyncio.Event()]
    keys = ["grouped-approval-one", "grouped-approval-two"]
    groups = [(object(), "77"), None]
    groups[1] = groups[0] if same_surface else (object(), "77")
    workers = []
    try:
        for i, key in enumerate(keys):
            def notify(data, event=ready[i]):
                loop.call_soon_threadsafe(event.set)
            approval.register_gateway_notify(key, notify, wait_group=groups[i])
            workers.append(asyncio.create_task(asyncio.to_thread(
                _await_gateway_decision, key, notify,
                {"command": f"command-{i}", "pattern_key": "test"},
            )))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 5)
        resumed = Mock()
        assert approval.resolve_gateway_approval(keys[0], "deny", before_wake=resumed) == 1
        assert resumed.call_count == (0 if same_surface else 1)
        resumed.reset_mock()
        assert approval.resolve_gateway_approval(keys[1], "deny", before_wake=resumed) == 1
        resumed.assert_called_once_with()
        await asyncio.wait_for(asyncio.gather(*workers), 5)
    finally:
        for key in keys:
            approval.unregister_gateway_notify(key)
        await asyncio.wait_for(asyncio.gather(*workers), 5)


@pytest.mark.asyncio
async def test_before_wake_failure_preserves_pending_decision():
    key = "approval-before-wake"
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    worker = asyncio.create_task(asyncio.to_thread(
        _await_gateway_decision, key, lambda data: loop.call_soon_threadsafe(ready.set),
        {"command": "test-command", "pattern_key": "test"},
    ))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        unmatched = Mock()
        assert approval.resolve_gateway_approval(
            key, "once", request_id="missing", before_wake=unmatched) == 0
        unmatched.assert_not_called()
        callback = Mock(side_effect=RuntimeError("UI state unavailable"))
        with pytest.raises(RuntimeError, match="UI state unavailable"):
            approval.resolve_gateway_approval(key, "once", before_wake=callback)
        assert approval.has_blocking_approval(key)
        assert not worker.done()
        callback.assert_called_once_with()
        before = Mock()
        assert approval.resolve_gateway_approval(key, "deny", before_wake=before) == 1
        before.assert_called_once_with()
        decision = await asyncio.wait_for(asyncio.shield(worker), 5)
        assert decision["choice"] == "deny"
        absent = Mock()
        assert approval.resolve_gateway_approval(key, "once", before_wake=absent) == 0
        absent.assert_not_called()
    finally:
        approval.resolve_gateway_approval(key, "deny", resolve_all=True)
        await asyncio.wait_for(worker, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["all", "request-id"])
async def test_before_wake_precedes_selected_workers(selection):
    key = "approval-selected-workers"
    ready = asyncio.Queue()
    loop = asyncio.get_running_loop()
    resumed = []

    def notify(data):
        loop.call_soon_threadsafe(ready.put_nowait, data["request_id"])

    async def wait(command):
        def work():
            result = _await_gateway_decision(key, notify, {"command": command})
            return result, bool(resumed)
        return await asyncio.to_thread(work)

    workers = [asyncio.create_task(wait(command)) for command in ("first", "second")]
    try:
        ids = [await asyncio.wait_for(ready.get(), 5) for _ in workers]
        callback = lambda: resumed.append(True)
        if selection == "request-id":
            assert approval.resolve_gateway_approval(key, "deny", request_id=ids[1], before_wake=callback) == 1
            assert not resumed
            assert approval.resolve_gateway_approval(key, "once", request_id=ids[0], before_wake=callback) == 1
        else:
            assert approval.resolve_gateway_approval(key, "once", resolve_all=True, before_wake=callback) == 2
        results = await asyncio.wait_for(asyncio.gather(*workers), 5)
        assert len(resumed) == 1
        assert all(was_resumed for result, was_resumed in results if result["choice"] == "once")
    finally:
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(asyncio.gather(*workers), 5)
