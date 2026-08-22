"""test_yuanbao_task_lifecycle.py - Yuanbao background-task lifecycle.

Two related defects in the fire-and-forget task lifecycle of the Yuanbao
adapter, both of which leave a task unowned at a lifecycle boundary:

* ``ConnectionManager.schedule_reconnect`` used a bare ``asyncio.create_task``.
  The event loop keeps only a weak reference to such a task, so a pending
  reconnect could be garbage-collected mid-flight — after which the adapter
  never reconnects and the bot is silently offline until the gateway restarts.
  ``YuanbaoAdapter._track_task`` exists for exactly this ("Register a
  fire-and-forget task so it won't be GC'd prematurely") and is already used
  for the inbound pipeline in the same file.

* ``YuanbaoAdapter.disconnect`` is documented as "Cancel background tasks and
  close the WebSocket connection", but only ever cancelled ``_inbound_tasks``.
  Everything registered through ``_track_task`` — the inbound pipeline, the
  recall-redaction job, and now the reconnect — landed in ``_background_tasks``,
  whose only removal path is the per-task done callback. Those tasks were
  therefore abandoned mid-flight at teardown, and an in-flight reconnect could
  outlive ``disconnect()`` and revive a deliberately stopped adapter.
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from gateway.config import PlatformConfig
from gateway.platforms.yuanbao import YuanbaoAdapter


def make_config(**kwargs):
    extra = kwargs.pop("extra", {})
    extra.setdefault("app_id", "test_key")
    extra.setdefault("app_secret", "test_secret")
    extra.setdefault("ws_url", "wss://test.example.com/ws")
    extra.setdefault("api_domain", "https://test.example.com")
    return PlatformConfig(extra=extra, **kwargs)


def _adapter() -> YuanbaoAdapter:
    """A real adapter with the status-writing teardown hooks neutralised."""
    adapter = YuanbaoAdapter(make_config())
    adapter._mark_disconnected = lambda: None
    adapter._release_platform_lock = lambda: None
    return adapter


# ---------------------------------------------------------------------------
# schedule_reconnect — the task must be strongly referenced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_reconnect_anchors_task_against_gc():
    """The pending reconnect must be reachable from _background_tasks."""
    adapter = _adapter()
    adapter._running = True
    cm = adapter._connection

    with patch.object(cm, "_reconnect_with_backoff", new_callable=AsyncMock, return_value=True):
        cm.schedule_reconnect()

        anchored = [t for t in adapter._background_tasks if t.get_name() == "yuanbao-reconnect"]
        assert len(anchored) == 1, (
            "reconnect task is not strongly referenced — the loop may GC it mid-flight"
        )

        await asyncio.gather(*anchored)

    # _track_task's done callback releases the reference once it completes.
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
async def test_schedule_reconnect_is_a_noop_when_adapter_stopped():
    """Anchoring must not weaken the existing _running guard."""
    adapter = _adapter()
    adapter._running = False
    cm = adapter._connection

    with patch.object(cm, "_reconnect_with_backoff", new_callable=AsyncMock) as reconnect:
        cm.schedule_reconnect()

    assert adapter._background_tasks == set()
    reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_reconnect_is_a_noop_while_already_reconnecting():
    """Anchoring must not weaken the existing _reconnecting guard."""
    adapter = _adapter()
    adapter._running = True
    cm = adapter._connection
    cm._reconnecting = True

    with patch.object(cm, "_reconnect_with_backoff", new_callable=AsyncMock) as reconnect:
        cm.schedule_reconnect()

    assert adapter._background_tasks == set()
    reconnect.assert_not_called()


# ---------------------------------------------------------------------------
# disconnect — background tasks must be drained, not abandoned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_drains_in_flight_background_tasks():
    """disconnect() must cancel AND await every _track_task job."""
    adapter = _adapter()
    started = asyncio.Event()
    unwound = asyncio.Event()

    async def _long_running():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            unwound.set()
            raise

    task = adapter._track_task(
        asyncio.create_task(_long_running(), name="yuanbao-test-background")
    )
    await started.wait()

    await adapter.disconnect()

    assert task.done(), "background task was abandoned by teardown"
    assert task.cancelled()
    # The await is the point: without it disconnect() returns before the task
    # has had a chance to unwind, so cleanup in its except/finally never runs.
    assert unwound.is_set(), "disconnect() returned before the task unwound"
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
async def test_disconnect_still_cancels_inbound_tasks():
    """The pre-existing _inbound_tasks teardown must be unaffected."""
    adapter = _adapter()
    started = asyncio.Event()

    async def _inbound():
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(_inbound(), name="yuanbao-test-inbound")
    adapter._inbound_tasks.add(task)
    task.add_done_callback(adapter._inbound_tasks.discard)
    await started.wait()

    await adapter.disconnect()

    assert adapter._inbound_tasks == set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_disconnect_survives_a_background_task_that_raises():
    """One misbehaving task must not abort teardown of the others."""
    adapter = _adapter()
    running = asyncio.Event()
    other_unwound = asyncio.Event()

    async def _raises_on_cancel():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise RuntimeError("cleanup blew up")

    async def _well_behaved():
        # Scheduled after _raises_on_cancel, so both are in flight once this
        # event is set.
        running.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            other_unwound.set()
            raise

    bad = adapter._track_task(asyncio.create_task(_raises_on_cancel(), name="yuanbao-test-bad"))
    good = adapter._track_task(asyncio.create_task(_well_behaved(), name="yuanbao-test-good"))
    await running.wait()

    await adapter.disconnect()

    assert bad.done() and good.done()
    assert other_unwound.is_set()
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
async def test_disconnect_from_inside_a_tracked_task_does_not_await_itself():
    """A shutdown driven from a tracked task must not deadlock on itself."""
    adapter = _adapter()
    finished = asyncio.Event()

    async def _self_shutdown():
        await adapter.disconnect()
        finished.set()

    task = adapter._track_task(
        asyncio.create_task(_self_shutdown(), name="yuanbao-test-self-shutdown")
    )

    await asyncio.wait_for(task, timeout=5)

    assert finished.is_set()
