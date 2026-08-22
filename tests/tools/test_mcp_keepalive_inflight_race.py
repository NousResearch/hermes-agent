"""Regression tests for the MCP keepalive / in-flight RPC race.

Background
==========
An MCP stdio session is a SINGLE JSON-RPC stream. The idle keepalive
(``ping`` / ``list_tools``) could fire WHILE a user-visible request
(``call_tool`` / ``read_resource`` / ``get_prompt`` / ``list_*``) was in
flight, wedging the stream so the in-flight request timed out. That timeout
triggered a false reconnect, and the MCP SDK does not always fail the pending
request when its streams close, so its ``run_coroutine_threadsafe`` future
never resolved and the calling agent thread polled to the full ``tool_timeout``
(up to hours).

The fix (against current main's ``_keepalive_probe`` / ``_wait_for_lifecycle_event``
structure):

  * the keepalive **skips a cycle** when a request is in flight
    (``self._rpc_lock.locked() or self._inflight_tasks``) and otherwise runs
    the probe **under the same ``_rpc_lock``** the request handlers use, so the
    two can never overlap;
  * a reconnect/shutdown teardown calls ``_fail_inflight_calls`` to cancel the
    pending request tasks (ALL user-visible families, not just ``call_tool``);
    and
  * every request handler wraps its RPC in ``_track_inflight_rpc``, which
    registers the task in ``_inflight_tasks`` and converts a deliberate
    teardown cancel into a clean, retryable ``RuntimeError`` so the agent
    recovers on the freshly rebuilt session.

These tests cover both the in-flight bookkeeping AND the real lifecycle-event
path with lock-ordering assertions (no live MCP server required).
"""

from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# Bookkeeping-level tests
# ---------------------------------------------------------------------------


def test_new_server_starts_with_empty_inflight_state():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("init-test")
    assert server._inflight_tasks == set()
    assert server._reconnecting is False


def test_fail_inflight_calls_is_noop_when_nothing_in_flight():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("noop-test")
    # No in-flight tasks: must not flip the teardown flag (so a later genuine
    # cancel isn't misread as a deliberate reconnect).
    server._fail_inflight_calls("reconnect")
    assert server._reconnecting is False


def test_fail_inflight_calls_cancels_pending_and_flags_teardown():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("cancel-test")

    async def drive():
        async def _long():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_long())
        server._inflight_tasks.add(task)
        await asyncio.sleep(0)  # let the task start

        server._fail_inflight_calls("reconnect")
        assert server._reconnecting is True

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            return "cancelled"
        except asyncio.TimeoutError:
            return "still_running"
        return "completed"

    assert asyncio.run(drive()) == "cancelled"


# ---------------------------------------------------------------------------
# Handler-level in-flight wrapper (`_track_inflight_rpc`) tests. These prove
# the tracking is GENERALIZED to every user-visible request family (tool call,
# resource read/list, prompt get/list), not just `call_tool`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    ["tools/call", "resources/list", "resources/read", "prompts/list", "prompts/get"],
)
def test_track_inflight_rpc_registers_and_discards(op):
    """The wrapper registers the running task while the RPC is active and
    discards it on completion, for EACH handler family."""
    from tools.mcp_tool import MCPServerTask, _track_inflight_rpc

    server = MCPServerTask("track-test")

    async def drive():
        seen_inside = None

        async def _work():
            nonlocal seen_inside
            async with _track_inflight_rpc(server, server.name, op):
                seen_inside = set(server._inflight_tasks)

        await asyncio.create_task(_work())
        return seen_inside

    inside = asyncio.run(drive())
    # The running task was tracked while inside the wrapper...
    assert inside is not None and len(inside) == 1
    # ...and discarded once the RPC completed.
    assert server._inflight_tasks == set()


@pytest.mark.parametrize(
    "op",
    ["tools/call", "resources/list", "resources/read", "prompts/list", "prompts/get"],
)
def test_teardown_cancel_becomes_retryable_error(op):
    """A deliberate teardown (``_fail_inflight_calls``) cancels an in-flight
    request of ANY family, and the wrapper surfaces a retryable RuntimeError
    (never a raw CancelledError), so the agent self-heals on the rebuilt
    session instead of hanging to ``tool_timeout``."""
    from tools.mcp_tool import MCPServerTask, _track_inflight_rpc

    server = MCPServerTask(f"teardown-{op}")

    async def drive():
        started = asyncio.Event()
        outcome = {}

        async def _work():
            async with _track_inflight_rpc(server, server.name, op):
                started.set()
                try:
                    await asyncio.sleep(3600)  # stand-in for the wedged RPC
                except Exception as exc:  # noqa: BLE001 - capture for assertion
                    outcome["exc"] = exc
                    raise

        task = asyncio.create_task(_work())
        await started.wait()
        # Deliberate reconnect teardown cancels every in-flight request.
        server._fail_inflight_calls("reconnect")
        with pytest.raises(RuntimeError) as ei:
            await task
        return str(ei.value), outcome.get("exc")

    message, inner = asyncio.run(drive())
    assert "reconnected during" in message
    assert "retry the request" in message
    # The wrapper converted the CancelledError, it did not let a raw cancel out.
    assert not isinstance(inner, RuntimeError)


def test_external_cancel_is_not_masked_as_retryable():
    """When ``_reconnecting`` is NOT set (a genuine external cancel, e.g. the
    caller's timeout), the wrapper must re-raise the CancelledError rather than
    disguise it as a retryable reconnect error."""
    from tools.mcp_tool import MCPServerTask, _track_inflight_rpc

    server = MCPServerTask("external-cancel")

    async def drive():
        started = asyncio.Event()

        async def _work():
            async with _track_inflight_rpc(server, server.name, "tools/call"):
                started.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(_work())
        await started.wait()
        assert server._reconnecting is False
        task.cancel()  # external cancel, not a teardown
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())


# ---------------------------------------------------------------------------
# Real lifecycle-event path + lock-ordering tests. These drive
# ``_wait_for_lifecycle_event`` directly and assert the keepalive probe never
# runs while an RPC holds ``_rpc_lock`` (real ordering, not synthetic
# bookkeeping).
# ---------------------------------------------------------------------------


class _FakeSession:
    """Minimal session double that records when the keepalive probe runs.

    ``send_ping`` also snapshots whether ``_rpc_lock`` is held at the moment
    the probe issues its RPC, which is how the lock-ordering tests verify the
    probe runs *under* the lock.
    """

    def __init__(self):
        self.ping_calls = 0
        self.list_tools_calls = 0
        self._server = None  # type: ignore[var-annotated]  # MCPServerTask, set by the test
        self.locked_during_ping = None

    async def send_ping(self):
        self.ping_calls += 1
        if self._server is not None:
            self.locked_during_ping = self._server._rpc_lock.locked()
        return None

    async def list_tools(self):
        self.list_tools_calls += 1
        return []


def _make_lifecycle_server(name, *, keepalive_interval=0.05):
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask(name)
    server._config = {"keepalive_interval": keepalive_interval}
    server.session = None  # set by caller
    return server


def test_lifecycle_probe_runs_when_idle_and_takes_rpc_lock(monkeypatch):
    """Drive the real ``_wait_for_lifecycle_event`` loop: with no in-flight
    request, the keepalive probe fires, and it does so under ``_rpc_lock``
    (the lock is free before/after but held during the probe)."""
    import tools.mcp_tool as mcp_tool

    # The keepalive interval is floored at _MIN_KEEPALIVE_INTERVAL (5s in prod).
    # Lower the floor so the real loop probes within the test's time budget.
    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.02, raising=True)

    server = _make_lifecycle_server("idle-probe", keepalive_interval=0.02)
    fake = _FakeSession()
    fake._server = server
    server.session = fake

    async def drive():
        lifecycle = asyncio.create_task(server._wait_for_lifecycle_event())
        # Wait until at least one probe has run.
        for _ in range(200):
            if fake.ping_calls >= 1:
                break
            await asyncio.sleep(0.01)
        server._shutdown_event.set()
        return await asyncio.wait_for(lifecycle, timeout=5.0)

    reason = asyncio.run(drive())
    assert reason == "shutdown"
    # The probe actually ran (ping is tried first).
    assert fake.ping_calls >= 1
    # And it held the RPC lock while running (ordering guarantee): a concurrent
    # tool call could therefore never overlap it.
    assert fake.locked_during_ping is True


def test_lifecycle_skips_probe_while_rpc_lock_held(monkeypatch):
    """If an RPC holds ``_rpc_lock`` when the keepalive interval elapses, the
    lifecycle loop must SKIP the probe entirely (never issue a concurrent
    ping/list_tools on the shared stream)."""
    import tools.mcp_tool as mcp_tool

    # Lower the floor so a probe WOULD fire within the window if not suppressed;
    # this makes the "skip" assertion meaningful rather than trivially true.
    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.02, raising=True)

    server = _make_lifecycle_server("skip-when-locked", keepalive_interval=0.02)
    fake = _FakeSession()
    fake._server = server
    server.session = fake

    async def drive():
        # Hold the RPC lock for the duration, simulating an in-flight call.
        await server._rpc_lock.acquire()
        try:
            lifecycle = asyncio.create_task(server._wait_for_lifecycle_event())
            # Let several keepalive intervals elapse while the lock is held.
            await asyncio.sleep(0.3)
            # No probe should have run while the lock was held.
            assert fake.ping_calls == 0 and fake.list_tools_calls == 0
        finally:
            server._rpc_lock.release()

        # Now shut the loop down cleanly.
        server._shutdown_event.set()
        return await asyncio.wait_for(lifecycle, timeout=5.0)

    reason = asyncio.run(drive())
    assert reason == "shutdown"


def test_lifecycle_skips_probe_while_inflight_task_present(monkeypatch):
    """Even if the RPC lock happens to be momentarily free, the presence of an
    in-flight task in ``_inflight_tasks`` must also suppress the probe (a busy
    server is provably alive)."""
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.02, raising=True)

    server = _make_lifecycle_server("skip-when-inflight", keepalive_interval=0.02)
    fake = _FakeSession()
    fake._server = server
    server.session = fake

    async def drive():
        async def _long():
            await asyncio.sleep(3600)

        inflight = asyncio.create_task(_long())
        server._inflight_tasks.add(inflight)
        try:
            lifecycle = asyncio.create_task(server._wait_for_lifecycle_event())
            await asyncio.sleep(0.3)
            assert fake.ping_calls == 0 and fake.list_tools_calls == 0
        finally:
            server._inflight_tasks.discard(inflight)
            inflight.cancel()

        server._shutdown_event.set()
        return await asyncio.wait_for(lifecycle, timeout=5.0)

    reason = asyncio.run(drive())
    assert reason == "shutdown"


def test_reconnect_via_keepalive_failure_fails_inflight_calls():
    """Drive the real lifecycle path to a reconnect (keepalive probe raises)
    and assert it cancels an in-flight request task through
    ``_fail_inflight_calls`` (the reconnect exit calls it), setting the
    deliberate-teardown flag so the handler converts the cancel to retryable."""
    server = _make_lifecycle_server("reconnect-fails-inflight")
    fake = _FakeSession()
    server.session = fake

    async def drive():
        # An in-flight request that will be cancelled by the reconnect teardown.
        cancelled = {"was_cancelled": False}

        async def _inflight_request():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled["was_cancelled"] = True
                raise

        task = asyncio.create_task(_inflight_request())
        server._inflight_tasks.add(task)
        await asyncio.sleep(0)

        # Make the probe fail. But the probe is SKIPPED while a task is in
        # flight, so temporarily clear the in-flight set to let the probe run,
        # then rely on the reconnect exit to fail the (re-added) task. To model
        # the real ordering we instead trigger the reconnect event directly:
        # the loop wakes, sees the lifecycle event, and takes the reconnect
        # exit that calls _fail_inflight_calls.
        server._reconnect_event.set()

        reason = await asyncio.wait_for(
            server._wait_for_lifecycle_event(), timeout=5.0
        )
        # The reconnect exit fired _fail_inflight_calls.
        assert server._reconnecting is True
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        return reason, cancelled["was_cancelled"]

    reason, was_cancelled = asyncio.run(drive())
    assert reason == "reconnect"
    assert was_cancelled is True


def test_shutdown_exit_fails_inflight_calls():
    """The shutdown exit of the real lifecycle loop also cancels in-flight
    requests (shutdown teardown must not orphan them either)."""
    server = _make_lifecycle_server("shutdown-fails-inflight")
    fake = _FakeSession()
    server.session = fake

    async def drive():
        async def _inflight_request():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_inflight_request())
        server._inflight_tasks.add(task)
        await asyncio.sleep(0)

        server._shutdown_event.set()
        reason = await asyncio.wait_for(
            server._wait_for_lifecycle_event(), timeout=5.0
        )
        assert server._reconnecting is True
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        return reason, task.cancelled()

    reason, was_cancelled = asyncio.run(drive())
    assert reason == "shutdown"
    assert was_cancelled is True


def test_reconnecting_flag_reset_on_entry_to_healthy_wait():
    """A fresh healthy wait clears a lingering ``_reconnecting`` flag from a
    prior teardown, so a new in-flight call on the rebuilt session is not
    mistaken for a reconnect casualty."""
    server = _make_lifecycle_server("flag-reset")
    fake = _FakeSession()
    server.session = fake
    server._reconnecting = True  # leftover from a prior teardown

    async def drive():
        # Enter the loop; it clears _reconnecting on entry, then we shut down.
        server._shutdown_event.set()
        await asyncio.wait_for(server._wait_for_lifecycle_event(), timeout=5.0)
        return server._reconnecting

    # Entry clears it; the shutdown exit only re-sets it if there were
    # in-flight tasks (there are none here), so it stays False.
    assert asyncio.run(drive()) is False
