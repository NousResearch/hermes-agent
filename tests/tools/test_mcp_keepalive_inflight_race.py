"""Regression tests for the generation-scoped MCP admission gate.

Residual of now-closed PR #48069 (fix/mcp-keepalive-inflight-race) that the
merged #94184 did NOT carry: the ZERO-ACTIVE-RPC late-admission window.

Background
----------
``MCPServerTask._fail_inflight_calls`` cancels in-flight RPCs on a
reconnect/shutdown teardown so they fail fast instead of riding a dying
transport to the full tool timeout. But it early-returns when there are no
victims to cancel::

    victims = [t for t in self._inflight_tasks if not t.done()]
    if not victims:
        return          # <-- zero-active-RPC: never flags the connection

In that zero-active-RPC case the connection was never flagged
``_reconnecting`` nor ``mark_suspect``. Meanwhile ``_track_inflight_rpc``
admitted a fresh task into ``_inflight_tasks`` with no admission gate. So a
LATE call arriving during the teardown window could be admitted against a
retiring ``ClientSession`` and then hang/fail on the dying transport instead
of cleanly rendezvousing with the rebuilt session. This is the sharp case
@andrexibiza identified in the #48069 review.

The fix (this PR) adds a generation-scoped admission gate:

* ``_rpc_generation`` (monotonic id of the live session) + ``_admitting_generation``
  (the generation currently accepting rpcs; ``None`` == draining/closed).
* ``_publish_session()`` opens admission BEFORE storing ``self.session``.
* ``_close_rpc_admission()`` closes admission at every lifecycle exit BEFORE
  the cancellation sweep, INDEPENDENTLY of whether there were victims, so the
  zero-active-RPC window is sealed.
* ``_track_inflight_rpc`` reads the gate SYNCHRONOUSLY (no await between the
  admission check and the ``_inflight_tasks.add``) and refuses a late call
  with the same retryable ``RuntimeError`` the teardown-cancel path raises.

Fail-first discipline: ``test_late_call_in_teardown_window_is_refused_not_admitted``
FAILS on unmodified main (the late call is admitted and the retiring session's
``call_tool`` runs) and PASSES with the gate in place.
"""

from __future__ import annotations

import asyncio

import pytest

from tools.mcp_tool import MCPServerTask, _track_inflight_rpc


class _RetiringSession:
    """A ClientSession that is being torn down. Its ``call_tool`` must NEVER run.

    Models the dying transport: any RPC that reaches it in the teardown window
    is exactly the bug, a call admitted against a retiring session.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def call_tool(self, *args, **kwargs):  # pragma: no cover - must not run
        self.call_count += 1
        raise AssertionError(
            "retiring session's call_tool was invoked, a late call was admitted "
            "against a session being torn down (the zero-active-RPC race)"
        )


async def _run_user_rpc_through_real_path(server: MCPServerTask) -> str:
    """Submit a user RPC through the REAL _rpc_lock + _track_inflight_rpc path.

    Returns "refused" if the admission gate refused the call (the fix), or
    "admitted" if the retiring session's call_tool was reached (the bug).
    """
    async with server._rpc_lock:
        try:
            async with _track_inflight_rpc(server, server.name, "tools/call probe"):
                # Only reached if the gate admitted the call. On buggy main this
                # runs against the retiring session and trips its AssertionError.
                await server.session.call_tool("probe", {})
        except RuntimeError as exc:
            # The controlled, retryable reconnect-teardown error.
            assert "retry the request on the rebuilt session" in str(exc), exc
            return "refused"
    return "admitted"


@pytest.mark.asyncio
async def test_late_call_in_teardown_window_is_refused_not_admitted():
    """Fail-first: a late RPC in the zero-active-RPC teardown window is refused.

    On unmodified main the connection is never flagged (zero victims), the late
    call is admitted, and ``_RetiringSession.call_tool`` runs (AssertionError).
    With the gate, admission is closed and the call is refused (retryable).
    """
    server = MCPServerTask("retiring")
    retiring = _RetiringSession()

    # Publish a live session the normal way so admission opens for its
    # generation (mirrors a real connect/reconnect).
    server._publish_session(retiring)
    assert server.session is retiring
    assert server._admitting_generation == server._rpc_generation

    # Fire a reconnect teardown with ZERO active RPCs. This drives the real
    # lifecycle exit, which closes admission BEFORE the (empty) cancel sweep.
    server._reconnect_event.set()
    reason = await server._wait_for_lifecycle_event()
    assert reason == "reconnect"

    # Model the teardown WINDOW: the retiring session is still published (the
    # transport has not finished unwinding yet), but admission is closed.
    server.session = retiring
    assert server._admitting_generation is None, (
        "admission must be closed even in the zero-victims case that "
        "_fail_inflight_calls early-returns from"
    )

    # Submit a user RPC through the REAL path.
    outcome = await _run_user_rpc_through_real_path(server)

    # (a) the retiring handler was never invoked
    assert retiring.call_count == 0, "late call reached the retiring session"
    # (b) the caller got the controlled retryable reconnect error
    assert outcome == "refused"
    # (c) it was never registered in _inflight_tasks
    assert all(t.done() for t in server._inflight_tasks) or not server._inflight_tasks
    assert asyncio.current_task() not in server._inflight_tasks


@pytest.mark.asyncio
async def test_publish_session_bumps_generation_monotonically():
    """Each published session gets a fresh, strictly increasing generation."""
    server = MCPServerTask("gen")
    assert server._rpc_generation == 0
    assert server._admitting_generation is None

    seen = []
    for _ in range(4):
        server._publish_session(object())
        seen.append(server._rpc_generation)
        # Admission always tracks the just-published generation.
        assert server._admitting_generation == server._rpc_generation

    # Strictly monotonic.
    assert seen == sorted(seen)
    assert len(set(seen)) == len(seen)
    assert seen[0] == 1 and seen[-1] == 4


@pytest.mark.asyncio
async def test_close_then_next_generation_reopens_admission():
    """After a teardown closes admission, the NEXT published session reopens it.

    A late call is refused while draining, then a rebuilt session admits calls
    again under a new generation, and a call on the rebuilt session succeeds.
    """
    server = MCPServerTask("reopen")

    # Generation 1, then close admission (teardown).
    first = _RetiringSession()
    server._publish_session(first)
    gen1 = server._rpc_generation
    server._close_rpc_admission()
    assert server._admitting_generation is None

    # While draining, a late call against the retiring session is refused.
    server.session = first
    outcome = await _run_user_rpc_through_real_path(server)
    assert outcome == "refused"
    assert first.call_count == 0

    # Rebuild: publish a fresh, healthy session (new generation reopens admission).
    class _HealthySession:
        def __init__(self) -> None:
            self.call_count = 0

        async def call_tool(self, *args, **kwargs):
            self.call_count += 1
            return "ok"

    healthy = _HealthySession()
    server._publish_session(healthy)
    gen2 = server._rpc_generation
    assert gen2 > gen1
    assert server._admitting_generation == gen2

    # A call on the rebuilt session is admitted and reaches the healthy handler.
    async with server._rpc_lock:
        async with _track_inflight_rpc(server, server.name, "tools/call probe"):
            result = await server.session.call_tool("probe", {})
    assert result == "ok"
    assert healthy.call_count == 1
