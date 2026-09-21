"""A same-generation MCP recovery must be visible, and a reconnect must not hang.

Observed against a hosted OAuth streamable-HTTP server behind a slow edge
(Cloudflare 524 after 100-125 s): two keepalive failures, the server
reconnected by itself both times — and logged NOTHING about it: tools were
already registered (no "registered" line) and "revived" is gated on a park
that never happened. Anything that monitors the log saw two failures and then
silence for hours while tool calls were succeeding.

- recovery is logged once per degraded episode, from ``_mark_session_proven``
  — the runtime's own definition of proven health — never from a bare
  handshake;
- a re-established session logs one informational line that does NOT claim
  health;
- ``tools/list`` on the RECONNECT path has a deadline (the initial connect is
  bounded by its caller; a reconnect has no caller).
"""

import asyncio
import logging

import pytest

from tools import mcp_tool
from tools.mcp_tool import MCPServerTask

LOGGER = "tools.mcp_tool"


def _messages(caplog, level=None):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and (level is None or r.levelno == level)]


def test_recovery_is_logged_from_proof_not_from_handshake(caplog):
    task = MCPServerTask("jira")
    task._ever_connected = True
    with caplog.at_level(logging.INFO, logger=LOGGER):
        task._note_degraded()
        task._note_session_established()
        established = _messages(caplog)
        assert any("session re-established" in m for m in established)
        # The informational line must not read as proof to a log consumer.
        assert not any("session healthy again" in m for m in established)
        assert task._degraded_since is not None, "a handshake must not close the episode"

        task._session_proven = False
        task._mark_session_proven()
    proof = [m for m in _messages(caplog) if "recovered — session healthy again" in m]
    assert len(proof) == 1
    assert "'jira'" in proof[0] and "degraded → connected" in proof[0]
    assert task._degraded_since is None


def test_no_recovery_line_without_a_degraded_episode(caplog):
    task = MCPServerTask("asana")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        task._note_session_established()      # first connect: nothing to report
        task._session_proven = False
        task._mark_session_proven()           # ordinary keepalive success
    assert not [m for m in _messages(caplog)
                if "recovered" in m or "re-established" in m]


def test_flapping_logs_one_line_per_episode_not_per_rebuild(caplog):
    task = MCPServerTask("jira")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        task._note_degraded()
        first = task._degraded_since
        for _ in range(5):                    # five rebuilds, never proven
            task._note_session_established()
            task._note_degraded()
        assert task._degraded_since == first, "the episode start must not move"
    assert len([m for m in _messages(caplog) if "re-established" in m]) == 1


def test_a_park_keeps_the_existing_revived_line(caplog):
    task = MCPServerTask("jira")
    task._note_degraded()
    task._was_parked = True
    task._session_proven = False
    with caplog.at_level(logging.INFO, logger=LOGGER):
        task._mark_session_proven()
    msgs = _messages(caplog)
    assert any("revived — session healthy again after parking" in m for m in msgs)
    assert not any("recovered —" in m for m in msgs), "one transition, one line"
    assert task._degraded_since is None


def test_an_intentional_stdio_recycle_closes_the_episode(caplog):
    """Dormancy is not degradation: the lazy restart hours later must not claim a recovery that
    spans the whole idle period."""
    task = MCPServerTask("local-stdio")
    task._note_degraded()
    task._note_session_established()
    task._mark_stdio_recycled("idle_timeout_seconds")
    assert task._degraded_since is None
    with caplog.at_level(logging.INFO, logger=LOGGER):
        task._session_proven = False
        task._mark_session_proven()
    assert not [m for m in _messages(caplog) if "recovered" in m]


def test_tools_list_has_a_deadline_on_reconnect_and_releases_the_rpc_lock(caplog):
    async def scenario():
        class _Hangs(MCPServerTask):          # __slots__: override, never assign
            async def _discover_tools(self):
                async with self._rpc_lock:
                    await asyncio.Event().wait()

        task = _Hangs("jira")
        task._ever_connected = True           # this is a RECONNECT
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(asyncio.TimeoutError):
                await task._discover_tools_bounded(0.05)
        assert not task._rpc_lock.locked(), "a timed-out discovery must not strand _rpc_lock"

    asyncio.run(scenario())
    assert any("tools/list did not answer" in m for m in _messages(caplog, logging.WARNING))


def test_a_server_that_parked_before_it_ever_connected_is_bounded_too():
    """``_ever_connected`` is still False when a parked never-connected server
    self-probes, and its original caller is long gone — the deadline must not
    depend on that flag."""
    async def scenario():
        class _Hangs(MCPServerTask):
            async def _discover_tools(self):
                async with self._rpc_lock:
                    await asyncio.Event().wait()

        task = _Hangs("jira")
        assert task._ever_connected is False
        with pytest.raises(asyncio.TimeoutError):
            await task._discover_tools_bounded(0.05)
        assert not task._rpc_lock.locked()

    asyncio.run(scenario())


def test_every_transport_serves_through_the_bounded_discovery(caplog):
    """Every transport (stdio, SSE, streamable HTTP) establishes through ``_serve_session`` — a
    hung ``tools/list`` there must end at ``connect_timeout`` with nobody else waiting on it."""
    async def scenario():
        class _Hangs(MCPServerTask):
            async def _negotiate_session(self, session, connect_timeout):
                return object()

            async def _discover_tools(self):
                async with self._rpc_lock:
                    await asyncio.Event().wait()

        task = _Hangs("jira")
        task._ever_connected = True
        with pytest.raises(asyncio.TimeoutError):
            await task._serve_session(object(), 0.05)
        assert not task._ready.is_set(), "an undiscovered session must not be published as ready"
        assert not task._rpc_lock.locked()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(scenario())
    assert any("tools/list did not answer" in m for m in _messages(caplog, logging.WARNING))


def test_a_dead_stdio_child_opens_the_episode(monkeypatch):
    """Upstream's stdio dead-child path (#115483) breaks into a reconnect without a keepalive
    failure; without the episode that recovery would be silent again."""
    monkeypatch.setattr(mcp_tool, "_DEFAULT_KEEPALIVE_INTERVAL", 0.01)

    async def scenario():
        class _Task(MCPServerTask):
            def _stdio_children_dead(self):
                return True

        task = _Task("local-stdio")
        task._config = {"command": "python"}
        task.session = object()
        task._ever_connected = True
        assert await task._wait_for_lifecycle_event() == "reconnect"
        assert task._degraded_since is not None

    asyncio.run(scenario())


def test_a_failed_keepalive_opens_the_episode_and_the_next_success_closes_it(monkeypatch, caplog):
    """The wiring, end to end through the real lifecycle wait: keepalive
    fails → "reconnect" with the episode open → (new session) keepalive
    succeeds → exactly one recovery line."""
    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.01)

    async def scenario():
        outcome = {"fail": True}

        class _Task(MCPServerTask):
            async def _keepalive_probe(self):
                if outcome["fail"]:
                    raise asyncio.TimeoutError()

        task = _Task("jira")
        task._config = {"keepalive_interval": 0.01}
        task.session = object()
        task._ever_connected = True
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert await task._wait_for_lifecycle_event() == "reconnect"
            assert task._degraded_since is not None

            # run() would now rebuild the transport; the new session:
            outcome["fail"] = False
            task._session_proven = False
            task._note_session_established()
            waiter = asyncio.ensure_future(task._wait_for_lifecycle_event())
            for _ in range(200):
                await asyncio.sleep(0.01)
                if task._degraded_since is None:
                    break
            task._shutdown_event.set()
            await waiter
        assert task._degraded_since is None

    asyncio.run(scenario())
    msgs = _messages(caplog)
    order = [("failed" if "keepalive failed, triggering" in m else
              "established" if "re-established" in m else
              "recovered" if "recovered — session healthy again" in m else None) for m in msgs]
    assert [o for o in order if o] == ["failed", "established", "recovered"], msgs
