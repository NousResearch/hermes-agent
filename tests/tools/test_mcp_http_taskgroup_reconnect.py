"""Regression tests for SDK TaskGroup failures on HTTP/SSE MCP transports (#66092).

Background
==========
The Python MCP SDK runs the SSE stream reader inside an internal anyio task
group. When that stream drops (idle timeout, LB reset, brief network blip),
the failure escapes the transport's ``async with`` as a ``BaseExceptionGroup``
— often carrying ``CancelledError`` leaves. Before the fix:

- Groups with only ``Exception`` leaves hit ``run()``'s broad
  ``except Exception``: full session teardown, exponential backoff
  (1s → 2s → 4s → ...), and eventually a 300s park with tools deregistered —
  even though the HTTP POST path still worked fine.
- Groups carrying ``CancelledError`` leaves escaped ``run()`` entirely
  (``except Exception`` can't catch them), silently killing the reconnect
  loop.

The fix wraps the transport blocks in ``_run_http`` with
``except BaseExceptionGroup`` and routes post-handshake drops through
``_handle_transport_exception_group``, which returns ``"reconnect"`` — the
same immediate-rebuild lifecycle path OAuth recovery uses — instead of
propagating. Pre-handshake failures, ``KeyboardInterrupt``/``SystemExit``,
external task cancellation, and rapid connect/drop loops still re-raise so
the normal retry/backoff classification applies.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


def _make_task(name="tg-test"):
    from tools.mcp_tool import MCPServerTask

    return MCPServerTask(name)


def _exc_group(*leaves):
    return BaseExceptionGroup("transport died", list(leaves))


# ---------------------------------------------------------------------------
# _handle_transport_exception_group unit tests
# ---------------------------------------------------------------------------


class TestHandleTransportExceptionGroup:
    def test_established_session_returns_reconnect(self):
        task = _make_task()
        eg = _exc_group(RuntimeError("SSE stream closed"))
        reason = task._handle_transport_exception_group(eg, time.monotonic() - 120)
        assert reason == "reconnect"

    def test_cancelled_error_leaves_return_reconnect(self):
        """The reported symptom: the SDK's internal reader task gets cancelled
        and the group carries CancelledError — a BaseExceptionGroup that
        ``except Exception`` can NOT catch. It must still map to reconnect."""
        task = _make_task()
        eg = _exc_group(asyncio.CancelledError())
        assert not isinstance(eg, Exception), (
            "test premise: a CancelledError-bearing group is not an Exception"
        )
        reason = task._handle_transport_exception_group(eg, time.monotonic() - 120)
        assert reason == "reconnect"

    def test_pre_handshake_failure_reraises(self):
        """No session was established — this is an ordinary connect failure
        and must go through run()'s normal retry/backoff classification."""
        task = _make_task()
        eg = _exc_group(ConnectionError("refused"))
        with pytest.raises(BaseExceptionGroup):
            task._handle_transport_exception_group(eg, None)

    def test_keyboard_interrupt_reraises(self):
        task = _make_task()
        eg = _exc_group(KeyboardInterrupt())
        with pytest.raises(BaseExceptionGroup):
            task._handle_transport_exception_group(eg, time.monotonic() - 120)

    def test_system_exit_reraises(self):
        task = _make_task()
        eg = _exc_group(SystemExit(1))
        with pytest.raises(BaseExceptionGroup):
            task._handle_transport_exception_group(eg, time.monotonic() - 120)

    def test_external_cancellation_reraises(self):
        """A group arriving while the surrounding task is being cancelled
        (shutdown / gateway restart) must propagate — swallowing it would
        wedge the run loop (#9930)."""
        task = _make_task()

        async def scenario():
            current = asyncio.current_task()
            current.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass  # absorb delivery; cancelling() stays > 0
            try:
                eg = _exc_group(RuntimeError("noise during cancel"))
                with pytest.raises(BaseExceptionGroup):
                    task._handle_transport_exception_group(
                        eg, time.monotonic() - 120
                    )
            finally:
                current.uncancel()

        asyncio.run(scenario())

    def test_rapid_drops_fall_back_to_backoff(self):
        """Sessions that die within _TRANSPORT_STABLE_SESSION_SECONDS over and
        over are not transient blips: after the budget is spent the group
        re-raises so backoff/park applies instead of hot-looping handshakes."""
        from tools import mcp_tool

        task = _make_task()
        budget = mcp_tool._MAX_CONSECUTIVE_RAPID_DROPS

        for _ in range(budget):
            reason = task._handle_transport_exception_group(
                _exc_group(RuntimeError("instant drop")), time.monotonic()
            )
            assert reason == "reconnect"

        with pytest.raises(BaseExceptionGroup):
            task._handle_transport_exception_group(
                _exc_group(RuntimeError("instant drop")), time.monotonic()
            )

        # The counter resets on exhaustion so the server isn't permanently
        # locked out of immediate reconnects after backoff heals it.
        reason = task._handle_transport_exception_group(
            _exc_group(RuntimeError("instant drop")), time.monotonic()
        )
        assert reason == "reconnect"

    def test_stable_session_resets_rapid_drop_counter(self):
        from tools import mcp_tool

        task = _make_task()
        budget = mcp_tool._MAX_CONSECUTIVE_RAPID_DROPS

        for _ in range(budget):
            task._handle_transport_exception_group(
                _exc_group(RuntimeError("instant drop")), time.monotonic()
            )
        assert task._rapid_transport_drops == budget

        # One long-lived session wipes the streak.
        stable_age = mcp_tool._TRANSPORT_STABLE_SESSION_SECONDS + 1
        reason = task._handle_transport_exception_group(
            _exc_group(RuntimeError("late drop")),
            time.monotonic() - stable_age,
        )
        assert reason == "reconnect"
        assert task._rapid_transport_drops == 0


# ---------------------------------------------------------------------------
# _run_http integration (SSE branch with a faked transport)
# ---------------------------------------------------------------------------


class _FakeClientSession:
    """Minimal stand-in for mcp.ClientSession."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def initialize(self):
        return SimpleNamespace(capabilities=None)


@asynccontextmanager
async def _fake_sse_client(**kwargs):
    yield (object(), object())


def _patch_sse_transport(monkeypatch):
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "sse_client", _fake_sse_client)
    monkeypatch.setattr(mcp_tool, "ClientSession", _FakeClientSession)


class TestRunHttpTaskGroupHandling:
    def test_post_handshake_group_returns_reconnect(self, monkeypatch, tmp_path):
        """A BaseExceptionGroup escaping the transport AFTER the handshake
        (session established, tools discovered) must surface as a clean
        "reconnect" return from _run_http, not propagate to run()."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool import MCPServerTask

        _patch_sse_transport(monkeypatch)

        class _Task(MCPServerTask):
            async def _discover_tools(self):
                self._tools = []

            async def _wait_for_lifecycle_event(self):
                # Simulate the SDK's internal reader dying mid-session: anyio
                # cancels the host body and the task group re-raises as a
                # cancellation-bearing BaseExceptionGroup.
                raise _exc_group(asyncio.CancelledError())

        task = _Task("sse-drop")
        reason = asyncio.run(
            task._run_http({"url": "https://example.com/mcp", "transport": "sse"})
        )
        assert reason == "reconnect"
        assert task._ready.is_set(), "readiness from the healthy session persists"

    def test_handshake_failure_group_propagates(self, monkeypatch, tmp_path):
        """A group raised BEFORE the session is established re-raises so
        run() applies its normal initial-connect retry/backoff."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools import mcp_tool
        from tools.mcp_tool import MCPServerTask

        _patch_sse_transport(monkeypatch)

        class _FailingSession(_FakeClientSession):
            async def initialize(self):
                raise _exc_group(ConnectionError("handshake refused"))

        monkeypatch.setattr(mcp_tool, "ClientSession", _FailingSession)

        task = MCPServerTask("sse-handshake-fail")
        with pytest.raises(BaseExceptionGroup):
            asyncio.run(
                task._run_http(
                    {"url": "https://example.com/mcp", "transport": "sse"}
                )
            )

    def test_run_reenters_transport_without_backoff_after_group(
        self, monkeypatch, tmp_path
    ):
        """End-to-end through run(): a post-handshake group must trigger an
        immediate transport rebuild (the clean "reconnect" path) — no backoff
        sleep, no retry-counter increment, no tool deregistration."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool import MCPServerTask

        state = {"connects": 0, "sleeps": 0, "deregistered": False}

        _real_sleep = asyncio.sleep

        async def _counting_sleep(delay, *a, **kw):
            if delay > 0:
                state["sleeps"] += 1
            await _real_sleep(0)

        from tools import mcp_tool

        monkeypatch.setattr(mcp_tool.asyncio, "sleep", _counting_sleep)

        class _Task(MCPServerTask):
            def _is_http(self):
                return True

            def _deregister_tools(self):
                state["deregistered"] = True
                self._registered_tool_names = []

            async def _run_http(self, config):
                state["connects"] += 1
                if state["connects"] == 1:
                    # First connect: healthy handshake, then the stream dies.
                    self.session = object()
                    self._ready.set()
                    self._reconnect_retries = 0
                    established_at = time.monotonic() - 60
                    try:
                        raise _exc_group(asyncio.CancelledError())
                    except BaseExceptionGroup as eg:
                        return self._handle_transport_exception_group(
                            eg, established_at
                        )
                # Second connect: stay up until shutdown.
                self.session = object()
                self._ready.set()
                return await self._wait_for_lifecycle_event()

        async def scenario():
            task = _Task("run-loop")
            task._registered_tool_names = ["run-loop__tool"]
            run_task = asyncio.ensure_future(
                # skip_preflight: the content-type probe would issue a real
                # HTTP request; this test never touches the network.
                task.run({"url": "https://example.com/mcp", "skip_preflight": True})
            )
            for _ in range(200):
                await _real_sleep(0)
                if state["connects"] >= 2:
                    break
            task._shutdown_event.set()
            task._reconnect_event.set()
            try:
                await asyncio.wait_for(run_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                run_task.cancel()
            return task

        task = asyncio.run(scenario())
        assert state["connects"] >= 2, "run() should rebuild the transport"
        assert state["sleeps"] == 0, (
            "immediate reconnect must not pass through a backoff sleep"
        )
        assert not state["deregistered"], (
            "tools must stay registered across an immediate reconnect"
        )
        assert task._reconnect_retries == 0
