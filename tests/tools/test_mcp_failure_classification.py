"""Tests for exception-group unwrapping and failure classification in
``tools/mcp_tool.py`` (#65673, #66092).

The MCP SDK's anyio TaskGroups wrap real errors in ``BaseExceptionGroup``,
whose ``str()`` is "unhandled errors in a TaskGroup (N sub-exceptions)" —
useless in logs. ``_unwrap_exception_group`` digs out the root cause;
``_classify_mcp_failure`` decides whether a failure is worth retrying.
"""

import asyncio
import errno
import logging

import pytest

from tools.mcp_tool_errors import (
    InvalidMcpUrlError, NonMcpEndpointError, _classify_mcp_failure, _unwrap_exception_group)
from tools.mcp_tool import MCPServerTask


def _group(*excs, msg="unhandled errors in a TaskGroup") -> BaseExceptionGroup:
    return BaseExceptionGroup(msg, list(excs))


# ── _unwrap_exception_group ──────────────────────────────────────────────────

class TestUnwrapExceptionGroup:
    def test_plain_exception_passes_through(self):
        exc = ConnectionError("boom")
        assert _unwrap_exception_group(exc) is exc

    def test_single_level_group(self):
        inner = BrokenPipeError()
        assert _unwrap_exception_group(_group(inner)) is inner


    def test_system_exit_reraises(self):
        with pytest.raises(SystemExit):
            _unwrap_exception_group(_group(SystemExit(2)))


# ── _classify_mcp_failure ────────────────────────────────────────────────────

class TestClassifyMcpFailure:
    @pytest.mark.parametrize("exc", [
        ConnectionError("connection refused"),
        ConnectionResetError("reset by peer"),
        BrokenPipeError(),
        EOFError(),
        TimeoutError("read timeout"),
        OSError(errno.ECONNRESET, "reset"),
        RuntimeError("something odd"),
    ])
    def test_transient_failures(self, exc):
        assert _classify_mcp_failure(exc) == "transient"


    def test_closed_resource_transient(self):
        anyio = pytest.importorskip("anyio")
        assert _classify_mcp_failure(anyio.ClosedResourceError()) == "transient"


    def test_permanent_inside_taskgroup(self):
        # Classification must apply to the UNWRAPPED root cause.
        g = _group(_group(FileNotFoundError("cmd not found")))
        assert _classify_mcp_failure(g) == "permanent"


# ── Keepalive failure log surfaces the root cause ────────────────────────────

@pytest.mark.no_isolate
def test_keepalive_failure_logs_root_cause(monkeypatch, tmp_path, caplog):
    """A keepalive that dies with a TaskGroup-wrapped BrokenPipeError (empty
    str) must log 'BrokenPipeError', not 'unhandled errors in a TaskGroup'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    class _Task(MCPServerTask):
        async def _keepalive_probe(self):
            raise _group(BrokenPipeError())

    task = _Task("pipey")
    task._config = {"keepalive_interval": 0.01}
    task.session = object()

    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.01)

    async def _scenario():
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            reason = await task._wait_for_lifecycle_event()
        assert reason == "reconnect"

    asyncio.run(_scenario())

    keepalive_logs = [
        r.getMessage() for r in caplog.records if "keepalive failed" in r.getMessage()
    ]
    assert keepalive_logs, "keepalive failure was not logged"
    assert any("BrokenPipeError" in m for m in keepalive_logs), keepalive_logs
    assert not any("unhandled errors in a TaskGroup" in m for m in keepalive_logs)


# ── run() parks permanent failures immediately ───────────────────────────────

@pytest.mark.no_isolate
def test_permanent_failure_parks_without_retry_ladder(monkeypatch, tmp_path, caplog):
    """A stdio command that doesn't exist (FileNotFoundError) must park after
    ONE attempt — not burn _MAX_INITIAL_CONNECT_RETRIES identical warnings."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                raise FileNotFoundError("nonexistent-mcp-command")

        task = _Task("missing-cmd")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "nope"}))
            for _ in range(500):
                await _real_sleep(0)
                if state["parked"]:
                    break

        assert state["parked"], "permanent failure never parked"
        assert state["transport_calls"] == 1, (
            f"permanent failure burned {state['transport_calls']} attempts — "
            "should park immediately"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "permanent error" in r.getMessage()
    ]
    assert len(park_warnings) == 1
    assert "FileNotFoundError" in park_warnings[0].getMessage()


# ── An initial 401 must stay revivable ───────────────────────────────────────

@pytest.mark.no_isolate
def test_initial_auth_failure_parks_and_revives_after_relogin(
    monkeypatch, tmp_path, caplog,
):
    """A 401 on the FIRST connect must park, not end the run task.

    Ending the task drops the only listener on ``_reconnect_event``, so the
    server stayed dead for the life of the process even after the user
    re-authenticated. Parking keeps it revivable via the self-probe.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    httpx = pytest.importorskip("httpx")

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    def _auth_error():
        request = httpx.Request("POST", "https://mcp.example.test/mcp")
        response = httpx.Response(401, request=request)
        return httpx.HTTPStatusError("401", request=request, response=response)

    state = {"transport_calls": 0, "parked": False, "authenticated": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if not state["authenticated"]:
                    raise _group(_auth_error())
                self.session = object()
                await self._wait_for_lifecycle_event()

        task = _Task("figma")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(500):
                await _real_sleep(0)
                if state["parked"]:
                    break

            assert state["parked"], "auth failure never parked"
            assert state["transport_calls"] == 1, (
                f"auth failure burned {state['transport_calls']} attempts"
            )
            assert not run_task.done(), (
                "run task exited on a 401 — the server is now unrevivable"
            )

            # The user re-authenticates. Nothing sets _reconnect_event:
            # revival must come from the timed self-probe alone.
            state["authenticated"] = True
            for _ in range(200):
                await _real_sleep(0.01)
                if task.session is not None:
                    break

        assert task.session is not None, (
            "parked server never recovered after re-authentication "
            f"(transport_calls={state['transport_calls']})"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    auth_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "failed initial authentication" in r.getMessage()
    ]
    assert len(auth_warnings) == 1
    assert "hermes mcp login figma" in auth_warnings[0].getMessage()
