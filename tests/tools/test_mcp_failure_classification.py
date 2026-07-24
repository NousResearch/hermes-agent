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

from tools.mcp_tool import (
    InvalidMcpUrlError,
    MCPServerTask,
    NonMcpEndpointError,
    _classify_mcp_failure,
    _unwrap_exception_group,
)


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

    def test_nested_groups(self):
        inner = ConnectionResetError("reset by peer")
        nested = _group(_group(_group(inner)))
        assert _unwrap_exception_group(nested) is inner

    def test_root_cause_name_visible_for_empty_message(self):
        # Dead stdio pipes raise BrokenPipeError with an EMPTY str() —
        # the log format must rely on type(exc).__name__, and unwrap must
        # hand back the BrokenPipeError, not the opaque group.
        root = _unwrap_exception_group(_group(BrokenPipeError()))
        assert type(root).__name__ == "BrokenPipeError"

    def test_prefers_non_cancellation_leaf(self):
        # anyio cancellation sprays CancelledError across sibling tasks;
        # the real error must win.
        real = ConnectionError("server hung up")
        g = _group(asyncio.CancelledError(), real, asyncio.CancelledError())
        assert _unwrap_exception_group(g) is real

    def test_prefers_non_cancellation_leaf_nested(self):
        real = TimeoutError("read timed out")
        g = _group(_group(asyncio.CancelledError()), _group(real))
        assert _unwrap_exception_group(g) is real

    def test_all_cancellation_returns_cancellation(self):
        g = _group(asyncio.CancelledError())
        assert isinstance(_unwrap_exception_group(g), asyncio.CancelledError)

    def test_keyboard_interrupt_reraises(self):
        with pytest.raises(KeyboardInterrupt):
            _unwrap_exception_group(_group(KeyboardInterrupt()))

    def test_nested_keyboard_interrupt_reraises(self):
        with pytest.raises(KeyboardInterrupt):
            _unwrap_exception_group(
                _group(ConnectionError("x"), _group(KeyboardInterrupt()))
            )

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

    def test_transient_taskgroup_drop(self):
        g = _group(ConnectionError("sse stream dropped"))
        assert _classify_mcp_failure(g) == "transient"

    def test_closed_resource_transient(self):
        anyio = pytest.importorskip("anyio")
        assert _classify_mcp_failure(anyio.ClosedResourceError()) == "transient"

    @pytest.mark.parametrize("exc_factory", [
        lambda: FileNotFoundError("no such file: nonexistent-mcp-cmd"),
        lambda: OSError(errno.ENOENT, "No such file or directory"),
        lambda: NonMcpEndpointError("url serves text/html"),
        lambda: InvalidMcpUrlError("bad scheme"),
    ])
    def test_permanent_failures(self, exc_factory):
        assert _classify_mcp_failure(exc_factory()) == "permanent"

    @pytest.mark.parametrize("status", [401, 403])
    def test_http_auth_status_permanent(self, status):
        httpx = pytest.importorskip("httpx")
        req = httpx.Request("POST", "http://x/mcp")
        resp = httpx.Response(status, request=req)
        exc = httpx.HTTPStatusError("auth", request=req, response=resp)
        assert _classify_mcp_failure(exc) == "permanent"

    def test_http_5xx_transient(self):
        httpx = pytest.importorskip("httpx")
        req = httpx.Request("POST", "http://x/mcp")
        resp = httpx.Response(503, request=req)
        exc = httpx.HTTPStatusError("unavailable", request=req, response=resp)
        assert _classify_mcp_failure(exc) == "transient"

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
