"""Regression tests for stdio MCP servers that crash at startup (#98763).

A stdio server whose child process exits before the MCP handshake completes
(e.g. ``ModuleNotFoundError`` under the shebang-resolved interpreter, any
import-time error) used to be classified ``transient``: the crash happens in
a separate process, so the parent only observes a closed stdio pipe, and
``run()`` burned the full initial retry ladder, parked, then repeated on
every parked self-probe — flooding logs for days (#98763: 1,982 park events
across two servers in ~12 days).

The fix detects the dead child when the ``stdio_client`` context unwinds and
raises :class:`StdioStartupCrashError`, which ``_classify_mcp_failure``
treats as permanent: park immediately with one clear log line (root cause +
the child's stderr tail), while staying revivable via the self-probe.

These tests drive the *real* ``MCPServerTask._run_stdio`` / ``run()`` with
fake transports — no real subprocess, no network.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")
pytest.importorskip("psutil")

DEAD_PID = 4194304  # > typical pid_max; only ever consulted via mocked psutil


class _FakeAsyncCM:
    """Minimal async context manager yielding a fixed value; spawns nothing."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_exc):
        return False


class _CrashingAtHandshakeSession:
    """Stand-in ClientSession: the child dies before ``initialize()`` answers."""

    async def initialize(self):
        raise EOFError("stream closed before initialize response")


class _HealthyThenDroppingSession:
    """Handshake completes, then the transport drops while serving."""

    async def initialize(self):
        return object()


def _drive_run_stdio(task, config, **patches):
    async def drive():
        return await task._run_stdio(config)

    # Caller-supplied patches plus the hermetic baseline every _run_stdio
    # drive needs (spawn-adjacent helpers would touch the real process
    # table / stderr log).
    defaults = {
        "stdio_client": lambda *a, **k: _FakeAsyncCM((object(), object())),
        "ClientSession": lambda *a, **k: _FakeAsyncCM(_CrashingAtHandshakeSession()),
        "_resolve_stdio_command": lambda c, e: (c, e),
        "_write_stderr_log_header": lambda *_a, **_k: None,
        "_get_mcp_stderr_log": lambda: None,
        "_kill_orphaned_mcp_children": lambda: None,
    }
    defaults.update(patches)
    with patch("tools.osv_check.check_package_for_malware", lambda *_a, **_k: None):
        with patch.multiple("tools.mcp_tool", **defaults):
            return asyncio.run(drive())


def _spawn_snapshot_mock():
    """``_snapshot_child_pids`` that sees no children before the spawn and
    exactly one (DEAD_PID) after it — as if the SDK spawned a child that has
    since exited."""
    calls = {"n": 0}

    def _snapshot():
        calls["n"] += 1
        return set() if calls["n"] == 1 else {DEAD_PID}

    return _snapshot


class TestRunStdioStartupCrashDetection:
    def test_dead_child_crash_is_wrapped_as_permanent(self, monkeypatch):
        """A child that dies before the handshake must surface as
        ``StdioStartupCrashError`` carrying the unwrapped root cause."""
        from tools import mcp_tool

        task = mcp_tool.MCPServerTask("xapi-bearer")
        monkeypatch.setattr("psutil.pid_exists", lambda pid: False)

        with pytest.raises(mcp_tool.StdioStartupCrashError) as excinfo:
            _drive_run_stdio(
                task,
                {"command": "python", "args": ["-m", "xapi_bearer"]},
                _snapshot_child_pids=_spawn_snapshot_mock(),
                _filter_mcp_children=lambda pids: set(pids),
            )

        assert isinstance(excinfo.value.__cause__, EOFError)
        assert "before the MCP handshake" in str(excinfo.value)
        assert "EOFError" in str(excinfo.value)

    def test_live_child_drop_is_not_wrapped(self, monkeypatch):
        """The same EOF with the child still alive (e.g. a server that
        answers garbage then hangs) must keep today's transient behaviour —
        only a dead child proves a deterministic startup crash."""
        from tools import mcp_tool

        task = mcp_tool.MCPServerTask("hangy")
        monkeypatch.setattr("psutil.pid_exists", lambda pid: True)

        with pytest.raises(EOFError):
            _drive_run_stdio(
                task,
                {"command": "python", "args": []},
                _snapshot_child_pids=_spawn_snapshot_mock(),
                _filter_mcp_children=lambda pids: set(pids),
            )

    def test_spawn_failure_is_not_wrapped(self, monkeypatch):
        """When no child PID was captured this round (e.g. ``stdio_client``
        entry itself failed), stale PIDs from a previous attempt must not
        reclassify the failure — the raw error propagates as before."""
        from tools import mcp_tool

        task = mcp_tool.MCPServerTask("gone-cmd")
        monkeypatch.setattr("psutil.pid_exists", lambda pid: False)

        with pytest.raises(EOFError):
            _drive_run_stdio(
                task,
                {"command": "python", "args": []},
                _snapshot_child_pids=lambda: set(),
                _filter_mcp_children=lambda pids: set(pids),
            )

    def test_runtime_drop_after_handshake_is_not_wrapped(self, monkeypatch):
        """A server that handshakes fine and drops later may still recover on
        reconnect — ``_ever_connected`` must keep it out of the startup-crash
        bucket."""
        from tools import mcp_tool

        task = mcp_tool.MCPServerTask("flapper")
        monkeypatch.setattr("psutil.pid_exists", lambda pid: False)

        async def _drop(_self):
            raise EOFError("stream closed mid-session")

        with (
            patch.object(mcp_tool.MCPServerTask, "_wait_for_lifecycle_event", _drop),
            patch.object(mcp_tool.MCPServerTask, "_discover_tools", _noop_async),
        ):
            with pytest.raises(EOFError):
                _drive_run_stdio(
                    task,
                    {"command": "python", "args": []},
                    ClientSession=lambda *a, **k: _FakeAsyncCM(
                        _HealthyThenDroppingSession()
                    ),
                    _snapshot_child_pids=_spawn_snapshot_mock(),
                    _filter_mcp_children=lambda pids: set(pids),
                )


async def _noop_async(*_a, **_k):
    return None


class TestClassifyStdioStartupCrash:
    def test_direct_and_taskgroup_wrapped_are_permanent(self):
        from tools.mcp_tool import (
            StdioStartupCrashError,
            _classify_mcp_failure,
        )

        direct = StdioStartupCrashError("stdio process exited early")
        grouped = BaseExceptionGroup(
            "unhandled errors in a TaskGroup (1 sub-exception)", [direct]
        )
        assert _classify_mcp_failure(direct) == "permanent"
        assert _classify_mcp_failure(grouped) == "permanent"


class TestStderrTail:
    def test_tail_returns_last_lines_after_offset(self, monkeypatch, tmp_path):
        from tools import mcp_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        header = "===== starting MCP server 'xapi-bearer' =====\n"
        crash = (
            "Traceback (most recent call last):\n"
            '  File "xapi-bearer", line 1, in <module>\n'
            "    import yaml\n"
            "ModuleNotFoundError: No module named 'yaml'\n"
            "\n"
        )
        (log_dir / "mcp-stderr.log").write_text(header + crash, encoding="utf-8")

        tail = mcp_tool._read_mcp_stderr_tail(len(header.encode()))
        assert "ModuleNotFoundError: No module named 'yaml'" in tail
        assert "starting MCP server" not in tail

    def test_tail_no_offset_or_missing_file_is_empty(self, monkeypatch, tmp_path):
        from tools import mcp_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # No logs directory at all — must degrade to "".
        assert mcp_tool._read_mcp_stderr_tail(0) == ""
        assert mcp_tool._read_mcp_stderr_tail(None) == ""

    def test_tail_read_failure_emits_debug_signal(self, monkeypatch, tmp_path, caplog):
        from tools import mcp_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # The empty-string return is pinned above; the read failure must also
        # leave a debug breadcrumb so a missing stderr tail is explainable.
        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            assert mcp_tool._read_mcp_stderr_tail(0) == ""
        assert any(
            "mcp stderr tail unavailable" in record.message
            for record in caplog.records
        )


# ── run() parks startup crashes without the retry ladder ────────────────────


class TestStartupCrashParksImmediately:
    def test_startup_crash_parks_without_retry_ladder(
        self,
        monkeypatch,
        tmp_path,
        caplog,
    ):
        """A StdioStartupCrashError must park after ONE attempt with the
        stderr diagnostic in the log — not burn _MAX_INITIAL_CONNECT_RETRIES
        identical warnings (#98763)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools import mcp_tool

        _real_sleep = asyncio.sleep

        async def _fast_sleep(_delay, *a, **kw):
            await _real_sleep(0)

        monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

        state = {"transport_calls": 0, "parked": False}

        async def _scenario():
            class _Task(mcp_tool.MCPServerTask):
                def _is_http(self):
                    return False

                def _deregister_tools(self):
                    state["parked"] = True
                    self._registered_tool_names = []

                async def _run_stdio(self, config):
                    state["transport_calls"] += 1
                    raise mcp_tool.StdioStartupCrashError(
                        "stdio process exited before the MCP handshake "
                        "completed (caused by EOFError); recent stderr:\n"
                        "ModuleNotFoundError: No module named 'yaml'"
                    )

            task = _Task("xapi-bearer")

            with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
                run_task = asyncio.ensure_future(task.run({"command": "x"}))
                for _ in range(500):
                    await _real_sleep(0)
                    if state["parked"]:
                        break

            assert state["parked"], "startup crash never parked"
            assert state["transport_calls"] == 1, (
                f"startup crash burned {state['transport_calls']} attempts — "
                "should park immediately"
            )
            assert not run_task.done(), (
                "run task exited on a startup crash — the server is now unrevivable"
            )

            task._shutdown_event.set()
            task._reconnect_event.set()
            try:
                await asyncio.wait_for(run_task, timeout=15)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                run_task.cancel()

        asyncio.run(_scenario())

        park_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "permanent error" in r.getMessage()
        ]
        assert len(park_warnings) == 1
        message = park_warnings[0].getMessage()
        assert "StdioStartupCrashError" in message
        assert "ModuleNotFoundError: No module named 'yaml'" in message
