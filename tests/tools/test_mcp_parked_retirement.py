"""Regression tests for parked-server retirement (#71948).

A chronically dead MCP server — one whose initial connection fails and whose
subsequent parked self-probes keep failing — must retire (deregister + stop)
after ``_PARKED_PROBE_RETIRE_LIMIT`` failed probes instead of self-probing
every ``_PARKED_RETRY_INTERVAL`` forever (the node_repl retry storm reported
from ChatGPT desktop app ACP injection).
"""

import asyncio
import pytest

from tools import mcp_tool


def _reset_mcp_state(mcp_tool) -> None:
    mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()
        mcp_tool._server_connect_retry_after.clear()
        mcp_tool._server_connect_failures.clear()


def _cleanup_mcp_state(mcp_tool, created=()) -> None:
    try:
        mcp_tool.shutdown_mcp_servers()
    except Exception:
        pass
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()
        mcp_tool._server_connect_retry_after.clear()
        mcp_tool._server_connect_failures.clear()
    for server in created:
        task = getattr(server, "_task", None)
        if task is not None and not task.done():
            task.cancel()


def _wait_until(predicate, timeout=10.0, interval=0.02) -> bool:
    """Poll a predicate across threads (the MCP task runs on its own loop)."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class _FailingServerTask(mcp_tool.MCPServerTask):
    """MCPServerTask subclass whose stdio transport always fails."""

    created = []

    def __init__(self, name):
        self.__class__.created.append(self)
        super().__init__(name)

    async def _run_stdio(self, config):
        raise ConnectionError("deterministic connection failure")


def test_parked_server_retires_after_failed_probes(monkeypatch, tmp_path):
    """A parked server whose self-probes keep failing retires: task ends,
    entry removed from _servers, tracking maps cleaned."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    _reset_mcp_state(mcp_tool)
    _FailingServerTask.created = []

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _FailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)
    monkeypatch.setattr(mcp_tool, "_PARKED_PROBE_RETIRE_LIMIT", 2)

    try:
        assert mcp_tool.register_mcp_servers({
            "dead-server": {"command": "unused", "connect_timeout": 5}
        }) == []

        assert len(_FailingServerTask.created) == 1
        server = _FailingServerTask.created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["dead-server"] is server

        # Wait for the retirement path: 3 probe wakes (limit 2) with a
        # 20ms park interval, then the task exits.
        assert _wait_until(lambda: server._task.done()), (
            "server task did not retire"
        )

        with mcp_tool._lock:
            assert "dead-server" not in mcp_tool._servers, (
                "retired server must be removed from _servers"
            )
            assert "dead-server" not in mcp_tool._server_connect_errors, (
                "retired server's connect error must be cleaned up"
            )
            assert "dead-server" not in mcp_tool._server_connecting
        assert server._parked_probe_failures > mcp_tool._PARKED_PROBE_RETIRE_LIMIT
        assert server._task.done(), "retired task must have exited"
    finally:
        _cleanup_mcp_state(mcp_tool, _FailingServerTask.created)


def test_retired_server_revives_via_re_registration(monkeypatch, tmp_path):
    """After retirement a fresh registration creates a NEW task — the
    explicit-revive path (/mcp refresh, next ACP session)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    _reset_mcp_state(mcp_tool)
    _FailingServerTask.created = []

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _FailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)
    monkeypatch.setattr(mcp_tool, "_PARKED_PROBE_RETIRE_LIMIT", 1)
    # No cooldown monkeypatch: retirement must clear the cooldown maps so a
    # re-registration proceeds immediately (validates _retire_server cleanup).

    try:
        assert mcp_tool.register_mcp_servers({
            "dead-server": {"command": "unused", "connect_timeout": 5}
        }) == []
        first = _FailingServerTask.created[0]
        assert _wait_until(lambda: first._task.done()), (
            "server task did not retire"
        )
        with mcp_tool._lock:
            assert "dead-server" not in mcp_tool._servers

        # Re-registration creates a fresh task (revive path).
        mcp_tool.register_mcp_servers({
            "dead-server": {"command": "unused", "connect_timeout": 5}
        })
        assert len(_FailingServerTask.created) == 2
        second = _FailingServerTask.created[1]
        assert second is not first
        with mcp_tool._lock:
            assert mcp_tool._servers["dead-server"] is second
        # Let the second task finish its own retirement.
        assert _wait_until(lambda: second._task.done()), (
            "re-registered server task did not retire"
        )
    finally:
        _cleanup_mcp_state(mcp_tool, _FailingServerTask.created)


def test_wait_wake_classification_explicit_vs_probe():
    """_wait_for_reconnect_or_shutdown distinguishes an explicit reconnect
    request from a timed self-probe wake — the distinction that drives the
    parked-probe counter (#71948)."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("classify")

    async def _drive():
        server._reconnect_event.set()
        assert await server._wait_for_reconnect_or_shutdown(timeout=5) == (
            "reconnect"
        )
        # Event was cleared by the previous wake; a fresh wait times out.
        assert await server._wait_for_reconnect_or_shutdown(timeout=0.05) == (
            "probe"
        )
        # Shutdown takes precedence.
        server._shutdown_event.set()
        assert await server._wait_for_reconnect_or_shutdown(timeout=5) == (
            "shutdown"
        )

    asyncio.run(_drive())


def test_retirement_emits_warning_and_stops_probing(monkeypatch, tmp_path, caplog):
    """Retirement is announced with a WARNING and the probe cycle stops —
    the log-storm suppression property of #71948."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import logging

    _reset_mcp_state(mcp_tool)
    _FailingServerTask.created = []

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _FailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)
    monkeypatch.setattr(mcp_tool, "_PARKED_PROBE_RETIRE_LIMIT", 1)

    try:
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            assert mcp_tool.register_mcp_servers({
                "dead-server": {"command": "unused", "connect_timeout": 5}
            }) == []
            server = _FailingServerTask.created[0]
            assert _wait_until(lambda: server._task.done()), (
                "server task did not retire"
            )

        assert any(
            "retiring" in record.message
            and "dead-server" in record.message
            for record in caplog.records
        ), "retirement must be announced with a WARNING"
    finally:
        _cleanup_mcp_state(mcp_tool, _FailingServerTask.created)
