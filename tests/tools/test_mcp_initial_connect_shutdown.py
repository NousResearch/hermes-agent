"""Regression tests for initial MCP failure ownership and teardown."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest


def _reset_mcp_state(mcp_tool) -> None:
    mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()


def _cleanup_mcp_state(mcp_tool, extra_servers=()) -> None:
    with mcp_tool._lock:
        loop = mcp_tool._mcp_loop
    if loop is not None and loop.is_running():
        for server in extra_servers:
            task = getattr(server, "_task", None)
            if task is not None and not task.done():
                mcp_tool._run_on_mcp_loop(server.shutdown, timeout=5)
    mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()


def test_initial_connect_failure_is_registry_owned_and_reaped(monkeypatch, tmp_path):
    """Normal discovery must retain the parked task for clean shutdown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _FailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise ConnectionError("deterministic initial failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _FailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)

    real_stop = mcp_tool._stop_mcp_loop
    pending_at_stop = []

    async def _pending_tasks():
        current = asyncio.current_task()
        return sorted(
            task.get_coro().__qualname__
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        )

    def _observed_stop(*, only_if_idle=False):
        pending_at_stop.extend(
            mcp_tool._run_on_mcp_loop(_pending_tasks, timeout=5)
        )
        return real_stop(only_if_idle=only_if_idle)

    monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", _observed_stop)

    try:
        assert mcp_tool.register_mcp_servers({
            "initial-failure": {"command": "unused", "connect_timeout": 5}
        }) == []

        assert len(created) == 1
        server = created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["initial-failure"] is server
            assert "deterministic initial failure" in (
                mcp_tool._server_connect_errors["initial-failure"]
            )
        assert server._task is not None
        assert not server._task.done(), "recoverable initial failure was not parked"

        mcp_tool.shutdown_mcp_servers()

        assert pending_at_stop == [], (
            "shutdown left MCP tasks pending at loop stop: "
            f"{pending_at_stop!r}"
        )
        assert server._task.done()
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is None
            assert mcp_tool._mcp_thread is None
    finally:
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", real_stop)
        _cleanup_mcp_state(mcp_tool, created)


def test_initial_connect_failure_revives_same_registered_server(monkeypatch, tmp_path):
    """A cached parked failure must revive through register_mcp_servers()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.registry import ToolRegistry
    import tools.registry as registry_module

    _reset_mcp_state(mcp_tool)
    created = []
    backend_up = threading.Event()
    revived = threading.Event()
    state = {"transport_calls": 0, "tool_calls": 0}
    mock_registry = ToolRegistry()

    class _Session:
        async def call_tool(self, name, arguments):
            state["tool_calls"] += 1
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=f"revived:{arguments['value']}")],
                structuredContent=None,
            )

    class _RecoveringServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            assert mcp_tool._connect_server_claim.get() is None
            state["transport_calls"] += 1
            if not backend_up.is_set():
                raise ConnectionError("backend still booting")

            self.session = _Session()
            self._tools = [SimpleNamespace(
                name="ping",
                description="Return a deterministic revival result",
                inputSchema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )]
            # Match the real transports: discovery runs before _ready is set.
            self._register_discovered_tools_if_needed()
            self._ready.set()
            revived.set()
            return await self._wait_for_lifecycle_event()

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _RecoveringServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    monkeypatch.setattr(registry_module, "registry", mock_registry)

    config = {
        "recovering": {"command": "unused", "connect_timeout": 5}
    }

    try:
        assert mcp_tool.register_mcp_servers(config) == []
        assert len(created) == 1
        server = created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["recovering"] is server
            assert "backend still booting" in (
                mcp_tool._server_connect_errors["recovering"]
            )
        assert not server._task.done()

        backend_up.set()
        mcp_tool.register_mcp_servers(config)

        assert revived.wait(timeout=5), "cached parked server did not revive"
        assert len(created) == 1, "revival created a duplicate server task"
        with mcp_tool._lock:
            assert mcp_tool._servers["recovering"] is server
            assert "recovering" not in mcp_tool._server_connect_errors
        assert state["transport_calls"] == 2
        assert server.session is not None
        assert server._error is None

        entry = mock_registry.get_entry("mcp__recovering__ping")
        assert entry is not None
        assert entry.check_fn() is True
        assert json.loads(entry.handler({"value": "ok"})) == {
            "result": "revived:ok"
        }
        assert state["tool_calls"] == 1
    finally:
        _cleanup_mcp_state(mcp_tool, created)


def test_terminal_initial_failure_is_not_retained(monkeypatch, tmp_path):
    """A non-recoverable startup error must not leave a dead cache entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _AuthFailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise PermissionError("terminal authentication failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _AuthFailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_is_auth_error", lambda exc: True)

    try:
        assert mcp_tool.register_mcp_servers({
            "auth-failure": {"command": "unused", "connect_timeout": 5}
        }) == []
        assert len(created) == 1
        assert created[0]._task.done()
        with mcp_tool._lock:
            assert "auth-failure" not in mcp_tool._servers
            assert "terminal authentication failure" in (
                mcp_tool._server_connect_errors["auth-failure"]
            )
    finally:
        _cleanup_mcp_state(mcp_tool, created)


def test_standalone_failed_connect_is_reaped_without_global_owner(monkeypatch, tmp_path):
    """Probe-only _connect_server failures must not publish parked servers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _ProbeServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise ConnectionError("probe target unavailable")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _ProbeServerTask)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    mcp_tool._ensure_mcp_loop()

    try:
        with pytest.raises(ConnectionError, match="probe target unavailable"):
            mcp_tool._run_on_mcp_loop(
                lambda: mcp_tool._connect_server(
                    "probe-only", {"command": "unused"}
                ),
                timeout=5,
            )

        assert len(created) == 1
        assert created[0]._task.done()
        with mcp_tool._lock:
            assert "probe-only" not in mcp_tool._servers
            assert "probe-only" not in mcp_tool._server_connect_errors
    finally:
        _cleanup_mcp_state(mcp_tool, created)
