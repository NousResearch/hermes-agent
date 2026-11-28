"""A status probe must reuse a live same-identity MCP connection, never spawn a second one.

A stdio server that pins its data home (x-use's ``X_USE_HOME``) refuses to start twice: the second
child exits before the handshake and the client only sees ``MCPError: Connection closed``, which the
connector panel renders as a permanent "Connecting…". The live connection already answers what the
probe asks, so it is reused instead.
"""

import asyncio

import pytest

from tools import mcp_tool as _core
from tools import mcp_tool_discovery as _discovery


STDIO_CONFIG = {
    "command": "C:/Users/tester/.local/bin/x-use.exe",
    "args": ["mcp"],
    "env": {"X_USE_HOME": "C:/Users/tester/AppData/Local/x-use"},
}


class _Tool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class _Session:
    """A live SDK session: no prompt/resource capability, so those probes stay absent."""

    async def list_prompts(self):
        raise RuntimeError("prompts unsupported")

    async def list_resources(self):
        raise RuntimeError("resources unsupported")


def _make_live_server(name: str, config: dict, tools=(("get_runtime_info", "Runtime info"),)):
    server = _core.MCPServerTask(name)
    server._config = dict(config)
    server.session = _Session()
    server._tools = [_Tool(n, d) for n, d in tools]
    server._ready.set()
    return server


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Isolate config reads and the module-global connection ledgers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    saved = {
        "servers": dict(_core._servers),
        "connecting": set(_core._server_connecting),
        "lazy": dict(_core._lazy_server_configs),
        "errors": dict(_core._server_connect_errors),
    }
    _core._servers.clear()
    _core._server_connecting.clear()
    _core._lazy_server_configs.clear()
    _core._server_connect_errors.clear()
    _discovery._PROBE_SPAWN_LOCKS.clear()
    from tools import mcp_tool_lifecycle, mcp_tool_loop

    monkeypatch.setattr(mcp_tool_loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(mcp_tool_lifecycle, "_stop_mcp_loop_if_idle", lambda: None)
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda coro, timeout=30: asyncio.run(coro))
    yield
    _core._servers.clear()
    _core._servers.update(saved["servers"])
    _core._server_connecting.clear()
    _core._server_connecting.update(saved["connecting"])
    _core._lazy_server_configs.clear()
    _core._lazy_server_configs.update(saved["lazy"])
    _core._server_connect_errors.clear()
    _core._server_connect_errors.update(saved["errors"])
    _discovery._PROBE_SPAWN_LOCKS.clear()


class TestProbeReusesLiveConnection:
    def test_live_same_identity_server_is_reused_without_spawning(self, monkeypatch):
        from hermes_cli.mcp_config import _probe_single_server

        server = _make_live_server("x-use", STDIO_CONFIG, tools=(("a", "A"), ("b", "B" * 120)))
        _core._servers["x-use"] = server

        def _boom(*_a, **_kw):
            raise AssertionError("probe spawned a second transport for a live server")

        monkeypatch.setattr(_discovery, "_connect_server", _boom)

        tools = _probe_single_server("x-use", dict(STDIO_CONFIG))

        assert [name for name, _ in tools] == ["a", "b"]
        assert tools[1][1].endswith("...") and len(tools[1][1]) == 80
        # The live connection is untouched: not shut down, still the registered entry.
        assert _core._servers["x-use"] is server
        assert server.session is not None

    def test_reused_connection_reports_details_flag(self, monkeypatch):
        from hermes_cli.mcp_config import _probe_single_server

        _core._servers["x-use"] = _make_live_server("x-use", STDIO_CONFIG)
        monkeypatch.setattr(_discovery, "_connect_server", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))

        details: dict = {}
        _probe_single_server("x-use", dict(STDIO_CONFIG), details=details)

        assert details.get("reused") is True
        assert "schema_chars" in details

    def test_different_env_identity_is_not_reused(self, monkeypatch):
        """A different X_USE_HOME is a different server; the probe must test what config says."""
        from hermes_cli.mcp_config import _probe_single_server

        _core._servers["x-use"] = _make_live_server(
            "x-use", {**STDIO_CONFIG, "env": {"X_USE_HOME": "C:/elsewhere"}}
        )
        calls: list[tuple] = []

        async def _fake_connect(name, config):
            calls.append((name, config))
            raise RuntimeError("spawn-path-reached")

        monkeypatch.setattr(_discovery, "_connect_server", _fake_connect)

        with pytest.raises(Exception) as excinfo:
            _probe_single_server("x-use", dict(STDIO_CONFIG))
        assert "spawn-path-reached" in str(excinfo.value)
        assert len(calls) == 1

    def test_held_stdio_server_that_is_down_is_reported_not_respawned(self, monkeypatch):
        """Parked / mid-reconnect: a second child would die on the same data-home lock."""
        from hermes_cli.mcp_config import _probe_single_server

        parked = _make_live_server("x-use", STDIO_CONFIG)
        parked.session = None
        parked._ready.clear()
        parked._park_reason = "connection closed"
        _core._servers["x-use"] = parked

        def _boom(*_a, **_kw):
            raise AssertionError("probe spawned a second transport for a held data home")

        monkeypatch.setattr(_discovery, "_connect_server", _boom)

        with pytest.raises(RuntimeError) as excinfo:
            _probe_single_server("x-use", dict(STDIO_CONFIG))
        message = str(excinfo.value)
        assert "already owned by this Hermes runtime" in message
        assert "connection closed" in message

    def test_held_http_server_that_is_down_still_probes(self, monkeypatch):
        """HTTP has no child process or data-home lock: a probe transport is harmless."""
        from hermes_cli.mcp_config import _probe_single_server

        http_config = {"url": "https://example.invalid/mcp"}
        down = _core.MCPServerTask("remote")
        down._config = dict(http_config)
        down.session = None
        _core._servers["remote"] = down
        calls: list[str] = []

        async def _fake_connect(name, config):
            calls.append(name)
            raise RuntimeError("connect-attempted")

        monkeypatch.setattr(_discovery, "_connect_server", _fake_connect)

        with pytest.raises(Exception) as excinfo:
            _probe_single_server("remote", dict(http_config))
        assert "connect-attempted" in str(excinfo.value)
        assert calls == ["remote"]

    def test_no_live_entry_takes_the_spawn_path(self, monkeypatch):
        from hermes_cli.mcp_config import _probe_single_server

        calls: list[str] = []

        async def _fake_connect(name, config):
            calls.append(name)
            raise RuntimeError("connect-attempted")

        monkeypatch.setattr(_discovery, "_connect_server", _fake_connect)

        with pytest.raises(Exception):
            _probe_single_server("x-use", dict(STDIO_CONFIG))
        assert calls == ["x-use"]


class TestReuseHelpers:
    def test_identity_includes_env(self):
        base = _discovery._probe_identity(STDIO_CONFIG)
        other_home = _discovery._probe_identity({**STDIO_CONFIG, "env": {"X_USE_HOME": "C:/other"}})
        assert base != other_home
        assert base == _discovery._probe_identity(dict(STDIO_CONFIG))

    def test_find_live_server_ignores_unready_and_mismatched(self):
        ready = _make_live_server("x-use", STDIO_CONFIG)
        _core._servers["x-use"] = ready
        assert _discovery.find_live_server_for_probe("x-use", dict(STDIO_CONFIG)) is ready

        ready._ready.clear()
        assert _discovery.find_live_server_for_probe("x-use", dict(STDIO_CONFIG)) is None
        ready._ready.set()
        assert _discovery.find_live_server_for_probe("x-use", {"command": "other"}) is None

    def test_probe_reuse_target_waits_for_an_in_flight_connect(self):
        async def _scenario():
            _core._server_connecting.add("x-use")
            task = asyncio.create_task(
                _discovery.probe_reuse_target("x-use", dict(STDIO_CONFIG), stdio=True, wait_seconds=5.0)
            )
            await asyncio.sleep(0.05)
            _core._servers["x-use"] = _make_live_server("x-use", STDIO_CONFIG)
            _core._server_connecting.discard("x-use")
            return await asyncio.wait_for(task, timeout=5.0)

        server, blocked = asyncio.run(_scenario())
        assert blocked is None
        assert server is _core._servers["x-use"]

    def test_probe_reuse_target_refuses_to_spawn_while_a_connect_is_still_running(self):
        async def _scenario():
            _core._server_connecting.add("x-use")
            return await _discovery.probe_reuse_target(
                "x-use", dict(STDIO_CONFIG), stdio=True, wait_seconds=0.2
            )

        server, blocked = asyncio.run(_scenario())
        assert server is None
        assert blocked and "still connecting" in blocked

    def test_probe_reuse_target_probes_http_while_a_connect_is_running(self):
        async def _scenario():
            _core._server_connecting.add("remote")
            return await _discovery.probe_reuse_target(
                "remote", {"url": "https://example.invalid/mcp"}, stdio=False, wait_seconds=0.2
            )

        server, blocked = asyncio.run(_scenario())
        assert (server, blocked) == (None, None)
