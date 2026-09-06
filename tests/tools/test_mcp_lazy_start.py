"""Behavior-contract tests for lazy MCP server startup (#56832).

A server configured with ``lazy: true`` whose config fingerprint matches an
on-disk schema-cache entry registers its tools WITHOUT spawning/connecting;
the first real call (raw tool OR resource/prompt utility) routes through the
existing connect path.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.mcp_tool as mcp
from tools import mcp_tool_discovery as _mcp_discovery
from tools import mcp_tool_handlers as _mcp_handlers
from tools import mcp_tool_loop as _mcp_loop
from tools import mcp_tool_registration as _mcp_registration
from tools import mcp_tool_schema as _mcp_schema


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    old_servers = dict(mcp._servers)
    old_lazy = dict(mcp._lazy_server_configs)
    old_fps = dict(mcp._lazy_server_fingerprints)
    old_names = dict(mcp._lazy_server_tool_names)
    old_connecting = set(mcp._server_connecting)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._lazy_server_configs.clear()
    mcp._lazy_server_configs.update(old_lazy)
    mcp._lazy_server_fingerprints.clear()
    mcp._lazy_server_fingerprints.update(old_fps)
    mcp._lazy_server_tool_names.clear()
    mcp._lazy_server_tool_names.update(old_names)
    mcp._server_connecting.clear()
    mcp._server_connecting.update(old_connecting)


def _fake_cache_entry():
    return {
        "fingerprint": "abc",
        "tools": [
            {
                "name": "browser_navigate",
                "description": "Navigate",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
        "utility_tools": [],
    }


def _lazy_config():
    return {
        "playwright": {
            "command": "npx",
            "args": ["-y", "@playwright/mcp"],
            "lazy": True,
        }
    }


class TestLazyMcpRegistration:
    def test_registers_from_cache_without_connect(self):
        config = _lazy_config()
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=_fake_cache_entry()), \
             patch(
                 "tools.mcp_tool_registration._register_from_cache_sync",
                 return_value=["mcp_playwright_browser_navigate"],
             ) as mock_register, \
             patch("tools.mcp_tool_discovery._discover_and_register_server", new_callable=AsyncMock) as mock_discover, \
             patch("tools.mcp_tool_loop._ensure_mcp_loop") as mock_loop, \
             patch("tools.mcp_tool_loop._run_on_mcp_loop") as mock_run:

            _mcp_discovery.register_mcp_servers(config)

        mock_register.assert_called_once()
        mock_discover.assert_not_called()
        mock_run.assert_not_called()
        mock_loop.assert_not_called()

    def test_cache_miss_falls_back_to_eager_connect(self):
        config = _lazy_config()
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=None), \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop") as mock_run:

            _mcp_discovery.register_mcp_servers(config)

        mock_run.assert_called_once()

    def test_non_lazy_server_never_touches_cache(self):
        config = {"playwright": {"command": "npx", "args": []}}
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.get_cached_entry") as mock_get, \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop") as mock_run:

            _mcp_discovery.register_mcp_servers(config)

        mock_get.assert_not_called()
        mock_run.assert_called_once()

    def test_lazy_server_not_reregistered_on_second_pass(self):
        config = _lazy_config()
        mcp._lazy_server_configs["playwright"] = dict(config["playwright"])
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool_registration._register_from_cache_sync") as mock_register, \
             patch("tools.mcp_tool_loop._run_on_mcp_loop") as mock_run:

            names = _mcp_discovery.register_mcp_servers(config)

        mock_register.assert_not_called()
        mock_run.assert_not_called()
        assert "mcp_playwright_browser_navigate" in names


class TestLazyFirstUseConnect:
    def _connected_server(self):
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(isError=False, content=[], structuredContent=None)
        )
        connected = SimpleNamespace(
            session=mock_session,
            _rpc_lock=MagicMock(),
            _pending_call_context=None,
        )
        connected._rpc_lock.__aenter__ = AsyncMock(return_value=None)
        connected._rpc_lock.__aexit__ = AsyncMock(return_value=None)
        return connected

    @staticmethod
    def _run_on_loop(coro_or_factory, timeout=120):
        import asyncio

        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_tool_handler_lazy_connects_on_first_call(self):
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"

        connected = self._connected_server()

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(_mcp_discovery, "_ensure_lazy_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(_mcp_loop, "_run_on_mcp_loop", side_effect=self._run_on_loop):
            handler = _mcp_handlers._make_tool_handler("playwright", "browser_navigate", 5)
            out = handler({}, task_id="t1")

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload.get("result") == ""

    def test_list_resources_handler_lazy_connects_on_first_call(self):
        # Regression for the resource/prompt gap: utility handlers must also
        # route through the first-use connect path, or the first
        # list_resources/get_prompt on a lazy server fails.
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)

        connected = self._connected_server()
        connected.session.list_resources = AsyncMock()

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        async def _fake_paginate(list_method, items_attr, server_name):
            return [SimpleNamespace(uri="file:///a", name="a", description="", mimeType="")]

        with patch.object(_mcp_discovery, "_ensure_lazy_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(mcp, "_paginate_full_list", side_effect=_fake_paginate), \
             patch.object(_mcp_loop, "_run_on_mcp_loop", side_effect=self._run_on_loop):
            handler = _mcp_handlers._make_list_resources_handler("playwright", 5)
            out = handler({})

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload["resources"][0]["uri"] == "file:///a"

    def test_get_prompt_handler_lazy_connects_on_first_call(self):
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)

        connected = self._connected_server()
        connected.session.get_prompt = AsyncMock(
            return_value=SimpleNamespace(messages=[])
        )

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(_mcp_discovery, "_ensure_lazy_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(_mcp_loop, "_run_on_mcp_loop", side_effect=self._run_on_loop):
            handler = _mcp_handlers._make_get_prompt_handler("playwright", 5)
            out = handler({"name": "greeting"})

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload

    def test_check_fn_passes_for_lazy_registered_server(self):
        mcp._lazy_server_configs["playwright"] = {"lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        assert _mcp_handlers._make_check_fn("playwright")() is True

    def test_check_fn_fails_for_unknown_server(self):
        assert _mcp_handlers._make_check_fn("nope")() is False

    def test_lazy_connect_respects_connect_cooldown(self):
        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}
        with patch.object(_mcp_discovery, "_connect_cooldown_active", return_value=True), \
             patch.object(_mcp_loop, "_run_on_mcp_loop") as mock_run:
            assert _mcp_discovery._ensure_lazy_server_connected("playwright") is False
        mock_run.assert_not_called()

    def test_lazy_connect_success_clears_lazy_state(self):
        config = {"command": "npx", "lazy": True}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]

        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=["mcp_playwright_browser_navigate"],
        )

        def _fake_run(coro_or_factory, timeout=30):
            mcp._servers["playwright"] = connected
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            coro.close()
            return ["mcp_playwright_browser_navigate"]

        with patch.object(_mcp_loop, "_ensure_mcp_loop"), \
             patch.object(_mcp_loop, "_run_on_mcp_loop", side_effect=_fake_run):
            assert _mcp_discovery._ensure_lazy_server_connected("playwright") is True

        assert "playwright" not in mcp._lazy_server_configs
        assert "playwright" not in mcp._lazy_server_fingerprints
        assert "playwright" not in mcp._lazy_server_tool_names

    def test_lazy_connect_deregisters_phantom_cached_tools(self):
        # Stale-cache reconciliation: the cached manifest advertised tool X,
        # but the live server only registers tool Y → X must be deregistered
        # after the first-use connect so the model stops seeing a phantom.
        from tools.registry import registry

        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "stale-fp"
        mcp._lazy_server_tool_names["playwright"] = [
            "mcp_playwright_tool_x",
            "mcp_playwright_tool_y",
        ]

        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=["mcp_playwright_tool_y"],
        )

        def _fake_run(coro_or_factory, timeout=30):
            mcp._servers["playwright"] = connected
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            coro.close()
            return ["mcp_playwright_tool_y"]

        with patch.object(_mcp_loop, "_ensure_mcp_loop"), \
             patch.object(_mcp_loop, "_run_on_mcp_loop", side_effect=_fake_run), \
             patch.object(registry, "deregister") as mock_dereg:
            assert _mcp_discovery._ensure_lazy_server_connected("playwright") is True

        mock_dereg.assert_called_once_with("mcp_playwright_tool_x", scope=None)

    def test_lazy_connect_failure_records_cooldown(self):
        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}

        def _fake_run(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            coro.close()
            raise RuntimeError("spawn failed")

        with patch.object(_mcp_loop, "_ensure_mcp_loop"), \
             patch.object(_mcp_loop, "_run_on_mcp_loop", side_effect=_fake_run), \
             patch.object(_mcp_discovery, "_record_connect_failure") as mock_record:
            assert _mcp_discovery._ensure_lazy_server_connected("playwright") is False

        mock_record.assert_called_once_with("playwright")
        # Config retained so a later call can retry after cooldown.
        assert "playwright" in mcp._lazy_server_configs


class TestCacheLoadDescriptionScan:
    def test_scan_runs_on_cache_load_path(self):
        # Defense-in-depth: the cache file is user-writable JSON, so the
        # cache-load registration path must run the same injection scan as
        # eager discovery.
        entry = _fake_cache_entry()
        config = {"command": "npx", "args": [], "lazy": True}
        with patch.object(_mcp_schema, "_scan_mcp_description", return_value=[]) as mock_scan, \
             patch.object(_mcp_schema, "_convert_mcp_schema", side_effect=RuntimeError("stop")), \
             pytest.raises(RuntimeError):
            _mcp_registration._register_from_cache_sync("playwright", config, entry)

        mock_scan.assert_called_once_with("playwright", "browser_navigate", "Navigate")


class TestResolveServerLazy:
    def test_default_off(self):
        assert _mcp_discovery._resolve_server_lazy("s", {"command": "npx"}) is False

    def test_explicit_true(self):
        assert _mcp_discovery._resolve_server_lazy("s", {"command": "npx", "lazy": True}) is True

    def test_explicit_false(self):
        assert _mcp_discovery._resolve_server_lazy("s", {"command": "npx", "lazy": False}) is False


class TestLazyMcpStatus:
    """``get_mcp_status`` must tell lazy servers apart from never-started ones.

    ``_register_from_cache_sync`` records a lazily registered server in
    ``_lazy_server_configs`` / ``_lazy_server_tool_names`` and puts nothing in
    ``_servers`` (no process, no session). Before the ``lazy`` status such a
    server reported ``status: "configured", tools: 0`` — the same shape as a
    server that never started — even though its cached tools were live and
    callable, and startup code counting ``connected`` treated an all-lazy
    discovery as one that connected nothing.
    """

    _CONFIG = {
        "playwright": {"command": "npx", "args": ["-y", "@playwright/mcp"], "lazy": True},
    }

    @pytest.fixture(autouse=True)
    def _isolate_error_state(self):
        # The module fixture restores _servers / _lazy_* / _server_connecting but
        # not the connect-error and cooldown registries, and the first-use
        # failure tests above leave a "playwright" error behind. Status is the
        # first thing in this file that reads them, so isolate them here.
        old_errors = dict(mcp._server_connect_errors)
        old_retry = dict(mcp._server_connect_retry_after)
        mcp._server_connect_errors.pop("playwright", None)
        mcp._server_connect_retry_after.pop("playwright", None)
        yield
        mcp._server_connect_errors.clear()
        mcp._server_connect_errors.update(old_errors)
        mcp._server_connect_retry_after.clear()
        mcp._server_connect_retry_after.update(old_retry)

    def _seed_lazy(self, name="playwright", tools=("mcp_playwright_browser_navigate", "mcp_playwright_browser_click")):
        mcp._lazy_server_configs[name] = dict(self._CONFIG[name])
        mcp._lazy_server_fingerprints[name] = "abc"
        mcp._lazy_server_tool_names[name] = list(tools)

    def test_lazy_server_reports_lazy_status_with_cached_tool_count(self):
        self._seed_lazy()
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)):
            [entry] = _mcp_discovery.get_mcp_status()

        assert entry["name"] == "playwright"
        assert entry["status"] == "lazy"
        assert entry["tools"] == 2
        assert entry["connected"] is False
        assert entry["disabled"] is False
        assert entry["transport"] == "stdio"

    def test_never_started_server_is_still_configured(self):
        # No lazy registration happened: nothing changes for the pre-existing shape.
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)):
            [entry] = _mcp_discovery.get_mcp_status()

        assert entry["status"] == "configured"
        assert entry["tools"] == 0

    def test_first_use_connect_in_flight_reports_connecting(self):
        # _ensure_lazy_server_connected adds the name to _server_connecting while
        # the deferred spawn runs; that transient state is the more current one.
        self._seed_lazy()
        mcp._server_connecting.add("playwright")
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)):
            [entry] = _mcp_discovery.get_mcp_status()

        assert entry["status"] == "connecting"

    def test_failed_first_use_connect_reports_failed_with_error(self):
        # A failed first-use connect records the error and keeps the lazy
        # registration (the next call retries after the cooldown); the recorded
        # failure is the actionable state to show.
        self._seed_lazy()
        mcp._server_connect_errors["playwright"] = "spawn failed: npx not found"
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)):
            [entry] = _mcp_discovery.get_mcp_status()

        assert entry["status"] == "failed"
        assert entry["error"] == "spawn failed: npx not found"

    def test_stale_tool_names_without_a_lazy_config_are_not_lazy(self):
        # _ensure_lazy_server_connected pops the config on a successful connect;
        # only a server still parked in _lazy_server_configs is lazy.
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)):
            [entry] = _mcp_discovery.get_mcp_status()

        assert entry["status"] == "configured"


class TestLazyDiscoverySummary:
    """``discover_mcp_tools`` must not report lazy servers as failed.

    The summary log counted "connected" as ``name in _servers`` and everything
    else as failed, so a fully lazy startup — the memory-saving configuration —
    logged ``MCP: 0 tool(s) from 0 server(s) (2 failed)`` right after
    registering every cached tool. Observed verbatim on a live install.
    """

    _CONFIG = {
        "playwright": {"command": "npx", "args": ["-y", "@playwright/mcp"], "lazy": True},
    }

    @pytest.fixture(autouse=True)
    def _isolate_error_state(self):
        old_errors = dict(mcp._server_connect_errors)
        old_retry = dict(mcp._server_connect_retry_after)
        mcp._server_connect_errors.pop("playwright", None)
        mcp._server_connect_retry_after.pop("playwright", None)
        yield
        mcp._server_connect_errors.clear()
        mcp._server_connect_errors.update(old_errors)
        mcp._server_connect_retry_after.clear()
        mcp._server_connect_retry_after.update(old_retry)

    def _fake_register_from_cache(self, name, cfg, entry):
        # What the real _register_from_cache_sync records, minus the registry writes.
        names = ["mcp_playwright_browser_navigate", "mcp_playwright_browser_click"]
        with mcp._lock:
            mcp._lazy_server_configs[name] = dict(cfg)
            mcp._lazy_server_fingerprints[name] = "abc"
            mcp._lazy_server_tool_names[name] = list(names)
        return names

    def _discover(self):
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)), \
             patch("tools.mcp_tool._ensure_mcp_sdk", return_value=True), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=_fake_cache_entry()), \
             patch("tools.mcp_tool_registration._register_from_cache_sync", side_effect=self._fake_register_from_cache), \
             patch("tools.mcp_tool_discovery._discover_and_register_server", new_callable=AsyncMock) as mock_discover, \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), patch("tools.mcp_tool_loop._run_on_mcp_loop"):
            _mcp_discovery.discover_mcp_tools()
        return mock_discover

    def test_lazy_startup_summary_counts_cached_tools_not_failures(self, caplog):
        with caplog.at_level("INFO", logger="tools.mcp_tool"):
            mock_discover = self._discover()

        mock_discover.assert_not_called()
        # register_mcp_servers logs its own "MCP: registered N lazy tool(s) from
        # schema cache" line; the discovery summary is the only "... server(s)" one.
        summaries = [r.getMessage() for r in caplog.records if "server(s)" in r.getMessage()]
        assert summaries == ["  MCP: 2 tool(s) from 1 server(s) (1 lazy, not spawned yet)"]
        assert not any("failed" in r.getMessage() for r in caplog.records)

    def test_repeated_discovery_does_not_reannounce_or_fail_lazy_servers(self, caplog):
        self._discover()
        caplog.clear()
        with caplog.at_level("INFO", logger="tools.mcp_tool"):
            self._discover()

        assert not any("failed" in r.getMessage() for r in caplog.records)
        assert not any("server(s)" in r.getMessage() for r in caplog.records)
