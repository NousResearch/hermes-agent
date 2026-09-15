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


class TestLazyServerIsReportedAsWorking:
    """A lazily registered server has live, callable cached tools and no process yet.

    Before the ``lazy`` status every surface misread it: ``get_mcp_status`` said ``configured``
    (never started) and the discovery summary said ``MCP: 0 tool(s) from 0 server(s) (2 failed)``
    right after registering every cached tool, observed verbatim on a live install.
    """

    _CONFIG = {
        "playwright": {"command": "npx", "args": ["-y", "@playwright/mcp"], "lazy": True},
    }

    @pytest.fixture(autouse=True)
    def _isolate_error_state(self):
        # The module fixture does not restore the connect-error and cooldown registries, and the
        # first-use failure tests above leave a "playwright" error behind.
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
             patch("tools.mcp_tool_discovery._discover_and_register_server", new_callable=AsyncMock), \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), patch("tools.mcp_tool_loop._run_on_mcp_loop"):
            _mcp_discovery.discover_mcp_tools()

    def _status(self):
        with patch("tools.mcp_tool_config._load_mcp_config", return_value=dict(self._CONFIG)):
            [entry] = _mcp_discovery.get_mcp_status()
        return entry

    def test_a_lazy_server_reads_as_working_on_every_surface(self, caplog):
        from hermes_cli import banner

        with caplog.at_level("INFO", logger="tools.mcp_tool"):
            self._discover()
        summaries = [r.getMessage() for r in caplog.records if "server(s)" in r.getMessage()]
        assert summaries == ["  MCP: 2 tool(s) from 1 server(s) (1 lazy, not spawned yet)"]
        caplog.clear()
        with caplog.at_level("INFO", logger="tools.mcp_tool"):
            self._discover()  # a repeat run neither re-announces nor fails it
        assert not any("failed" in r.getMessage() or "server(s)" in r.getMessage() for r in caplog.records)

        entry = self._status()
        assert (entry["status"], entry["tools"], entry["connected"]) == ("lazy", 2, False)
        line = banner._mcp_server_line(entry, dim="dim", text="text")
        assert "2 tool(s)" in line and "lazy, starts on first use" in line and "failed" not in line

        # Controls: a first-use connect in flight or failed outranks lazy, and once a connect has
        # succeeded (the parked config is gone) a stale tool list is not lazy.
        mcp._server_connecting.add("playwright")
        assert self._status()["status"] == "connecting"
        mcp._server_connecting.discard("playwright")
        mcp._server_connect_errors["playwright"] = "spawn failed: npx not found"
        assert self._status()["status"] == "failed"
        del mcp._server_connect_errors["playwright"]
        mcp._lazy_server_configs.pop("playwright")
        assert self._status()["status"] == "configured"

    def test_lazy_state_is_read_by_server_key_and_scoped_to_its_owner(self):
        """Under a multiplexer the lazy dicts are keyed by resolved server key, not by name.

        Two things this pins, both invisible without a multiplexer (where key == name):
        reading them by NAME finds nothing for a scoped profile, and gating them with
        ``_server_visible_in_scope`` — the predicate the LIVE dicts use — hides the server
        from the profile that owns it, because a lazy registration never populates the
        adoption/teardown maps that predicate reads.
        """
        from tools.mcp_tool_scope import _server_key

        key = _server_key("playwright", "profile-b", current=False)
        assert key != "playwright"
        mcp._lazy_server_configs[key] = dict(self._CONFIG["playwright"])
        mcp._lazy_server_fingerprints[key] = "abc"
        mcp._lazy_server_tool_names[key] = ["mcp_playwright_browser_navigate",
                                            "mcp_playwright_browser_click"]

        def status_from(scope):
            with patch.object(mcp, "_mcp_registry_scope", return_value=scope):
                [entry] = _mcp_discovery.get_mcp_status(configured=dict(self._CONFIG))
            return entry

        owner = status_from("profile-b")
        assert owner["status"] == "lazy"
        assert owner["tools"] == 2

        # A different profile must not see another profile's lazy registration.
        other = status_from("profile-a")
        assert other["status"] == "configured"
        assert other["tools"] == 0

        # ``include_runtime=False`` hides it from the owner too.
        with patch.object(mcp, "_mcp_registry_scope", return_value="profile-b"):
            [no_runtime] = _mcp_discovery.get_mcp_status(configured=dict(self._CONFIG),
                                                         include_runtime=False)
        assert no_runtime["status"] == "configured"
