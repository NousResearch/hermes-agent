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
    def test_lazy_registration_reports_lazy_with_cached_tools_not_failed(self, caplog):
        """A ``lazy: true`` server registered from the schema cache is a working server: status
        ``lazy`` with its cached tool count (``connected`` False), and the discovery summary counts
        it as a lazy server, never as failed (#111717). Controls: an unregistered eager server stays
        ``configured``; a live one stays ``connected``."""
        import logging

        from tools import mcp_tool_config as _mcp_config
        from tools import mcp_tool_loop as _mcp_loop
        from tools.mcp_tool_scope import _server_key

        servers = _lazy_config()
        controls = {"eager": {"command": "/nonexistent/eager"}, "live": {"command": "/nonexistent/live"}}
        live = SimpleNamespace(session=object(), _registered_tool_names=["l1", "l2"], _sampling=None, _tools=[])
        cached = ["mcp_playwright_browser_navigate", "mcp_playwright_browser_click"]

        def _fake_register(name, cfg, entry):
            key = _server_key(name)
            mcp._lazy_server_configs[key] = dict(cfg)
            mcp._lazy_server_tool_names[key] = list(cached)
            return list(cached)

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch.object(_mcp_config, "_load_mcp_config", return_value=dict(servers)), \
             patch.object(_mcp_loop, "_try_acquire_mcp_discovery_lock", return_value=mcp._LOCK_UNAVAILABLE), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=_fake_cache_entry()), \
             patch("tools.mcp_tool_registration._register_from_cache_sync", side_effect=_fake_register), \
             patch("tools.mcp_tool_discovery._discover_and_register_server", new_callable=AsyncMock), \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), patch("tools.mcp_tool_loop._run_on_mcp_loop"), \
             caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
            _mcp_discovery.discover_mcp_tools()
            mcp._servers[_server_key("live")] = live
            status = {e["name"]: e for e in _mcp_discovery.get_mcp_status({**servers, **controls})}

        summaries = [r.getMessage() for r in caplog.records if "tool(s) from" in r.getMessage()]
        assert summaries and all("failed" not in m for m in summaries), summaries
        assert any("1 lazy, not spawned yet" in m for m in summaries), summaries
        assert (status["playwright"]["status"], status["playwright"]["tools"],
                status["playwright"]["connected"]) == ("lazy", len(cached), False)
        assert status["eager"]["status"] == "configured" and status["eager"]["tools"] == 0
        assert status["live"]["status"] == "connected" and status["live"]["tools"] == 2


class TestStaleCacheEntryIsNotServed:
    """A manifest an OLDER build wrote must not be trusted by the lazy path.

    Measured 2026-09-17 (mcp 2.0.0): the pre-fix hint reader persisted
    ``annotations.readOnlyHint: false`` for EVERY read-only tool — the operator's live ``gmail`` row
    read ``{"name": "list_labels", "annotations": {"readOnlyHint": false}}`` while the server
    declared it read-only. A ``lazy: true`` server serves its manifest without ever connecting, and
    its first call is gated by the very annotation the stale row gets wrong, so the wrong value would
    outlive the fix. The entry_version stamp makes such an entry a MISS — which falls back to the
    eager connect whose write-through rewrites the row (``_register_lazy_from_cache``).
    """

    # The exact pre-fix row shape: every hint false, and no entry_version stamp.
    _STALE_ROWS = [
        {"name": "list_labels", "description": "", "inputSchema": {"type": "object", "properties": {}},
         "annotations": {"readOnlyHint": False}},
        {"name": "create_draft", "description": "", "inputSchema": {"type": "object", "properties": {}},
         "annotations": {"readOnlyHint": False}},
    ]

    def _cache_with(self, monkeypatch, tmp_path, **entry_extra):
        """Write a real cache file for the lazy fixture server; returns (path, fingerprint)."""
        import tools.mcp_schema_cache as msc
        from tools.mcp_schema_cache import config_fingerprint

        fingerprint = config_fingerprint(_lazy_config()["playwright"])
        entry = {"fingerprint": fingerprint, "tools": list(self._STALE_ROWS),
                 "utility_tools": [], **entry_extra}
        path = tmp_path / "mcp_schema_cache.json"
        path.write_text(json.dumps({"playwright": entry}), encoding="utf-8")
        monkeypatch.setattr(msc, "_cache_path", lambda: path)
        return path, fingerprint

    def test_unversioned_entry_is_a_miss_and_falls_back_to_eager(self, monkeypatch, tmp_path):
        self._cache_with(monkeypatch, tmp_path)  # no entry_version: written by a pre-fix build
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool_registration._register_from_cache_sync") as lazy_register, \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop") as run_on_loop:

            _mcp_discovery.register_mcp_servers(_lazy_config())

        lazy_register.assert_not_called()  # a pre-fix manifest must not be trusted by the lazy path
        run_on_loop.assert_called_once()  # eager connect -> write-through refreshes the row

    def test_current_version_entry_is_still_served_lazily(self, monkeypatch, tmp_path):
        import tools.mcp_schema_cache as msc

        self._cache_with(monkeypatch, tmp_path, entry_version=msc._ENTRY_VERSION)
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool_registration._register_from_cache_sync",
                   return_value=["mcp_playwright_list_labels"]) as lazy_register, \
             patch("tools.mcp_tool_loop._ensure_mcp_loop"), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop") as run_on_loop:

            _mcp_discovery.register_mcp_servers(_lazy_config())

        lazy_register.assert_called_once()
        run_on_loop.assert_not_called()
        # the current-version entry itself was handed to the lazy registration
        assert lazy_register.call_args[0][2]["entry_version"] == msc._ENTRY_VERSION

    def test_write_through_heals_the_row_and_it_is_trusted_again(self, monkeypatch, tmp_path):
        """The self-heal, without a hand-edit: miss -> eager connect -> rewritten row is served."""
        import tools.mcp_schema_cache as msc

        self._cache_with(monkeypatch, tmp_path)
        from tools.mcp_schema_cache import config_fingerprint

        fingerprint = config_fingerprint(_lazy_config()["playwright"])
        assert msc.get_cached_entry("playwright", fingerprint) is None, "stale row was served"

        # what the eager connect's write-through does once the hint reader sees the 2.x spelling (#88858)
        msc.write_cache_entry("playwright", fingerprint, tools=[
            {"name": "list_labels", "description": "", "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": True}},
            {"name": "create_draft", "description": "", "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": False}},
        ], utility_tools=[])

        entry = msc.get_cached_entry("playwright", fingerprint)
        assert entry is not None and entry["entry_version"] == msc._ENTRY_VERSION
        rows = {r["name"]: r["annotations"]["readOnlyHint"] for r in msc.tools_from_cache_entry(entry)}
        assert rows == {"list_labels": True, "create_draft": False}
