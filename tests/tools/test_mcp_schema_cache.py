"""Unit tests for the on-disk MCP schema cache (tools/mcp_schema_cache.py).

The module landed in #56832's extraction without its tests; these cover the
fingerprint keying, read/write round-trip, and invalidation behavior.
"""

import tools.mcp_schema_cache as msc
from tools import mcp_tool_registration as _mcp_registration


class TestConfigFingerprint:
    def test_stable_for_same_config(self):
        cfg = {"command": "npx", "args": ["-y", "@playwright/mcp"]}
        assert msc.config_fingerprint(cfg) == msc.config_fingerprint(dict(cfg))

    def test_changes_when_connection_config_changes(self):
        base = {"command": "npx", "args": ["-y", "@playwright/mcp"]}
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "args": ["-y", "@playwright/mcp", "--headless"]}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "command": "uvx"}
        )
        assert msc.config_fingerprint(base) != msc.config_fingerprint(
            {**base, "tools": {"include": ["a"]}}
        )

    def test_ignores_non_connection_keys(self):
        base = {"command": "npx", "args": []}
        assert msc.config_fingerprint(base) == msc.config_fingerprint(
            {**base, "timeout": 5, "enabled": True, "lazy": True}
        )


class TestCacheRoundTrip:
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")

    def test_write_then_read_with_matching_fingerprint(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        tools = [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]
        msc.write_cache_entry("srv", "fp1", tools=tools, utility_tools=[])
        entry = msc.get_cached_entry("srv", "fp1")
        assert entry is not None
        assert msc.tools_from_cache_entry(entry) == tools
        assert msc.utility_tools_from_cache_entry(entry) == []

    def test_fingerprint_mismatch_returns_none(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        msc.write_cache_entry("srv", "fp1", tools=[], utility_tools=[])
        assert msc.get_cached_entry("srv", "OTHER") is None

    def test_missing_server_returns_none(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        assert msc.get_cached_entry("nope", "fp") is None

    def test_corrupt_cache_file_is_tolerated(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / "cache.json").write_text("{not json", encoding="utf-8")
        assert msc.get_cached_entry("srv", "fp") is None
        # And writes recover the file.
        msc.write_cache_entry("srv", "fp", tools=[], utility_tools=[])
        assert msc.get_cached_entry("srv", "fp") is not None

    def test_malformed_entry_shapes_are_tolerated(self):
        assert msc.tools_from_cache_entry({"tools": "nope"}) == []
        assert msc.utility_tools_from_cache_entry({}) == []


class TestCacheFileLocation:
    def test_cache_lives_under_hermes_home_cache_dir_with_0600(
        self, monkeypatch, tmp_path
    ):
        # Real path (no _cache_path monkeypatch): HERMES_HOME/cache/…, 0o600,
        # matching the discovery-cache precedent in tools/registry.py.
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        path = msc._cache_path()
        assert path == tmp_path / "cache" / "mcp_schema_cache.json"
        msc.write_cache_entry("srv", "fp", tools=[], utility_tools=[])
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600


class TestWriteSkip:
    def test_identical_payload_skips_rewrite(self, monkeypatch, tmp_path):
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        saves = []
        real_save = msc._save_all

        def _counting_save(data):
            saves.append(1)
            real_save(data)

        monkeypatch.setattr(msc, "_save_all", _counting_save)
        tools = [{"name": "t1", "description": "d", "inputSchema": {}}]
        msc.write_cache_entry("srv", "fp1", tools=tools, utility_tools=[])
        assert len(saves) == 1
        # Identical payload (reconnect / list_changed refresh) → no rewrite.
        msc.write_cache_entry("srv", "fp1", tools=list(tools), utility_tools=[])
        assert len(saves) == 1
        # Changed payload → rewrite.
        msc.write_cache_entry("srv", "fp2", tools=tools, utility_tools=[])
        assert len(saves) == 2


class TestWriteThroughPreservesSchema:
    """Regression: the write-through path must persist real tool parameters.

    ``mcp`` 2.0 renamed ``Tool.inputSchema`` to ``input_schema``, keeping the
    camelCase spelling only as a *serialization* alias — pydantic aliases do
    not apply to attribute access, so ``getattr(tool, "inputSchema")`` returns
    None on 2.x instead of raising. The cache-write path used exactly that
    bare read, so every entry landed on disk with ``"inputSchema": {}``. A
    server later registered from that cache (``lazy: true``) was advertised to
    the model with every parameter stripped, which makes required-argument
    tools such as zhihu's ``zhida`` (``query`` + ``model`` both required)
    uncallable.

    These tests drive the live ``_register_server_tools`` write-through with a
    genuine SDK ``Tool`` so the field-rename is actually exercised — the mock
    fixtures elsewhere build ``SimpleNamespace`` objects and cannot catch it.
    (Salvaged from #91451 / #102129.)
    """

    _SCHEMA = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "model": {"type": "string"},
        },
        "required": ["query", "model"],
    }

    def _cache_write_through(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import MagicMock, patch

        from mcp.types import Tool

        import tools.mcp_tool as mt
        from tools.registry import ToolRegistry

        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        # Registration records per-server state in module globals (lazy tool names, trust
        # levels, read-only hints...); isolate them so the probe server never leaks into
        # later tests such as ``discover_mcp_tools() == []`` assertions.
        for attr in ("_lazy_server_tool_names", "_lazy_server_configs", "_lazy_server_fingerprints",
                     "_mcp_tool_server_names", "_server_trust_levels", "_tool_read_only_hints"):
            monkeypatch.setattr(mt, attr, {})
        server = mt.MCPServerTask("probe_srv")
        server._tools = [
            Tool(name="zhida", description="知乎直答", inputSchema=self._SCHEMA)
        ]
        server.session = MagicMock()

        with patch("tools.registry.registry", ToolRegistry()):
            registered = _mcp_registration._register_server_tools("probe_srv", server, {})
        assert registered, "tool was not registered; write-through never fired"
        entry = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))["probe_srv"]
        return entry

    def test_cached_schema_keeps_properties(self, tmp_path, monkeypatch):
        cached = self._cache_write_through(tmp_path, monkeypatch)["tools"][0]["inputSchema"]
        assert set(cached.get("properties", {})) == {"query", "model"}, (
            "write-through persisted an empty schema — the SDK field rename "
            "was read with a bare camelCase getattr"
        )

    def test_cached_schema_keeps_required(self, tmp_path, monkeypatch):
        cached = self._cache_write_through(tmp_path, monkeypatch)["tools"][0]["inputSchema"]
        assert cached.get("required") == ["query", "model"]

    def test_cache_round_trip_reaches_agent_schema(self, tmp_path, monkeypatch):
        """The whole point of the cache: a lazy server re-advertises params."""
        from unittest.mock import patch

        import tools.mcp_tool as mt
        from tools import mcp_tool_registration as _mcp_registration
        from tools.registry import ToolRegistry

        entry = self._cache_write_through(tmp_path, monkeypatch)
        lazy_reg = ToolRegistry()
        with patch("tools.registry.registry", lazy_reg):
            names = _mcp_registration._register_from_cache_sync("probe_srv", {}, entry)
        assert names, "lazy registration produced no tools"
        schema = lazy_reg.get_schema("mcp__probe_srv__zhida")
        assert schema is not None, "lazy path did not register the tool"
        assert set(schema["parameters"].get("properties", {})) == {"query", "model"}
        assert schema["parameters"].get("required") == ["query", "model"]
