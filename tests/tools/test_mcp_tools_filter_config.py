"""Regression tests for malformed ``mcp_servers.<name>.tools`` config shapes.

Bug class: ``tools`` written as a bare string (``tools: search_x,query_docs``) or a
name list reached every ``config.get("tools")`` consumer as a non-dict and crashed
``config_fingerprint`` with ``AttributeError: 'str' object has no attribute 'get'``,
silently registering 0 tools for the server. The contract now:

- dict       -> canonical shape, used as-is
- str        -> comma-separated ``include`` whitelist shorthand
- list/tuple -> ``include`` whitelist shorthand
- falsy      -> no filter (unchanged legacy behavior)
- anything else -> warned and degraded to no filter; never raises
"""

import logging

import tools.mcp_schema_cache as msc
from tools.mcp_tool_common import normalize_tools_filter
from tools.mcp_tool_registration import _make_tool_filter, _select_utility_schemas
from tools.mcp_tool_schema import _build_utility_schemas


BASE = {"command": "npx", "args": ["-y", "some-server"]}


class TestNormalizeToolsFilter:
    def test_dict_passes_through(self):
        cfg = {"include": ["a"], "exclude": ["b"]}
        assert normalize_tools_filter(cfg, "mcp_servers.s.tools") is cfg

    def test_falsy_is_no_filter(self):
        for value in (None, "", [], ()):
            assert normalize_tools_filter(value, "mcp_servers.s.tools") == {}

    def test_comma_string_is_include_shorthand(self):
        assert normalize_tools_filter("a,b", "mcp_servers.s.tools") == {"include": ["a", "b"]}
        assert normalize_tools_filter(" a , b ", "mcp_servers.s.tools") == {"include": ["a", "b"]}
        assert normalize_tools_filter("solo", "mcp_servers.s.tools") == {"include": ["solo"]}

    def test_list_shapes_are_include_shorthand(self):
        for value in (["a", "b"], ("a", "b"), {"a", "b"}):
            assert normalize_tools_filter(value, "mcp_servers.s.tools") == {"include": ["a", "b"]}

    def test_invalid_type_warns_and_degrades_to_no_filter(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            assert normalize_tools_filter(42, "mcp_servers.s.tools") == {}
        assert "mcp_servers.s.tools" in caplog.text


class TestConfigFingerprintToolsShapes:
    def test_string_tools_does_not_crash(self):
        msc.config_fingerprint({**BASE, "tools": "search_x,query_docs"})

    def test_shorthand_equals_canonical_include(self):
        assert msc.config_fingerprint({**BASE, "tools": "b,a"}) == msc.config_fingerprint(
            {**BASE, "tools": {"include": ["a", "b"]}}
        )
        assert msc.config_fingerprint({**BASE, "tools": ["b", "a"]}) == msc.config_fingerprint(
            {**BASE, "tools": {"include": ["a", "b"]}}
        )

    def test_shorthand_differs_from_unfiltered(self):
        assert msc.config_fingerprint({**BASE, "tools": "a"}) != msc.config_fingerprint(BASE)

    def test_invalid_tools_type_does_not_crash_or_change_fingerprint(self):
        assert msc.config_fingerprint({**BASE, "tools": 42}) == msc.config_fingerprint(BASE)

    def test_lazy_cache_round_trip_matches_shorthand_and_canonical(self, monkeypatch, tmp_path):
        # The lazy-registration path keys the cache entry on config_fingerprint: a config
        # rewritten from shorthand to canonical must keep hitting the same entry.
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        tools = [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]
        msc.write_cache_entry("srv", msc.config_fingerprint({**BASE, "tools": "a,b"}),
                              tools=tools, utility_tools=[])
        entry = msc.get_cached_entry("srv", msc.config_fingerprint({**BASE, "tools": {"include": ["b", "a"]}}))
        assert entry is not None
        assert msc.tools_from_cache_entry(entry) == tools

    def test_scalar_include_equals_one_item_list_not_multi(self):
        # A scalar entry is ONE tool name at runtime; it must fingerprint as its one-item
        # list and never collide with a multi-name list (lazy-cache key collision).
        scalar = {**BASE, "tools": {"include": "ab"}}
        one_item = {**BASE, "tools": {"include": ["ab"]}}
        two_items = {**BASE, "tools": {"include": ["a", "b"]}}
        assert msc.config_fingerprint(scalar) == msc.config_fingerprint(one_item)
        assert msc.config_fingerprint(scalar) != msc.config_fingerprint(two_items)

    def test_scalar_exclude_equals_one_item_list_not_multi(self):
        scalar = {**BASE, "tools": {"exclude": "ab"}}
        one_item = {**BASE, "tools": {"exclude": ["ab"]}}
        two_items = {**BASE, "tools": {"exclude": ["a", "b"]}}
        assert msc.config_fingerprint(scalar) == msc.config_fingerprint(one_item)
        assert msc.config_fingerprint(scalar) != msc.config_fingerprint(two_items)

    def test_lazy_cache_round_trip_scalar_entries(self, monkeypatch, tmp_path):
        # A cache entry written for a multi-name filter must not be served for the scalar
        # single-name filter (the reviewer's collision scenario), and the scalar's list
        # form is the same filter and must hit.
        monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")
        tools = [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]
        msc.write_cache_entry("srv", msc.config_fingerprint({**BASE, "tools": {"include": ["a", "b"]}}),
                              tools=tools, utility_tools=[])
        assert msc.get_cached_entry("srv", msc.config_fingerprint({**BASE, "tools": {"include": "ab"}})) is None
        msc.write_cache_entry("srv2", msc.config_fingerprint({**BASE, "tools": {"include": "ab"}}),
                              tools=tools, utility_tools=[])
        entry = msc.get_cached_entry("srv2", msc.config_fingerprint({**BASE, "tools": {"include": ["ab"]}}))
        assert entry is not None
        assert msc.tools_from_cache_entry(entry) == tools


class TestMakeToolFilterToolsShapes:
    def test_comma_string_is_whitelist(self):
        should = _make_tool_filter("s", {**BASE, "tools": "a,b"})
        assert should("a") and should("b") and not should("c")

    def test_list_is_whitelist(self):
        should = _make_tool_filter("s", {**BASE, "tools": ["a"]})
        assert should("a") and not should("b")

    def test_canonical_dict_unchanged(self):
        assert _make_tool_filter("s", {**BASE, "tools": {"exclude": ["a"]}})("b")
        assert not _make_tool_filter("s", {**BASE, "tools": {"exclude": ["a"]}})("a")
        assert not _make_tool_filter("s", {**BASE, "tools": {"include": []}})("a")

    def test_invalid_type_degrades_to_no_filter(self):
        assert _make_tool_filter("s", {**BASE, "tools": 42})("anything")


class TestUtilitySchemaSelectionToolsShapes:
    class _Server:
        initialize_result = None
        session = object()

    def test_comma_string_does_not_crash(self):
        selected = _select_utility_schemas("s", self._Server(), {**BASE, "tools": "resources"})
        assert isinstance(selected, list)

    def test_invalid_type_does_not_crash(self):
        selected = _select_utility_schemas("s", self._Server(), {**BASE, "tools": 42})
        assert isinstance(selected, list)

    def test_build_utility_schemas_still_shaped(self):
        assert all("handler_key" in e for e in _build_utility_schemas("s"))
