"""Tests for MCP list_* pagination (nextCursor draining).

The MCP spec allows servers to paginate ``tools/list``, ``resources/list``,
and ``prompts/list`` via an opaque ``nextCursor`` token. The Python SDK
fetches one page per call, so hermes must follow the cursor to see items
past page 1. Port of the invariant behind anomalyco/opencode#35439/#35500.
"""

import asyncio

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools.mcp_tool import _MCP_LIST_MAX_PAGES, _paginate_full_list


def _tool(name):
    t = MagicMock()
    t.name = name
    return t


class TestPaginateFullList:
    def test_single_page_no_cursor(self):
        """A result without nextCursor returns just that page."""
        list_method = AsyncMock(
            return_value=SimpleNamespace(tools=[_tool("a"), _tool("b")])
        )
        items = asyncio.run(_paginate_full_list(list_method, "tools", "srv"))
        assert [t.name for t in items] == ["a", "b"]
        list_method.assert_called_once_with()


    def test_runaway_cursor_capped(self):
        """A server that returns a cursor forever is bounded by the page cap."""
        calls = {"n": 0}

        async def evil_list(cursor=None):
            calls["n"] += 1
            return SimpleNamespace(
                tools=[_tool(f"t{calls['n']}")], nextCursor=f"c{calls['n']}"
            )

        items = asyncio.run(_paginate_full_list(evil_list, "tools", "srv"))
        assert calls["n"] == _MCP_LIST_MAX_PAGES
        assert len(items) == _MCP_LIST_MAX_PAGES


class TestDiscoveryUsesPagination:
    def test_discover_tools_drains_all_pages(self):
        """MCPServerTask._discover_tools registers tools from every page."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("pag_srv")
        server._config = {"command": "test"}
        pages = {
            None: SimpleNamespace(tools=[_tool("first")], nextCursor="page-2"),
            "page-2": SimpleNamespace(tools=[_tool("second")]),
        }

        async def fake_list(cursor=None):
            return pages[cursor]

        server.session = MagicMock()
        server.session.list_tools = fake_list
        # capability gate: _advertises_tools() returns True when no
        # capability info was captured (legacy fallback), so no override
        # is needed here.

        asyncio.run(server._discover_tools())
        assert [t.name for t in server._tools] == ["first", "second"]


class TestTypeErrorPropagation:
    """#104150: a TypeError raised INSIDE the list call (e.g. a server
    response decode failure) must propagate — it must not be caught by the
    mcp-2.0/1.x calling-convention fallback and replaced by a misleading
    'unexpected keyword argument cursor' error."""

    def test_type_error_from_modern_list_call_propagates(self):
        calls = {"n": 0}

        async def broken_list(*, params=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(tools=[_tool("first")], nextCursor="page-2")
            raise TypeError("server response decode failed")

        with pytest.raises(TypeError, match="server response decode failed"):
            asyncio.run(_paginate_full_list(broken_list, "tools", "srv"))

        assert calls["n"] == 2, "the legacy-cursor retry must not run"

    def test_legacy_signature_still_uses_cursor_fallback(self):
        """A genuinely 1.x-shaped method (no params kwarg) keeps working."""
        calls = {"n": 0}

        async def legacy_list(cursor=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(tools=[_tool("first")], nextCursor="p2")
            return SimpleNamespace(tools=[_tool("second")], nextCursor=None)

        items = asyncio.run(_paginate_full_list(legacy_list, "tools", "srv"))
        assert [t.name for t in items] == ["first", "second"]
        assert calls["n"] == 2


class TestCacheMetaExtraction:
    """SEP-2549 hints reach the schema cache only when the server sent them.

    mcp SDK 2.x result models synthesize ``ttl_ms=0`` / ``cache_scope="private"``
    on a hint-less ``tools/list``; recording that 0 expired every cache entry at
    the instant it was written, so ``lazy: true`` never engaged."""

    @staticmethod
    def _run(result):
        meta: dict = {}

        async def single_page():
            return result

        asyncio.run(_paginate_full_list(single_page, "tools", "srv", cache_meta_out=meta))
        return meta

    def test_sdk_result_without_wire_hint_records_nothing(self):
        from mcp import types

        result = types.ListToolsResult.model_validate({"tools": []})
        assert result.ttl_ms == 0  # the synthesized default this guards against
        assert self._run(result) == {}

    def test_sdk_result_with_explicit_zero_ttl_records_it(self):
        """A wire ``ttlMs: 0`` is a real "immediately stale" hint."""
        from mcp import types

        result = types.ListToolsResult.model_validate({"tools": [], "ttlMs": 0})
        assert self._run(result) == {"ttl_ms": 0}

    def test_sdk_result_with_wire_hints_records_them(self):
        from mcp import types

        result = types.ListToolsResult.model_validate(
            {"tools": [], "ttlMs": 5_000, "cacheScope": "public"})
        assert self._run(result) == {"ttl_ms": 5_000, "cache_scope": "public"}

    def test_untracked_result_zero_ttl_reads_as_absent(self):
        """Without field tracking a 0 cannot be told from the synthesized default."""
        result = SimpleNamespace(tools=[_tool("a")], ttl_ms=0, cache_scope="private")
        assert self._run(result) == {"cache_scope": "private"}

    def test_untracked_result_positive_ttl_is_recorded(self):
        result = SimpleNamespace(tools=[_tool("a")], ttl_ms=2_500)
        assert self._run(result) == {"ttl_ms": 2_500}
