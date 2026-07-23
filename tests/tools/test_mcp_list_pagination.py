"""Tests for MCP list_* pagination (nextCursor draining).

The MCP spec allows servers to paginate ``tools/list``, ``resources/list``,
and ``prompts/list`` via an opaque ``nextCursor`` token. The Python SDK
fetches one page per call, so hermes must follow the cursor to see items
past page 1. Port of the invariant behind anomalyco/opencode#35439/#35500.
"""

import asyncio
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

    def test_follows_next_cursor_across_pages(self):
        """Pages are concatenated in order; cursor passed back verbatim."""
        pages = {
            None: SimpleNamespace(tools=[_tool("p1a"), _tool("p1b")], nextCursor="c2"),
            "c2": SimpleNamespace(tools=[_tool("p2a")], nextCursor="c3"),
            "c3": SimpleNamespace(tools=[_tool("p3a")], nextCursor=None),
        }

        async def fake_list(cursor=None):
            return pages[cursor]

        items = asyncio.run(_paginate_full_list(fake_list, "tools", "srv"))
        assert [t.name for t in items] == ["p1a", "p1b", "p2a", "p3a"]

    def test_empty_page_with_cursor_continues(self):
        """An empty middle page doesn't abort the walk."""
        pages = {
            None: SimpleNamespace(resources=[_tool("r1")], nextCursor="c2"),
            "c2": SimpleNamespace(resources=[], nextCursor="c3"),
            "c3": SimpleNamespace(resources=[_tool("r2")]),
        }

        async def fake_list(cursor=None):
            return pages[cursor]

        items = asyncio.run(_paginate_full_list(fake_list, "resources", "srv"))
        assert [t.name for t in items] == ["r1", "r2"]

    def test_missing_items_attr_tolerated(self):
        """A malformed result without the items attribute yields nothing."""
        list_method = AsyncMock(return_value=SimpleNamespace())
        items = asyncio.run(_paginate_full_list(list_method, "prompts", "srv"))
        assert items == []

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
