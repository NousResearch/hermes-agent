"""End-to-end tests for the file_read toolset.

Tests the full resolution path: enabled_toolsets → resolve_toolset → registry.get_definitions
→ actual OpenAI-format schemas sent to the LLM.

Since test environments may lack dependencies for some tool imports (requests,
openai, etc.), we register mock tool entries in a fresh ToolRegistry and call
get_definitions directly, bypassing module-level globals.

For the full get_tool_definitions() integration, see test_toolset_file_read.py
which tests the toolset resolution layer in isolation.
"""

import pytest
from tools.registry import ToolRegistry
from toolsets import resolve_toolset


def _make_schema(name: str, description: str = "test tool"):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


def _dummy_handler(args, **kwargs):
    return "{}"


def _register_file_tools(registry: ToolRegistry):
    """Register the four file tools in the given registry."""
    for name, desc in [
        ("read_file", "Read a file"),
        ("write_file", "Write a file"),
        ("patch", "Patch a file"),
        ("search_files", "Search files"),
    ]:
        registry.register(
            name=name,
            toolset="file",
            schema=_make_schema(name, desc),
            handler=_dummy_handler,
        )


class TestFileReadE2E:
    """End-to-end: file_read toolset resolves and produces correct tool schemas."""

    def test_resolve_then_get_definitions_yields_two_tools(self):
        """resolve_toolset('file_read') → registry.get_definitions → exactly 2 schemas."""
        reg = ToolRegistry()
        _register_file_tools(reg)

        tool_names = resolve_toolset("file_read")
        assert set(tool_names) == {"read_file", "search_files"}

        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}
        assert names == {"read_file", "search_files"}
        assert len(defs) == 2

    def test_file_read_schemas_have_correct_structure(self):
        """Each tool definition has valid OpenAI function-calling format."""
        reg = ToolRegistry()
        _register_file_tools(reg)

        tool_names = resolve_toolset("file_read")
        defs = reg.get_definitions(set(tool_names), quiet=True)

        for tool in defs:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    def test_write_file_not_in_file_read_definitions(self):
        """write_file and patch are never in file_read definitions."""
        reg = ToolRegistry()
        _register_file_tools(reg)

        tool_names = resolve_toolset("file_read")
        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert "write_file" not in names
        assert "patch" not in names

    def test_file_yields_all_four_tools(self):
        """The full 'file' toolset yields all four tools."""
        reg = ToolRegistry()
        _register_file_tools(reg)

        tool_names = resolve_toolset("file")
        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert names == {"read_file", "write_file", "patch", "search_files"}

    def test_file_read_is_proper_subset_of_file(self):
        """file_read tools are a proper subset of file tools."""
        reg = ToolRegistry()
        _register_file_tools(reg)

        file_read_names = set(resolve_toolset("file_read"))
        file_names = set(resolve_toolset("file"))

        assert file_read_names.issubset(file_names)
        assert file_read_names != file_names  # proper, not equal

        # Also verify through get_definitions
        file_read_defs = reg.get_definitions(file_read_names, quiet=True)
        file_defs = reg.get_definitions(file_names, quiet=True)

        assert {d["function"]["name"] for d in file_read_defs}.issubset(
            {d["function"]["name"] for d in file_defs}
        )

    def test_file_read_plus_terminal_excludes_writes(self):
        """file_read + terminal still excludes write_file and patch."""
        reg = ToolRegistry()
        _register_file_tools(reg)
        reg.register(
            name="terminal",
            toolset="terminal",
            schema=_make_schema("terminal", "Run shell commands"),
            handler=_dummy_handler,
        )
        reg.register(
            name="process",
            toolset="terminal",
            schema=_make_schema("process", "Manage processes"),
            handler=_dummy_handler,
        )

        combined = set(resolve_toolset("file_read")) | set(resolve_toolset("terminal"))
        defs = reg.get_definitions(combined, quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert "read_file" in names
        assert "search_files" in names
        assert "terminal" in names
        assert "write_file" not in names
        assert "patch" not in names


class TestCheckFnWithFileRead:
    """check_fn (availability gating) works correctly with file_read."""

    def test_check_fn_filters_unavailable_tools(self):
        """If read_file has a check_fn that returns False, it's excluded."""
        reg = ToolRegistry()
        reg.register(
            name="read_file",
            toolset="file",
            schema=_make_schema("read_file", "Read a file"),
            handler=_dummy_handler,
            check_fn=lambda: False,  # unavailable
        )
        reg.register(
            name="search_files",
            toolset="file",
            schema=_make_schema("search_files", "Search files"),
            handler=_dummy_handler,
        )

        tool_names = resolve_toolset("file_read")
        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert "read_file" not in names
        assert "search_files" in names

    def test_check_fn_allows_available_tools(self):
        """If check_fn returns True, tools are included normally."""
        reg = ToolRegistry()
        reg.register(
            name="read_file",
            toolset="file",
            schema=_make_schema("read_file", "Read a file"),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="search_files",
            toolset="file",
            schema=_make_schema("search_files", "Search files"),
            handler=_dummy_handler,
        )

        tool_names = resolve_toolset("file_read")
        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert names == {"read_file", "search_files"}


class TestFileReadToolsetIntegration:
    """Integration test: get_tool_definitions produces only read_file + search_files
    when file_read is the enabled toolset — no write_file or patch leak through."""

    def test_file_read_definitions_only_read_tools(self, monkeypatch):
        """get_tool_definitions(enabled_toolsets=['file_read']) yields only read_file and search_files."""
        from model_tools import get_tool_definitions

        tools = get_tool_definitions(enabled_toolsets=["file_read"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "read_file" in names, f"read_file missing from file_read definitions: {names}"
        assert "search_files" in names, f"search_files missing from file_read definitions: {names}"
        assert "write_file" not in names, f"write_file leaked into file_read definitions: {names}"
        assert "patch" not in names, f"patch leaked into file_read definitions: {names}"

    def test_file_read_definitions_count(self, monkeypatch):
        """file_read should produce exactly 2 tool definitions (read_file + search_files)."""
        from model_tools import get_tool_definitions

        tools = get_tool_definitions(enabled_toolsets=["file_read"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert names == {"read_file", "search_files"}

    def test_file_definitions_include_all_four(self, monkeypatch):
        """Full 'file' toolset still produces all four tools (regression guard)."""
        from model_tools import get_tool_definitions

        tools = get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert {"read_file", "write_file", "patch", "search_files"}.issubset(names)


class TestSearchPrecedentE2E:
    """Verify the search (subset of web) precedent works the same way."""

    def test_search_yields_only_web_search(self):
        """search toolset resolves to exactly web_search."""
        reg = ToolRegistry()
        reg.register(
            name="web_search",
            toolset="web",
            schema=_make_schema("web_search", "Search the web"),
            handler=_dummy_handler,
        )
        reg.register(
            name="web_extract",
            toolset="web",
            schema=_make_schema("web_extract", "Extract web content"),
            handler=_dummy_handler,
        )

        tool_names = resolve_toolset("search")
        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert names == {"web_search"}

    def test_web_yields_both(self):
        """web toolset resolves to web_search + web_extract."""
        reg = ToolRegistry()
        reg.register(
            name="web_search",
            toolset="web",
            schema=_make_schema("web_search", "Search the web"),
            handler=_dummy_handler,
        )
        reg.register(
            name="web_extract",
            toolset="web",
            schema=_make_schema("web_extract", "Extract web content"),
            handler=_dummy_handler,
        )

        tool_names = resolve_toolset("web")
        defs = reg.get_definitions(set(tool_names), quiet=True)
        names = {d["function"]["name"] for d in defs}

        assert names == {"web_search", "web_extract"}