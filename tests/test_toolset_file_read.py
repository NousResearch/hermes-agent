"""Tests for the file_read toolset — a read-only subset of the file toolset.

The file_read toolset provides read_file and search_files without write_file
or patch, enabling sandboxed agents to read files without modification access.

This follows the same pattern as the 'search' toolset (read-only subset of 'web').
"""

from toolsets import (
    TOOLSETS,
    get_toolset,
    resolve_toolset,
    validate_toolset,
    get_toolset_info,
)


class TestFileReadToolsetExists:
    """file_read is a valid toolset that can be resolved."""

    def test_file_read_is_valid(self):
        assert validate_toolset("file_read") is True

    def test_file_read_in_toolsets_dict(self):
        assert "file_read" in TOOLSETS

    def test_get_toolset_returns_entry(self):
        ts = get_toolset("file_read")
        assert ts is not None
        assert "description" in ts
        assert "tools" in ts
        assert "includes" in ts


class TestFileReadToolsetContent:
    """file_read resolves to exactly read_file + search_files — no write tools."""

    def test_resolves_to_read_file_and_search_files(self):
        tools = resolve_toolset("file_read")
        assert set(tools) == {"read_file", "search_files"}

    def test_does_not_include_write_file(self):
        tools = resolve_toolset("file_read")
        assert "write_file" not in tools

    def test_does_not_include_patch(self):
        tools = resolve_toolset("file_read")
        assert "patch" not in tools

    def test_is_strict_subset_of_file_toolset(self):
        """Every tool in file_read must also be in the file toolset."""
        file_read_tools = set(resolve_toolset("file_read"))
        file_tools = set(resolve_toolset("file"))
        assert file_read_tools.issubset(file_tools), (
            f"file_read has tools outside file: {file_read_tools - file_tools}"
        )

    def test_is_proper_subset_of_file_toolset(self):
        """file_read must be a proper (strict) subset — not equal to file."""
        file_read_tools = set(resolve_toolset("file_read"))
        file_tools = set(resolve_toolset("file"))
        assert file_read_tools != file_tools, (
            "file_read should be a proper subset of file, not equal"
        )


class TestFileReadToolsetInfo:
    """get_toolset_info returns correct metadata for file_read."""

    def test_info_returns_name(self):
        info = get_toolset_info("file_read")
        assert info is not None
        assert info["name"] == "file_read"

    def test_info_is_not_composite(self):
        info = get_toolset_info("file_read")
        assert info is not None
        assert info["is_composite"] is False

    def test_info_tool_count(self):
        info = get_toolset_info("file_read")
        assert info is not None
        assert info["tool_count"] == 2

    def test_info_direct_tools_match_resolve(self):
        info = get_toolset_info("file_read")
        assert info is not None
        assert set(info["direct_tools"]) == {"read_file", "search_files"}


class TestFileReadToolsetStructure:
    """file_read follows the same structural conventions as other subset toolsets."""

    def test_has_description(self):
        ts = TOOLSETS["file_read"]
        assert isinstance(ts["description"], str)
        assert len(ts["description"]) > 0

    def test_has_empty_includes(self):
        ts = TOOLSETS["file_read"]
        assert ts["includes"] == []

    def test_tools_is_list_of_strings(self):
        ts = TOOLSETS["file_read"]
        assert isinstance(ts["tools"], list)
        for tool in ts["tools"]:
            assert isinstance(tool, str)


class TestSearchPrecedent:
    """Verify the existing 'search' subset-of-web precedent still holds,
    confirming that the same pattern we use for file_read works."""

    def test_search_is_subset_of_web(self):
        search_tools = set(resolve_toolset("search"))
        web_tools = set(resolve_toolset("web"))
        assert search_tools.issubset(web_tools)

    def test_search_is_proper_subset_of_web(self):
        search_tools = set(resolve_toolset("search"))
        web_tools = set(resolve_toolset("web"))
        assert search_tools != web_tools