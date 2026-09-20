"""Contract tests for the file / file_write toolset split.

Behaviour contracts, not counts: a read-only role must be able to drop file
mutation without losing file reading, and the default posture must be unchanged
for everyone who never asks for the split.
"""

import pytest

from model_tools import get_tool_definitions

READ_TOOLS = {"read_file", "search_files"}
WRITE_TOOLS = {"write_file", "patch"}


def _names(disabled_toolsets=None):
    defs = get_tool_definitions(None, disabled_toolsets or [], True)
    return {d["function"]["name"] for d in defs}


def test_default_posture_grants_read_and_write():
    """No disables = every file tool, exactly as before the split."""
    names = _names()
    assert READ_TOOLS | WRITE_TOOLS <= names


def test_disabling_file_write_keeps_reading():
    """The point of the split: a review role loses mutation, keeps inspection."""
    names = _names(["file_write"])
    assert READ_TOOLS <= names
    assert not (WRITE_TOOLS & names)


def test_disabling_file_keeps_writing():
    """The tiers are independent in both directions."""
    names = _names(["file"])
    assert WRITE_TOOLS <= names
    assert not (READ_TOOLS & names)


def test_disabling_both_removes_all_file_tools():
    names = _names(["file", "file_write"])
    assert not ((READ_TOOLS | WRITE_TOOLS) & names)


@pytest.mark.parametrize("toolset,expected", [
    ("file", {"read_file", "search_files"}),
    ("file_write", {"write_file", "patch"}),
])
def test_registry_assigns_each_file_tool_to_its_tier(toolset, expected):
    """The registry entry is what the resolver actually reads; TOOLSETS alone
    is not enough, so pin the per-tool assignment too."""
    from tools.registry import registry  # noqa: PLC0415

    assert set(registry.get_tool_names_for_toolset(toolset)) == expected


def test_debugging_composite_can_still_patch_code():
    """`debugging` includes file_write; a debugger that cannot edit is useless."""
    from toolsets import TOOLSETS  # noqa: PLC0415

    includes = TOOLSETS["debugging"].get("includes") or []
    assert "file" in includes
    assert "file_write" in includes
