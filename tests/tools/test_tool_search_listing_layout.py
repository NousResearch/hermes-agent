"""Layout of the deferred-tool catalog embedded in the tool_search description."""

from __future__ import annotations

from typing import Any


def _td(name: str, description: str) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {}}}}


def _register(toolset: str, names_and_descs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    from tools.registry import registry

    defs = []
    for name, desc in names_and_descs:
        registry.register(name=name, handler=lambda args, **kwargs: "{}", schema=_td(name, desc),
                          toolset=toolset)
        defs.append(_td(name, desc))
    return defs


def test_collapsing_one_huge_server_leaves_the_others_their_descriptions():
    """An 800-tool server collapses to one summary line; a small server that fits in full
    keeps its one-line descriptions instead of dropping to bare names."""
    from tools.tool_search_catalog import build_catalog_listing_with_form

    defs = _register("mcp-layouthuge", [(f"layouthuge_op_{i:04d}", "Do a huge-server thing.")
                                        for i in range(800)])
    defs += _register("mcp-layoutsmall", [("layoutsmall_render", "Render the current scene to a file."),
                                          ("layoutsmall_inspect", "Summarize the objects in the scene.")])

    text, form = build_catalog_listing_with_form(defs, max_tokens=1000)

    assert text is not None and form == "mixed"
    assert "layouthuge (800 tools — names not listed" in text
    assert "- layoutsmall_render: Render the current scene to a file." in text.splitlines()
    assert "layouthuge_op_0000" not in text


def test_single_tool_groups_render_without_a_count_heading():
    """A one-tool group is its tool line alone, placed before the multi-tool groups so it cannot
    read as a member of the group above it."""
    from tools.tool_search_catalog import build_catalog_listing_with_form

    defs = _register("mcp-layoutpair", [("layoutpair_b", "Second paired tool."),
                                        ("layoutpair_a", "First paired tool.")])
    defs += _register("mcp-layoutsolo", [("layoutsolo_only", "The only solo tool.")])

    text, form = build_catalog_listing_with_form(defs, max_tokens=4000)

    assert text is not None and form == "full"
    assert text.splitlines()[1:] == [
        "- layoutsolo_only: The only solo tool.",
        "layoutpair tools (2):",
        "- layoutpair_a: First paired tool.",
        "- layoutpair_b: Second paired tool.",
    ]
