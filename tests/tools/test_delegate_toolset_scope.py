"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace

from tools.delegate_tool import _strip_blocked_tools, _emit_parent_console

class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_mcp_server_named_like_a_blocked_toolset_reaches_the_child(self):
        """An MCP server configured as ``memory`` is the user's server, not Hermes' MEMORY.md tool: whether the
        child inherits or narrows its toolsets, it keeps the server's tools exactly when the parent has them
        (not when the parent disabled ``memory``), never gains a tool the parent lacks, and never gets the built-in."""
        import model_tools
        from tools.delegate_tool_toolsets import _resolve_child_toolsets
        from tools.registry import registry

        registry.register(
            name="mcp__memory__search_nodes", toolset="mcp-memory", handler=lambda args, **kw: "{}",
            schema={"name": "mcp__memory__search_nodes", "description": "kg",
                    "parameters": {"type": "object", "properties": {}}})
        registry.register_toolset_alias("memory", "mcp-memory")
        try:
            for parent_disabled in ([], ["memory"]):
                # Config lists an enabled MCP server by its bare name (tools_config._merge_mcp_servers).
                parent = SimpleNamespace(enabled_toolsets=["hermes-cli", "memory"], disabled_toolsets=parent_disabled)
                parent_tools = model_tools._select_tool_names(parent.enabled_toolsets, parent_disabled, quiet_mode=True)
                assert ("mcp__memory__search_nodes" in parent_tools) == (not parent_disabled)
                for requested in (None, ["web"]):
                    enabled, disabled = _resolve_child_toolsets(parent, requested, "leaf")
                    child_tools = model_tools._select_tool_names(enabled, disabled, quiet_mode=True)
                    context = (parent_disabled, requested, enabled, disabled)
                    assert child_tools <= parent_tools, (sorted(child_tools - parent_tools), context)
                    assert ("mcp__memory__search_nodes" in child_tools) == (not parent_disabled), context
                    assert "memory" not in child_tools, context
        finally:
            registry.deregister("mcp__memory__search_nodes")

class TestEmitParentConsole:
    """Progress lines (e.g. ``✓ [N/M] …``) must route through the parent's
    configured ``_safe_print`` in headless stdio hosts (ACP, gateway) so
    they don't land on stdout and corrupt JSON-RPC frames. Regression for a
    bug where delegate_task completion lines pushed to stdout caused
    ``Failed to parse JSON message: ✓ [3/3] …`` errors in the ACP adapter."""

    def test_routes_through_parent_safe_print_when_available(self, capsys):
        captured_lines = []
        parent = SimpleNamespace(_safe_print=lambda line: captured_lines.append(line))

        _emit_parent_console(parent, "  ✓ [1/3] Research done  (11.55s)")

        assert captured_lines == ["  ✓ [1/3] Research done  (11.55s)"]
        stdout_stderr = capsys.readouterr()
        assert stdout_stderr.out == ""
        assert stdout_stderr.err == ""
