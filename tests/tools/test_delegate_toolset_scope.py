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

class TestContextEngineToolsFollowTheEngine:
    """A child runs the parent's context engine (same config), which rewrites the child's old tool
    results into recall stubs; the engine's tools must come along or the stubs are unreadable."""

    def _parent(self, enabled, engine_tools):
        return SimpleNamespace(enabled_toolsets=enabled, disabled_toolsets=[],
                               _context_engine_tool_names=set(engine_tools), valid_tool_names=set())

    def test_engine_tools_carried_to_child_with_or_without_explicit_toolsets(self):
        from tools.delegate_tool_toolsets import _resolve_child_toolsets
        parent = self._parent(["terminal", "file"], {"cmi_recall"})
        for requested in (None, ["terminal"]):
            enabled, _ = _resolve_child_toolsets(parent, requested, "leaf")
            assert "context_engine" in enabled

    def test_no_engine_tools_means_nothing_added(self):
        from tools.delegate_tool_toolsets import _resolve_child_toolsets
        enabled, _ = _resolve_child_toolsets(self._parent(["terminal"], ()), None, "leaf")
        assert "context_engine" not in enabled


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
