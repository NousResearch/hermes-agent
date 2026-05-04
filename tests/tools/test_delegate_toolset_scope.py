"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from tools.delegate_tool import (
    _resolve_parent_tool_universe,
    _strip_blocked_tools,
    _toolset_subset_of_parent,
)


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        # Simulate the intersection logic from _build_child_agent
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped

    def test_all_requested_toolsets_available_on_parent(self):
        """LLM requests subset of parent tools — all pass through."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web", "browser"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "web"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["terminal", "web"]

    def test_no_toolsets_requested_inherits_parent(self):
        """When toolsets is None/empty, child inherits parent's set."""
        parent_toolsets = ["terminal", "file", "web"]
        child = _strip_blocked_tools(parent_toolsets)
        assert "terminal" in child
        assert "file" in child
        assert "web" in child

    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []


class TestCompositeToolsetIntersection:
    """Composite presets (hermes-cli, hermes-acp, …) must be expanded before
    intersecting with the child's requested toolsets — otherwise the child
    gets zero tools because the toolset names don't compare equal even though
    the underlying tools are present (issue #19447)."""

    def test_resolve_parent_tool_universe_expands_hermes_cli(self):
        tools = _resolve_parent_tool_universe({"hermes-cli"})
        assert "web_search" in tools
        assert "web_extract" in tools
        assert "terminal" in tools
        assert "read_file" in tools

    def test_web_subset_of_hermes_cli(self):
        parent_tools = _resolve_parent_tool_universe({"hermes-cli"})
        assert _toolset_subset_of_parent("web", parent_tools) is True

    def test_terminal_subset_of_hermes_cli(self):
        parent_tools = _resolve_parent_tool_universe({"hermes-cli"})
        assert _toolset_subset_of_parent("terminal", parent_tools) is True

    def test_browser_not_subset_of_terminal_only_parent(self):
        parent_tools = _resolve_parent_tool_universe({"terminal"})
        assert _toolset_subset_of_parent("browser", parent_tools) is False

    def test_web_subset_of_hermes_acp(self):
        parent_tools = _resolve_parent_tool_universe({"hermes-acp"})
        assert _toolset_subset_of_parent("web", parent_tools) is True

    def test_unknown_toolset_is_not_subset(self):
        """Unknown/unresolvable toolset names cannot smuggle access in."""
        parent_tools = _resolve_parent_tool_universe({"hermes-cli"})
        assert _toolset_subset_of_parent("does-not-exist", parent_tools) is False

    def test_intersection_with_composite_parent_keeps_requested(self):
        """End-to-end shape of the new intersection: parent runs hermes-cli,
        child requests ['web', 'file'] — both should survive the filter."""
        parent_toolsets = {"hermes-cli"}
        parent_tools = _resolve_parent_tool_universe(parent_toolsets)
        requested = ["web", "file", "browser"]
        scoped = [
            t for t in requested
            if t in parent_toolsets
            or _toolset_subset_of_parent(t, parent_tools)
        ]
        assert "web" in scoped
        assert "file" in scoped
        assert "browser" in scoped

    def test_intersection_with_composite_parent_drops_uncovered(self):
        """A toolset whose tools aren't in the parent's universe is dropped."""
        parent_toolsets = {"web"}
        parent_tools = _resolve_parent_tool_universe(parent_toolsets)
        requested = ["web", "terminal"]
        scoped = [
            t for t in requested
            if t in parent_toolsets
            or _toolset_subset_of_parent(t, parent_tools)
        ]
        assert "web" in scoped
        assert "terminal" not in scoped
