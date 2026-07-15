"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the ``toolsets`` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace

from tools.delegate_tool import _strip_blocked_tools


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
        """Empty requested inherits parent toolsets without delegation blocked."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web"])
        parent_enabled = list(parent.enabled_toolsets)
        result = _strip_blocked_tools(parent_enabled)

        assert "terminal" in result
        assert "delegate_task" not in result

    def test_strip_blocked_removes_delegation(self):
        """Blocked tools are always filtered out."""
        result = _strip_blocked_tools(["delegate_task", "terminal", "memory", "clarify"])
        assert "terminal" in result
        assert "delegate_task" not in result
        assert "memory" not in result

    def test_empty_intersection_yields_empty_toolsets(self):
        """No overlap between requested and parent — empty result."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []
