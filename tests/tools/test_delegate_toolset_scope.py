"""Tests for delegate_tool toolset scoping.

By default, subagents cannot gain tools that the parent does not have.  The
LLM controls the `toolsets` parameter, so requested toolsets are intersected
with the parent's enabled toolsets.  Operators may opt into a narrow
``delegation.allowed_child_toolsets`` allowlist so lean router parents can
spawn explicit subtools without exposing those schemas globally.
"""
from unittest.mock import patch
from types import SimpleNamespace

from tools.delegate_tool import _scope_requested_toolsets, _strip_blocked_tools


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        with patch("tools.delegate_tool._load_config", return_value={}):
            scoped = _scope_requested_toolsets(requested, parent_toolsets)

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped

    def test_all_requested_toolsets_available_on_parent(self):
        """LLM requests subset of parent tools — all pass through."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web", "browser"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "web"]
        with patch("tools.delegate_tool._load_config", return_value={}):
            scoped = _scope_requested_toolsets(requested, parent_toolsets)

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
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory", "messaging"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "messaging" not in child
        assert "terminal" in child

    def test_strip_blocked_expands_router_without_blocked_includes(self):
        """Router inheritance must not leak delegation/memory/messaging to leaves."""
        child = _strip_blocked_tools(["router"])

        assert "router" not in child
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "messaging" not in child
        assert "skills" in child
        assert "todo" in child
        assert "session_search" in child
        assert "cronjob" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        with patch("tools.delegate_tool._load_config", return_value={}):
            scoped = _scope_requested_toolsets(requested, parent_toolsets)

        assert scoped == []

    def test_allowed_child_toolsets_let_router_parent_grant_subtools(self):
        """Opt-in allowlist permits explicit child subtools absent from parent."""
        parent = SimpleNamespace(enabled_toolsets=["router"])
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]

        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allowed_child_toolsets": ["terminal", "file", "web"]},
        ):
            scoped = _scope_requested_toolsets(requested, parent_toolsets)

        assert sorted(scoped) == ["file", "terminal", "web"]
        assert "browser" not in scoped
        assert "rl" not in scoped
