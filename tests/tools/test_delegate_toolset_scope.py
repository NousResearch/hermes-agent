"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from tools.delegate_tool import _build_child_agent, _strip_blocked_tools


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


def _make_mcp_restricted_parent():
    """Parent agent whose MCP context is empty (e.g. no_mcp orchestrator).

    enabled_toolsets=[] means the parent loaded no toolsets at all — the
    intersection against it would normally drop every requested toolset.
    """
    parent = MagicMock()
    parent.enabled_toolsets = []
    parent._delegate_depth = 0
    parent._credential_pool = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._print_fn = None
    return parent


class TestProfileMcpToolsetBypass:
    """MCP toolsets declared by a named agent_profile bypass parent intersection.

    Regression coverage for NousResearch/hermes-agent#32668: an orchestrator
    that restricts its own MCP servers must still be able to hand domain MCP
    toolsets to a child via a named profile. Non-MCP toolsets keep going
    through the parent intersection (the security boundary).
    """

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_profile_mcp_toolsets_bypass_parent_intersection(self, _):
        """profile_name set → MCP toolsets pass through even when parent has none."""
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Check mail",
                context=None,
                toolsets=["mcp-fastmail", "mcp-knowledge"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name="mail",
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        assert "mcp-fastmail" in child_toolsets
        assert "mcp-knowledge" in child_toolsets

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_no_profile_mcp_toolsets_still_intersected(self, _):
        """No profile → MCP toolset request is dropped (intersection enforced)."""
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Check mail",
                context=None,
                toolsets=["mcp-fastmail"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name=None,
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        assert "mcp-fastmail" not in child_toolsets

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_profile_non_mcp_toolsets_still_intersected(self, _):
        """Even with a profile, non-MCP toolsets the parent lacks are dropped."""
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Browse the web",
                context=None,
                toolsets=["mcp-fastmail", "browser"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name="mail",
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        # MCP toolset bypasses intersection ...
        assert "mcp-fastmail" in child_toolsets
        # ... but the non-MCP toolset the parent never had is still dropped.
        assert "browser" not in child_toolsets
