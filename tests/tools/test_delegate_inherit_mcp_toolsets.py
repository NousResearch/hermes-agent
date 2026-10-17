"""Regression for #138695: ``delegation.inherit_mcp_toolsets: false`` must reach the full-inherit
child path, not only narrowed ``toolsets=[...]`` requests.

Every delegate_task child is built with ``toolsets=None``, so the resolver's full-inherit branches
are the ones that actually run in production; before the fix the flag was consulted only in the
narrowed branch, making the opt-out a silent no-op exactly where it matters — children carried
every parent MCP server's tool schema and paid its prefill on each spawn regardless.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent


class TestInheritMcpToolsetsOptOut(unittest.TestCase):
    @patch(
        "tools.delegate_tool._load_config",
        return_value={"inherit_mcp_toolsets": False},
    )
    def test_opted_out_drops_parent_mcp_on_full_inherit(self, mock_cfg):
        """MCP names leave enabled_toolsets and land in disabled_toolsets for a ``toolsets=None``
        spawn, so alias/composite expansion cannot re-add them after the fact."""
        parent = _make_mock_parent()
        parent.enabled_toolsets = ["web", "browser", "mcp-MiniMax", "mcp-work-tools"]

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="Full inherit with the MCP opt-out",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )

        kwargs = MockAgent.call_args[1]
        self.assertIn("web", kwargs["enabled_toolsets"])
        self.assertNotIn("mcp-MiniMax", kwargs["enabled_toolsets"])
        self.assertNotIn("mcp-work-tools", kwargs["enabled_toolsets"])
        self.assertIn("mcp-MiniMax", kwargs["disabled_toolsets"])
        self.assertIn("mcp-work-tools", kwargs["disabled_toolsets"])

    def test_full_inherit_keeps_parent_mcp_by_default(self):
        """Control for the opt-out test: the default (inherit) keeps the parent's MCP toolsets,
        so the pair cannot be satisfied by both paths stripping or both keeping."""
        parent = _make_mock_parent()
        parent.enabled_toolsets = ["web", "mcp-MiniMax"]

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="Full inherit with the default",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )

        kwargs = MockAgent.call_args[1]
        self.assertIn("mcp-MiniMax", kwargs["enabled_toolsets"])
        self.assertNotIn("mcp-MiniMax", kwargs["disabled_toolsets"])


if __name__ == "__main__":
    unittest.main()
