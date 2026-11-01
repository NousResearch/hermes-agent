"""delegation.inherit_soul: opt-in SOUL.md identity for subagents."""

import unittest
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent


class TestDelegationInheritSoul(unittest.TestCase):
    """delegation.inherit_soul decides whether children get SOUL.md as their identity."""

    def _child_kwargs(self, delegation_cfg):
        parent = _make_mock_parent()
        parent.enabled_toolsets = ["file"]
        parent.disabled_toolsets = []
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._load_config", return_value=delegation_cfg),
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None,
                               max_iterations=10, parent_agent=parent, task_count=1, role="leaf")
        return MockAgent.call_args[1]

    def test_children_skip_soul_by_default(self):
        kwargs = self._child_kwargs({})
        self.assertTrue(kwargs["skip_context_files"])
        # Off by default: the kwarg is omitted, so AIAgent's own default (False) applies.
        self.assertFalse(kwargs.get("load_soul_identity", False))

    def test_inherit_soul_gives_children_the_soul_identity(self):
        """delegation.inherit_soul: SOUL.md fills the child's identity slot; cwd context files stay skipped."""
        kwargs = self._child_kwargs({"inherit_soul": True})
        self.assertTrue(kwargs["skip_context_files"])
        self.assertTrue(kwargs["load_soul_identity"])


if __name__ == "__main__":
    unittest.main()
