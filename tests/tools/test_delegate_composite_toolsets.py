"""Tests for composite toolset expansion in delegate_task intersection."""

import unittest

from tools.delegate_tool import _expand_parent_toolsets
from toolsets import TOOLSETS


class TestExpandParentToolsets(unittest.TestCase):
    """Verify _expand_parent_toolsets recognises individual toolsets within composites."""

    def test_composite_hermes_cli_expands_web(self):
        """hermes-cli includes web_search/web_extract → 'web' should be in expansion."""
        expanded = _expand_parent_toolsets({"hermes-cli"})
        self.assertIn("web", expanded)
        self.assertIn("terminal", expanded)
        # Expand only capabilities actually available in the current parent;
        # Workstation browser additions do not implicitly grow the CLI core.
        expected_browser = set(TOOLSETS["browser"]["tools"]).issubset(TOOLSETS["hermes-cli"]["tools"])
        self.assertEqual("browser" in expanded, expected_browser)
        # Original composite is preserved
        self.assertIn("hermes-cli", expanded)


    def test_intersection_with_expanded_composite(self):
        """End-to-end: requesting ['web'] from parent with ['hermes-cli'] yields ['web']."""
        parent_toolsets = {"hermes-cli"}
        expanded = _expand_parent_toolsets(parent_toolsets)
        toolsets = ["web"]
        child_toolsets = [t for t in toolsets if t in expanded]
        self.assertEqual(child_toolsets, ["web"])


if __name__ == "__main__":
    unittest.main()
