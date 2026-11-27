"""Tests for composite toolset expansion in delegate_task intersection."""

import unittest

from tools.delegate_tool import _expand_parent_toolsets, _strip_blocked_tools


class TestExpandParentToolsets(unittest.TestCase):
    """Verify _expand_parent_toolsets recognises individual toolsets within composites."""

    def test_composite_hermes_cli_expands_web(self):
        """hermes-cli includes web_search/web_extract → 'web' should be in expansion."""
        expanded = _expand_parent_toolsets({"hermes-cli"})
        self.assertIn("web", expanded)
        self.assertIn("terminal", expanded)
        self.assertIn("browser", expanded)
        # Original composite is preserved
        self.assertIn("hermes-cli", expanded)



    def test_included_toolsets_of_composite_parent_are_grantable(self):
        """A composite parent holds its ``includes`` too (#111700): ``debugging`` = terminal/process_manage +
        includes web/file, so a child may request ``web``/``file`` — but never a toolset the parent lacks."""
        expanded = _expand_parent_toolsets({"debugging"})
        self.assertTrue({"debugging", "terminal", "web", "file"} <= expanded)
        self.assertNotIn("browser", expanded)
        self.assertNotIn("hermes-cli", expanded)

    def test_composites_with_allowed_included_tools_are_not_stripped(self):
        toolsets = ["safe", "hermes-gateway", "hermes-cli", "delegation", "kanban"]

        self.assertEqual(
            _strip_blocked_tools(toolsets),
            ["safe", "hermes-gateway", "hermes-cli"],
        )


def test_retired_platform_bundle_parent_still_grants_its_core_toolsets():
    """A parent on the retired ``hermes-qqbot`` preset holds the core tools it carried, so a child asking for
    ``terminal``/``web`` gets them; the retired name stays out of every listing."""
    from types import SimpleNamespace

    from tools.delegate_tool_toolsets import _resolve_child_toolsets
    from toolsets import get_toolset_names

    parent = SimpleNamespace(enabled_toolsets=["hermes-qqbot"], disabled_toolsets=[])
    child, _disabled = _resolve_child_toolsets(parent, ["terminal", "web"], "leaf")
    assert child == ["terminal", "web"]
    assert "hermes-qqbot" not in get_toolset_names()


def test_mcp_alias_named_like_a_platform_preset_grants_no_builtin_toolset(monkeypatch):
    """An MCP server aliased ``hermes-web`` whose tools share built-in names is the parent's MCP toolset, not a
    left-core preset: a child asking for the built-in ``web`` does not get it through the alias."""
    from types import SimpleNamespace

    from tools import registry as registry_module
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    for tool in ("web_search", "web_extract"):
        registry.register(tool, "mcp-web", {"name": tool}, lambda: None)
    registry.register_toolset_alias("hermes-web", "mcp-web")
    parent = SimpleNamespace(enabled_toolsets=["hermes-web"], disabled_toolsets=[])
    child, _disabled = _resolve_child_toolsets(parent, ["web"], "leaf")
    assert "web" not in child


if __name__ == "__main__":
    unittest.main()
