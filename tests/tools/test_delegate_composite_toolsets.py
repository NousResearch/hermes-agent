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


def test_left_core_platform_bundle_parent_still_grants_its_toolsets(monkeypatch, tmp_path):
    """A parent on the left-core ``hermes-feishu`` preset holds the core + plugin tools it resolves to, so a child
    asking for ``terminal``/``web``/``feishu_doc`` gets them; never a toolset the bundle lacks."""
    from types import SimpleNamespace

    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools import registry as registry_module
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    home = tmp_path / "feishu-profile"
    for tool, toolset in (("feishu_doc_read", "feishu_doc"), ("feishu_drive_add_comment", "feishu_drive")):
        registry.register(tool, toolset, {"name": tool}, lambda: None, scope=hermes_home_key(home))
    token = set_hermes_home_override(home)
    try:
        parent = SimpleNamespace(enabled_toolsets=["hermes-feishu"], disabled_toolsets=[])
        child, _disabled = _resolve_child_toolsets(parent, ["terminal", "web", "feishu_doc", "discord_admin"], "leaf")
    finally:
        reset_hermes_home_override(token)
    assert child == ["terminal", "web", "feishu_doc"]


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
