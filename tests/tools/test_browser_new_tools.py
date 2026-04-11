"""Tests for 6 new browser tools: hover, select, wait, forward, reload, scroll_to.

Verifies schemas, handlers, description fixes, and _EMPTY_OK_COMMANDS updates.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset browser_tool module-level caches between tests."""
    import tools.browser_tool as bt
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    yield
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False


# ---------------------------------------------------------------------------
# PART 1: All 6 new tools present in BROWSER_TOOL_SCHEMAS
# ---------------------------------------------------------------------------

NEW_TOOL_NAMES = [
    "browser_hover",
    "browser_select",
    "browser_wait",
    "browser_forward",
    "browser_reload",
    "browser_scroll_to",
]


class TestNewToolSchemas:
    """Verify all 6 new tool schemas are present and well-formed."""

    def _schema_names(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        return [s["name"] for s in BROWSER_TOOL_SCHEMAS]

    @pytest.mark.parametrize("tool_name", NEW_TOOL_NAMES)
    def test_schema_exists(self, tool_name):
        assert tool_name in self._schema_names()

    def test_total_schema_count(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        assert len(BROWSER_TOOL_SCHEMAS) == 16

    def test_hover_schema_has_ref(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_hover")
        assert "ref" in schema["parameters"]["properties"]
        assert "ref" in schema["parameters"]["required"]

    def test_select_schema_has_ref_and_value(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_select")
        props = schema["parameters"]["properties"]
        assert "ref" in props
        assert "value" in props
        assert set(schema["parameters"]["required"]) == {"ref", "value"}

    def test_wait_schema_has_selector_or_ms(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_wait")
        assert "selector_or_ms" in schema["parameters"]["properties"]
        assert "selector_or_ms" in schema["parameters"]["required"]

    def test_forward_schema_no_params(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_forward")
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []

    def test_reload_schema_no_params(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_reload")
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []

    def test_scroll_to_schema_has_ref(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_scroll_to")
        assert "ref" in schema["parameters"]["properties"]
        assert "ref" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# PART 1 (continued): All 6 handlers are callable
# ---------------------------------------------------------------------------

class TestNewToolHandlersCallable:
    """Verify all 6 handler functions exist and are callable."""

    @pytest.mark.parametrize("handler_name", NEW_TOOL_NAMES)
    def test_handler_is_callable(self, handler_name):
        import tools.browser_tool as bt
        handler = getattr(bt, handler_name)
        assert callable(handler)


# ---------------------------------------------------------------------------
# PART 1: forward and reload in _EMPTY_OK_COMMANDS
# ---------------------------------------------------------------------------

class TestEmptyOkCommands:

    def test_forward_in_empty_ok(self):
        from tools.browser_tool import _EMPTY_OK_COMMANDS
        assert "forward" in _EMPTY_OK_COMMANDS

    def test_reload_in_empty_ok(self):
        from tools.browser_tool import _EMPTY_OK_COMMANDS
        assert "reload" in _EMPTY_OK_COMMANDS

    def test_original_commands_still_present(self):
        from tools.browser_tool import _EMPTY_OK_COMMANDS
        assert "close" in _EMPTY_OK_COMMANDS
        assert "record" in _EMPTY_OK_COMMANDS


# ---------------------------------------------------------------------------
# PART 2: Schema description fixes
# ---------------------------------------------------------------------------

class TestSchemaDescriptionFixes:

    def _get_description(self, name):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == name)
        return schema["description"]

    def test_browser_click_no_browser_snapshot(self):
        desc = self._get_description("browser_click")
        assert "browser_snapshot" not in desc
        assert "browser_navigate" in desc

    def test_browser_type_no_browser_snapshot(self):
        desc = self._get_description("browser_type")
        assert "browser_snapshot" not in desc
        assert "browser_navigate" in desc

    def test_browser_scroll_500px(self):
        desc = self._get_description("browser_scroll")
        assert "500px" in desc


# ---------------------------------------------------------------------------
# PART 1: Camofox mode returns unsupported for new tools
# ---------------------------------------------------------------------------

class TestCamofoxUnsupported:

    @pytest.mark.parametrize("handler_name,kwargs", [
        ("browser_hover", {"ref": "@e1"}),
        ("browser_select", {"ref": "@e1", "value": "opt1"}),
        ("browser_wait", {"selector_or_ms": "1000"}),
        ("browser_forward", {}),
        ("browser_reload", {}),
        ("browser_scroll_to", {"ref": "@e1"}),
    ])
    @patch("tools.browser_tool._is_camofox_mode", return_value=True)
    def test_camofox_returns_unsupported(self, _mock_camofox, handler_name, kwargs):
        import tools.browser_tool as bt
        handler = getattr(bt, handler_name)
        result = json.loads(handler(**kwargs, task_id="test"))
        assert result["success"] is False
        assert "not supported in Camofox mode" in result["error"]


# ---------------------------------------------------------------------------
# PART 1: Ref normalization for ref-based tools
# ---------------------------------------------------------------------------

class TestRefNormalization:

    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    @patch("tools.browser_tool._run_browser_command")
    def test_hover_adds_at_prefix(self, mock_cmd, _mock_camo):
        mock_cmd.return_value = {"success": True}
        import tools.browser_tool as bt
        bt.browser_hover(ref="e5", task_id="test")
        args = mock_cmd.call_args[0]
        assert args[1] == "hover"
        assert args[2] == ["@e5"]

    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    @patch("tools.browser_tool._run_browser_command")
    def test_select_adds_at_prefix(self, mock_cmd, _mock_camo):
        mock_cmd.return_value = {"success": True}
        import tools.browser_tool as bt
        bt.browser_select(ref="e3", value="option_a", task_id="test")
        args = mock_cmd.call_args[0]
        assert args[1] == "select"
        assert args[2] == ["@e3", "option_a"]

    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    @patch("tools.browser_tool._run_browser_command")
    def test_scroll_to_adds_at_prefix(self, mock_cmd, _mock_camo):
        mock_cmd.return_value = {"success": True}
        import tools.browser_tool as bt
        bt.browser_scroll_to(ref="e10", task_id="test")
        args = mock_cmd.call_args[0]
        assert args[1] == "scrollintoview"
        assert args[2] == ["@e10"]

    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    @patch("tools.browser_tool._run_browser_command")
    def test_hover_preserves_at_prefix(self, mock_cmd, _mock_camo):
        mock_cmd.return_value = {"success": True}
        import tools.browser_tool as bt
        bt.browser_hover(ref="@e5", task_id="test")
        args = mock_cmd.call_args[0]
        assert args[2] == ["@e5"]
