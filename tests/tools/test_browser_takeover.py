"""Tests for the browser_takeover tool wiring and gating."""

import json
from unittest.mock import patch


class TestBrowserTakeoverSchema:
    def test_schema_in_browser_schemas(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        names = [s["name"] for s in BROWSER_TOOL_SCHEMAS]
        assert "browser_takeover" in names

    def test_schema_has_reason_and_ttl(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_takeover")
        props = schema["parameters"]["properties"]
        assert props["reason"]["type"] == "string"
        assert props["ttl_seconds"]["type"] == "integer"


class TestBrowserTakeoverToolsetWiring:
    def test_in_browser_toolset(self):
        from toolsets import TOOLSETS

        assert "browser_takeover" in TOOLSETS["browser"]["tools"]

    def test_in_hermes_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS

        assert "browser_takeover" in _HERMES_CORE_TOOLS

    def test_in_legacy_toolset_map(self):
        from model_tools import _LEGACY_TOOLSET_MAP

        assert "browser_takeover" in _LEGACY_TOOLSET_MAP["browser_tools"]

    def test_in_registry(self):
        from tools.registry import registry
        from tools import browser_tool  # noqa: F401

        assert "browser_takeover" in registry._tools


class TestBrowserTakeoverBehavior:
    @patch("tools.browser_tool._is_camofox_mode", return_value=True)
    @patch("tools.browser_tool.check_browser_takeover_requirements", return_value=True)
    @patch("tools.browser_tool.camofox_takeover")
    def test_routes_to_camofox_backend(self, mock_takeover, _mock_requirements, _mock_mode):
        from tools.browser_tool import browser_takeover

        mock_takeover.return_value = json.dumps({
            "success": True,
            "backend": "camofox",
            "url": "https://takeover.test/session",
        })

        result = json.loads(browser_takeover(reason="captcha", ttl_seconds=900, task_id="takeover-task"))
        assert result["success"] is True
        assert result["backend"] == "camofox"
        mock_takeover.assert_called_once_with(reason="captcha", ttl_seconds=900, task_id="takeover-task")

    @patch("tools.browser_tool._is_camofox_mode", return_value=True)
    @patch("tools.browser_tool.check_camofox_takeover_available", return_value=False)
    def test_requirement_check_verifies_helper_reachability(self, _mock_reachable, _mock_mode):
        from tools.browser_tool import check_browser_takeover_requirements

        assert check_browser_takeover_requirements() is False

    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_errors_when_backend_does_not_support_takeover(self, _mock_mode):
        from tools.browser_tool import browser_takeover

        result = json.loads(browser_takeover(task_id="no-backend"))
        assert result["success"] is False
        assert "currently supported only" in result["error"]

    @patch.dict("os.environ", {}, clear=True)
    def test_not_available_without_camofox_and_mint_url(self):
        from tools.browser_tool import check_browser_takeover_requirements

        assert check_browser_takeover_requirements() is False
