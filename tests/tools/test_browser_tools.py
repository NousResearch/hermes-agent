"""Tests for browser tools: navigate, snapshot, click, type, press, scroll, back, get_images.

Coverage for tools that had no tests. Follows the same mock-based pattern as
test_browser_console.py and test_browser_cleanup.py.
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_OK = {"success": True, "data": {}}
_FAIL = {"success": False, "error": "browser error"}


# ── browser_navigate ──────────────────────────────────────────────────


class TestBrowserNavigate:
    def _nav(self, url="https://example.com", task_id="test"):
        from tools.browser_tool import browser_navigate
        return browser_navigate(url, task_id=task_id)

    def test_success_returns_url_and_title(self):
        result_data = {"success": True, "data": {"url": "https://example.com", "title": "Example"}}
        with (
            patch("tools.browser_tool.check_website_access", return_value=None),
            patch("tools.browser_tool._get_session_info", return_value={"_first_nav": False}),
            patch("tools.browser_tool._run_browser_command", return_value=result_data),
        ):
            result = json.loads(self._nav())

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"

    def test_failure_returns_error(self):
        with (
            patch("tools.browser_tool.check_website_access", return_value=None),
            patch("tools.browser_tool._get_session_info", return_value={"_first_nav": False}),
            patch("tools.browser_tool._run_browser_command", return_value=_FAIL),
        ):
            result = json.loads(self._nav())

        assert result["success"] is False
        assert "error" in result

    def test_blocked_by_policy(self):
        blocked = {"message": "Blocked by policy", "host": "example.com", "rule": "deny", "source": "config"}
        with patch("tools.browser_tool.check_website_access", return_value=blocked):
            result = json.loads(self._nav())

        assert result["success"] is False
        assert result["blocked_by_policy"]["host"] == "example.com"

    def test_bot_detection_warning_on_suspicious_title(self):
        result_data = {"success": True, "data": {"url": "https://example.com", "title": "Just a moment..."}}
        with (
            patch("tools.browser_tool.check_website_access", return_value=None),
            patch("tools.browser_tool._get_session_info", return_value={"_first_nav": False}),
            patch("tools.browser_tool._run_browser_command", return_value=result_data),
        ):
            result = json.loads(self._nav())

        assert result["success"] is True
        assert "bot_detection_warning" in result


# ── browser_snapshot ─────────────────────────────────────────────────


class TestBrowserSnapshot:
    def test_compact_mode_sends_dash_c_flag(self):
        snap_data = {"success": True, "data": {"snapshot": "page content", "refs": {"@e1": "button"}}}
        with patch("tools.browser_tool._run_browser_command", return_value=snap_data) as mock_cmd:
            from tools.browser_tool import browser_snapshot
            browser_snapshot(full=False, task_id="test")

        args = mock_cmd.call_args[0]
        assert "-c" in args[2]

    def test_full_mode_sends_no_dash_c_flag(self):
        snap_data = {"success": True, "data": {"snapshot": "full content", "refs": {}}}
        with patch("tools.browser_tool._run_browser_command", return_value=snap_data) as mock_cmd:
            from tools.browser_tool import browser_snapshot
            browser_snapshot(full=True, task_id="test")

        args = mock_cmd.call_args[0]
        assert "-c" not in args[2]

    def test_success_returns_snapshot_and_element_count(self):
        snap_data = {"success": True, "data": {"snapshot": "content", "refs": {"@e1": "a", "@e2": "b"}}}
        with patch("tools.browser_tool._run_browser_command", return_value=snap_data):
            from tools.browser_tool import browser_snapshot
            result = json.loads(browser_snapshot(task_id="test"))

        assert result["success"] is True
        assert result["snapshot"] == "content"
        assert result["element_count"] == 2

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_snapshot
            result = json.loads(browser_snapshot(task_id="test"))

        assert result["success"] is False
        assert "error" in result


# ── browser_click ─────────────────────────────────────────────────────


class TestBrowserClick:
    def test_success_returns_clicked_ref(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK):
            from tools.browser_tool import browser_click
            result = json.loads(browser_click("@e5", task_id="test"))

        assert result["success"] is True
        assert result["clicked"] == "@e5"

    def test_auto_prepends_at_sign(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK) as mock_cmd:
            from tools.browser_tool import browser_click
            browser_click("e5", task_id="test")

        args = mock_cmd.call_args[0]
        assert "@e5" in args[2]

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_click
            result = json.loads(browser_click("@e1", task_id="test"))

        assert result["success"] is False
        assert "error" in result


# ── browser_type ──────────────────────────────────────────────────────


class TestBrowserType:
    def test_success_returns_typed_text_and_element(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK):
            from tools.browser_tool import browser_type
            result = json.loads(browser_type("@e3", "hello world", task_id="test"))

        assert result["success"] is True
        assert result["typed"] == "hello world"
        assert result["element"] == "@e3"

    def test_auto_prepends_at_sign(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK) as mock_cmd:
            from tools.browser_tool import browser_type
            browser_type("e3", "text", task_id="test")

        args = mock_cmd.call_args[0]
        assert "@e3" in args[2]

    def test_uses_fill_command(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK) as mock_cmd:
            from tools.browser_tool import browser_type
            browser_type("@e3", "text", task_id="test")

        cmd = mock_cmd.call_args[0][1]
        assert cmd == "fill"

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_type
            result = json.loads(browser_type("@e1", "text", task_id="test"))

        assert result["success"] is False
        assert "error" in result


# ── browser_press ─────────────────────────────────────────────────────


class TestBrowserPress:
    def test_success_returns_pressed_key(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK):
            from tools.browser_tool import browser_press
            result = json.loads(browser_press("Enter", task_id="test"))

        assert result["success"] is True
        assert result["pressed"] == "Enter"

    def test_passes_key_to_command(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK) as mock_cmd:
            from tools.browser_tool import browser_press
            browser_press("Tab", task_id="test")

        args = mock_cmd.call_args[0]
        assert "Tab" in args[2]

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_press
            result = json.loads(browser_press("Enter", task_id="test"))

        assert result["success"] is False


# ── browser_scroll ────────────────────────────────────────────────────


class TestBrowserScroll:
    def test_scroll_down_success(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK):
            from tools.browser_tool import browser_scroll
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is True
        assert result["scrolled"] == "down"

    def test_scroll_up_success(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_OK):
            from tools.browser_tool import browser_scroll
            result = json.loads(browser_scroll("up", task_id="test"))

        assert result["success"] is True
        assert result["scrolled"] == "up"

    def test_invalid_direction_rejected(self):
        from tools.browser_tool import browser_scroll
        result = json.loads(browser_scroll("sideways", task_id="test"))

        assert result["success"] is False
        assert "Invalid direction" in result["error"]

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_scroll
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is False


# ── browser_back ──────────────────────────────────────────────────────


class TestBrowserBack:
    def test_success_returns_url(self):
        back_data = {"success": True, "data": {"url": "https://example.com/prev"}}
        with patch("tools.browser_tool._run_browser_command", return_value=back_data):
            from tools.browser_tool import browser_back
            result = json.loads(browser_back(task_id="test"))

        assert result["success"] is True
        assert result["url"] == "https://example.com/prev"

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_back
            result = json.loads(browser_back(task_id="test"))

        assert result["success"] is False
        assert "error" in result


# ── browser_get_images ────────────────────────────────────────────────


class TestBrowserGetImages:
    def test_success_returns_images_and_count(self):
        images = [{"src": "https://example.com/img.png", "alt": "logo", "width": 100, "height": 50}]
        img_data = {"success": True, "data": {"result": json.dumps(images)}}
        with patch("tools.browser_tool._run_browser_command", return_value=img_data):
            from tools.browser_tool import browser_get_images
            result = json.loads(browser_get_images(task_id="test"))

        assert result["success"] is True
        assert result["count"] == 1
        assert result["images"][0]["src"] == "https://example.com/img.png"

    def test_empty_page_returns_zero_count(self):
        img_data = {"success": True, "data": {"result": "[]"}}
        with patch("tools.browser_tool._run_browser_command", return_value=img_data):
            from tools.browser_tool import browser_get_images
            result = json.loads(browser_get_images(task_id="test"))

        assert result["success"] is True
        assert result["count"] == 0
        assert result["images"] == []

    def test_failure_returns_error(self):
        with patch("tools.browser_tool._run_browser_command", return_value=_FAIL):
            from tools.browser_tool import browser_get_images
            result = json.loads(browser_get_images(task_id="test"))

        assert result["success"] is False


# ── schema coverage ───────────────────────────────────────────────────


EXPECTED_TOOLS = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_scroll",
    "browser_back",
    "browser_close",
    "browser_console",
    "browser_get_images",
    "browser_vision",
]


class TestBrowserSchemas:
    """All 11 browser tools must be registered in BROWSER_TOOL_SCHEMAS."""

    def setup_method(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        self.names = [s["name"] for s in BROWSER_TOOL_SCHEMAS]

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_tool_in_schemas(self, tool_name):
        assert tool_name in self.names, f"{tool_name} missing from BROWSER_TOOL_SCHEMAS"

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_schema_has_description(self, tool_name):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == tool_name)
        assert "description" in schema
        assert len(schema["description"]) > 0


# ── browser skill file ────────────────────────────────────────────────


class TestBrowserSkill:
    """Browser skill SKILL.md exists and is well-formed."""

    @pytest.fixture(autouse=True)
    def _skill_path(self):
        self.skill_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "skills", "browser", "browser", "SKILL.md"
        )

    def test_skill_md_exists(self):
        assert os.path.exists(self.skill_file), "browser/browser/SKILL.md not found"

    def test_skill_has_frontmatter(self):
        with open(self.skill_file) as f:
            content = f.read()
        assert content.startswith("---")
        assert "name: browser" in content
        assert "description:" in content

    def test_skill_documents_all_tools(self):
        with open(self.skill_file) as f:
            content = f.read()
        for tool in EXPECTED_TOOLS:
            assert tool in content, f"{tool} not documented in browser skill"

    def test_skill_has_workflow_section(self):
        with open(self.skill_file) as f:
            content = f.read()
        assert "browser_navigate" in content
        assert "browser_snapshot" in content
        assert "browser_close" in content
