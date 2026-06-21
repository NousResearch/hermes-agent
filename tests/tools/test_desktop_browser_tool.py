"""Tests for tools/desktop_browser_tool.py — driving the in-app desktop browser pane."""

import json
import os
from unittest import mock

from tools.desktop_browser_tool import (
    browser_navigate_tool,
    browser_read_tool,
    browser_act_tool,
    browser_extract_tool,
    browser_screenshot_tool,
    check_browser_requirements,
    BROWSER_NAVIGATE_SCHEMA,
    BROWSER_READ_SCHEMA,
    BROWSER_ACT_SCHEMA,
    BROWSER_EXTRACT_SCHEMA,
    BROWSER_SCREENSHOT_SCHEMA,
)


class TestBrowserNavigateTool:
    """browser_navigate_tool — opens/drives the in-app browser."""

    def test_navigate_with_url_calls_callback_and_returns_resolved_url(self):
        def cb(url=None):
            assert url == "example.com"
            return "https://example.com/"

        result = json.loads(browser_navigate_tool(url="example.com", callback=cb))
        assert result["url"] == "https://example.com/"

    def test_open_without_url_calls_callback_no_args(self):
        calls = []

        def cb(url=None):
            calls.append(url)
            return "https://duckduckgo.com"

        result = json.loads(browser_navigate_tool(url=None, callback=cb))
        # Opening with no URL must not pass a url kwarg through.
        assert calls == [None] or calls == []
        assert result["url"] == "https://duckduckgo.com"

    def test_no_callback_returns_error(self):
        result = json.loads(browser_navigate_tool(url="x"))
        assert "error" in result
        assert "only available" in result["error"].lower()

    def test_callback_exception_returns_error(self):
        def cb(url=None):
            raise RuntimeError("renderer gone")

        result = json.loads(browser_navigate_tool(url="x", callback=cb))
        assert "error" in result
        assert "renderer gone" in result["error"]

    def test_empty_response_returns_error(self):
        result = json.loads(browser_navigate_tool(url="x", callback=lambda url=None: ""))
        assert "error" in result
        assert "timed out" in result["error"].lower() or "did not open" in result["error"].lower()


class TestBrowserReadTool:
    """browser_read_tool — reports the in-app browser nav state."""

    def test_passes_through_json_state(self):
        state = {
            "open": True,
            "url": "https://example.com/",
            "title": "Example",
            "loading": False,
            "canGoBack": True,
            "canGoForward": False,
            "loadError": None,
        }
        result = json.loads(browser_read_tool(callback=lambda: json.dumps(state)))
        assert result == state

    def test_wraps_non_json_text(self):
        result = json.loads(browser_read_tool(callback=lambda: "plain text"))
        assert result["text"] == "plain text"

    def test_no_callback_returns_error(self):
        result = json.loads(browser_read_tool())
        assert "error" in result
        assert "only available" in result["error"].lower()

    def test_callback_exception_returns_error(self):
        def cb():
            raise RuntimeError("boom")

        result = json.loads(browser_read_tool(callback=cb))
        assert "error" in result
        assert "boom" in result["error"]

    def test_empty_response_returns_error(self):
        result = json.loads(browser_read_tool(callback=lambda: ""))
        assert "error" in result


class TestCheckBrowserRequirements:
    """Desktop-only gating via HERMES_DESKTOP."""

    def test_true_when_desktop_env_set(self):
        for val in ("1", "true", "TRUE", "yes"):
            with mock.patch.dict(os.environ, {"HERMES_DESKTOP": val}):
                assert check_browser_requirements() is True

    def test_false_when_env_absent_or_other(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": ""}):
            assert check_browser_requirements() is False
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": "0"}):
            assert check_browser_requirements() is False


class TestSchemas:
    """OpenAI function-calling schemas."""

    def test_navigate_schema_name_and_optional_url(self):
        assert BROWSER_NAVIGATE_SCHEMA["name"] == "desktop_browser_navigate"
        assert "url" in BROWSER_NAVIGATE_SCHEMA["parameters"]["properties"]
        # url is optional (no required array, or url not in it)
        assert "url" not in BROWSER_NAVIGATE_SCHEMA["parameters"].get("required", [])

    def test_read_schema_name_and_no_params(self):
        assert BROWSER_READ_SCHEMA["name"] == "desktop_browser_read"
        assert BROWSER_READ_SCHEMA["parameters"]["properties"] == {}

    def test_both_registered_under_browser_toolset(self):
        from tools.registry import registry, discover_builtin_tools

        discover_builtin_tools()
        nav = registry.get_entry("desktop_browser_navigate")
        read = registry.get_entry("desktop_browser_read")
        assert nav is not None and nav.toolset == "browser"
        assert read is not None and read.toolset == "browser"

    def test_production_browser_navigate_not_clobbered(self):
        from tools.registry import registry, discover_builtin_tools

        discover_builtin_tools()
        prod = registry.get_entry("browser_navigate")
        assert prod is not None
        # The production web-automation tool must still own browser_navigate.
        assert prod.handler.__module__ != "tools.desktop_browser_tool"


class TestBrowserActTool:
    """browser_act_tool — interact with the in-app browser page."""

    def test_act_calls_callback_with_params_and_returns_json(self):
        def cb(**kw):
            assert kw["action"] == "click"
            assert kw["selector"] == "#button"
            return json.dumps({"ok": True, "action": "click"})

        result = json.loads(browser_act_tool(action="click", selector="#button", callback=cb))
        assert result["ok"] is True
        assert result["action"] == "click"

    def test_no_callback_returns_error(self):
        result = json.loads(browser_act_tool(action="click", selector="#x"))
        assert "error" in result
        assert "only available" in result["error"].lower()

    def test_callback_exception_returns_error(self):
        def cb(**kw):
            raise RuntimeError("boom")

        result = json.loads(browser_act_tool(action="type", text="x", callback=cb))
        assert "error" in result
        assert "boom" in result["error"]

    def test_empty_response_returns_error(self):
        result = json.loads(browser_act_tool(action="scroll", direction="down", callback=lambda **kw: ""))
        assert "error" in result


class TestBrowserExtractTool:
    """browser_extract_tool — extract content from the in-app browser page."""

    def test_extract_calls_callback_with_mode_and_selector(self):
        def cb(mode=None, selector=None):
            assert mode == "text"
            assert selector == ".content"
            return json.dumps({"text": "Hello, world!"})

        result = json.loads(browser_extract_tool(mode="text", selector=".content", callback=cb))
        assert result["text"] == "Hello, world!"

    def test_default_mode_is_text(self):
        called = []

        def cb(mode=None, selector=None):
            called.append(mode)
            return json.dumps({"text": "..."})

        browser_extract_tool(callback=cb)
        assert called == ["text"]

    def test_no_callback_returns_error(self):
        result = json.loads(browser_extract_tool(mode="html"))
        assert "error" in result
        assert "only available" in result["error"].lower()

    def test_callback_exception_returns_error(self):
        def cb(mode=None, selector=None):
            raise RuntimeError("fail")

        result = json.loads(browser_extract_tool(mode="links", callback=cb))
        assert "error" in result
        assert "fail" in result["error"]


class TestBrowserScreenshotTool:
    """browser_screenshot_tool — capture a screenshot of the in-app browser panel."""

    def test_screenshot_calls_callback_and_returns_image_data(self):
        def cb(full_page=False):
            return json.dumps({"image": "data:image/png;base64,iVBORw0K..."})

        result = json.loads(browser_screenshot_tool(callback=cb))
        assert "image" in result
        assert result["image"].startswith("data:image/png")

    def test_full_page_param_forwarded(self):
        called = []

        def cb(full_page=False):
            called.append(full_page)
            return json.dumps({"image": "data:..."})

        browser_screenshot_tool(full_page=True, callback=cb)
        assert called == [True]

    def test_no_callback_returns_error(self):
        result = json.loads(browser_screenshot_tool())
        assert "error" in result
        assert "only available" in result["error"].lower()

    def test_callback_exception_returns_error(self):
        def cb(full_page=False):
            raise RuntimeError("capture failed")

        result = json.loads(browser_screenshot_tool(callback=cb))
        assert "error" in result
        assert "capture failed" in result["error"]
