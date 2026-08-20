"""Regression tests for missing/blank browser_navigate URLs."""

import json

from tools.browser_tool import browser_navigate


def test_browser_navigate_blank_url_returns_clean_error():
    result = json.loads(browser_navigate("   \n\t  "))

    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_browser_navigate_non_string_url_returns_clean_error():
    result = json.loads(browser_navigate(None))  # type: ignore[arg-type]

    assert result["success"] is False
    assert "expected string" in result["error"].lower()
