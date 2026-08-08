"""Tests for browser_scroll smart JS scroll (SPA container fallback).

Regression coverage for the bug where browser_scroll was a no-op on SPAs
that set body{overflow:hidden} and use a child element as the real scroll
container (e.g. LinkedIn's <main id="workspace">).

Root cause: the native agent-browser binary calls window.scrollBy(), which
does nothing when the page has locked the body overflow. The fix adds a JS
IIFE as the primary scroll path that walks the DOM to find the real scrollable
container. The native call is retained as a last-resort fallback.
"""

import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _js_ok(container: str, by: int = 500) -> str:
    """Simulate a successful _browser_eval response from the IIFE."""
    return json.dumps({"result": {"scrolled": True, "by": by, "container": container}})


def _js_no_scroll() -> str:
    """Simulate the IIFE reporting that nothing moved (e.g. already at bottom)."""
    return json.dumps({"result": {"scrolled": False, "by": 0, "container": "none"}})


def _native_ok(direction: str) -> dict:
    """Simulate a successful native agent-browser scroll response."""
    return {"success": True, "scrolled": direction}


def _native_fail(msg: str = "No browser session") -> dict:
    return {"success": False, "error": msg}


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestBrowserScrollJSPrimaryPath:
    """JS IIFE fires first; native binary is not called when JS succeeds."""

    def test_normal_page_window_scroll_down(self):
        """Standard page: window.scrollBy works → no container_fallback key."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", return_value=_js_ok("window")) as mock_eval,
            patch("tools.browser_tool._run_browser_command") as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is True
        assert result["scrolled"] == "down"
        assert "method" not in result, "window scroll should not set method=container_fallback"
        assert "container" not in result
        mock_eval.assert_called_once()
        mock_native.assert_not_called()

    def test_normal_page_window_scroll_up(self):
        """Scroll up on a standard page uses window path."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", return_value=_js_ok("window", by=-500)),
            patch("tools.browser_tool._run_browser_command") as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("up", task_id="test"))

        assert result["success"] is True
        assert result["scrolled"] == "up"
        mock_native.assert_not_called()

    def test_spa_custom_container_sets_container_fallback(self):
        """SPA with body{overflow:hidden}: IIFE finds MAIN#workspace → sets method."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", return_value=_js_ok("MAIN#workspace")),
            patch("tools.browser_tool._run_browser_command") as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is True
        assert result["scrolled"] == "down"
        assert result["method"] == "container_fallback"
        assert result["container"] == "MAIN#workspace"
        mock_native.assert_not_called()

    def test_spa_scrolling_element_fallback(self):
        """Third IIFE stage: scrollingElement used → reports as container_fallback."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", return_value=_js_ok("scrollingElement")),
            patch("tools.browser_tool._run_browser_command") as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is True
        assert result["method"] == "container_fallback"
        assert result["container"] == "scrollingElement"
        mock_native.assert_not_called()

    def test_dy_is_positive_for_down(self):
        """IIFE receives +500 when direction is 'down'."""
        from tools.browser_tool import browser_scroll

        captured = {}

        def capture_js(js_str, *args, **kwargs):
            captured["js"] = js_str
            return _js_ok("window")

        with (
            patch("tools.browser_tool._browser_eval", side_effect=capture_js),
            patch("tools.browser_tool._run_browser_command"),
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            browser_scroll("down", task_id="test")

        assert captured["js"].endswith(")(500)")

    def test_dy_is_negative_for_up(self):
        """IIFE receives -500 when direction is 'up'."""
        from tools.browser_tool import browser_scroll

        captured = {}

        def capture_js(js_str, *args, **kwargs):
            captured["js"] = js_str
            return _js_ok("window", by=-500)

        with (
            patch("tools.browser_tool._browser_eval", side_effect=capture_js),
            patch("tools.browser_tool._run_browser_command"),
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            browser_scroll("up", task_id="test")

        assert captured["js"].endswith(")(-500)")


# ---------------------------------------------------------------------------
# Native fallback
# ---------------------------------------------------------------------------


class TestBrowserScrollNativeFallback:
    """Native binary is used only when the JS path fails entirely."""

    def test_falls_back_to_native_when_browser_eval_raises(self):
        """Exception from _browser_eval → native scroll called."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", side_effect=RuntimeError("CDP gone")),
            patch("tools.browser_tool._run_browser_command", return_value=_native_ok("down")) as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is True
        assert result["scrolled"] == "down"
        mock_native.assert_called_once()

    def test_falls_back_to_native_when_js_reports_no_scroll(self):
        """IIFE returns scrolled:false (page already at boundary) → try native."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", return_value=_js_no_scroll()),
            patch("tools.browser_tool._run_browser_command", return_value=_native_ok("down")) as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="test"))

        # Native succeeded
        assert result["success"] is True
        mock_native.assert_called_once()

    def test_falls_back_to_native_when_eval_returns_invalid_json(self):
        """Malformed JSON from _browser_eval → falls through to native."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", return_value="not-json"),
            patch("tools.browser_tool._run_browser_command", return_value=_native_ok("up")) as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("up", task_id="test"))

        assert result["success"] is True
        mock_native.assert_called_once()

    def test_native_failure_propagated(self):
        """If JS fails AND native fails, return success:false with native error."""
        from tools.browser_tool import browser_scroll

        with (
            patch("tools.browser_tool._browser_eval", side_effect=RuntimeError("no CDP")),
            patch("tools.browser_tool._run_browser_command", return_value=_native_fail("No session")),
            patch("tools.browser_tool._last_session_key", return_value="test"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="test"))

        assert result["success"] is False
        assert "No session" in result["error"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestBrowserScrollValidation:
    def test_invalid_direction_returns_error(self):
        from tools.browser_tool import browser_scroll

        result = json.loads(browser_scroll("sideways", task_id="test"))

        assert result["success"] is False
        assert "Invalid direction" in result["error"]
        assert "sideways" in result["error"]

    def test_empty_direction_returns_error(self):
        from tools.browser_tool import browser_scroll

        result = json.loads(browser_scroll("", task_id="test"))

        assert result["success"] is False
        assert "Invalid direction" in result["error"]


# ---------------------------------------------------------------------------
# Regression: original bug (window.scrollBy no-op on SPAs)
# ---------------------------------------------------------------------------


class TestBrowserScrollSPARegressionLinkedIn:
    """Regression test for the LinkedIn/SPA scroll bug.

    Before the fix, browser_scroll called window.scrollBy via the native
    agent-browser binary. On pages with body{overflow:hidden} this was a
    complete no-op — scrollY never changed — but the tool still returned
    {"success": true}. Downstream callers received stale DOM snapshots.

    The fix: JS IIFE as primary path, ancestor walk to find the real
    scrollable container (e.g. <main id="workspace">).
    """

    def test_spa_scroll_succeeds_without_native_binary(self):
        """Simulate LinkedIn: body locked, MAIN#workspace is the real container."""
        from tools.browser_tool import browser_scroll

        # Simulate: window.scrollY didn't move, but MAIN#workspace did
        iife_response = json.dumps({
            "result": {"scrolled": True, "by": 500, "container": "MAIN#workspace"}
        })

        with (
            patch("tools.browser_tool._browser_eval", return_value=iife_response),
            patch("tools.browser_tool._run_browser_command") as mock_native,
            patch("tools.browser_tool._last_session_key", return_value="linkedin-session"),
            patch("tools.browser_tool._is_camofox_mode", return_value=False),
        ):
            result = json.loads(browser_scroll("down", task_id="linkedin-session"))

        assert result["success"] is True
        assert result["method"] == "container_fallback"
        assert result["container"] == "MAIN#workspace"
        # Critical: native binary must NOT be called — it would be a no-op
        mock_native.assert_not_called()

    def test_old_behaviour_was_silent_noop(self):
        """Document the pre-fix behaviour for clarity.

        Before the fix: _browser_eval was never called; only the native binary
        ran. On SPAs the native binary returned success:true even though nothing
        moved. This test confirms that if we skip _browser_eval entirely and
        only call the native binary we get a misleading success with no
        container_fallback information.
        """
        # Simulate the old code path: native binary only, claims success
        fake_native_result = {"success": True, "scrolled": "down"}

        # Reconstruct the old one-liner result manually (no JS path)
        import json as _json
        from tools.browser_tool import _copy_fallback_warning
        response = {"success": True, "scrolled": "down"}
        old_result = _json.loads(_json.dumps(_copy_fallback_warning(response, fake_native_result)))

        # The old result has no indication of which container actually moved
        assert old_result["success"] is True
        assert "method" not in old_result  # no container_fallback info
        assert "container" not in old_result  # silent lie on SPAs
