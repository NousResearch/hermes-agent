"""Tests that browser_get_images blocks image data from eval-navigated private pages.

browser_snapshot, browser_vision, and _browser_eval all re-check the page URL
before returning content, but browser_get_images bypasses _browser_eval and
calls _run_browser_command("eval", ...) directly. Without its own guard, image
src URLs and alt text from a private page would leak.

Sibling of the snapshot/vision/eval guards for issue #44731.
"""

import json

import pytest

from tools import browser_tool

PRIVATE_URL = "http://127.0.0.1:8080/internal"
IMAGES_JS_RESULT = json.dumps([
    {"src": "http://127.0.0.1:8080/logo.png", "alt": "Internal Logo", "width": 200, "height": 100},
])


@pytest.fixture(autouse=True)
def _patches(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda key: key)


def _mock_run_success(monkeypatch):
    def _run(task_id, command, args=None, **kwargs):
        return {"success": True, "data": {"result": IMAGES_JS_RESULT}}
    monkeypatch.setattr(browser_tool, "_run_browser_command", _run)


def test_blocks_images_on_private_page(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda tid: True)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda tid: PRIVATE_URL)

    result = json.loads(browser_tool.browser_get_images(task_id="test"))
    assert result["success"] is False
    assert "private or internal address" in result["error"]
    assert PRIVATE_URL in result["error"]


def test_allows_images_on_public_page(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda tid: True)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda tid: None)

    result = json.loads(browser_tool.browser_get_images(task_id="test"))
    assert result["success"] is True
    assert result["count"] == 1
    assert result["images"][0]["src"] == "http://127.0.0.1:8080/logo.png"


def test_skips_guard_for_local_backend(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda tid: False)

    result = json.loads(browser_tool.browser_get_images(task_id="test"))
    assert result["success"] is True
    assert result["count"] == 1


def test_skips_guard_when_private_urls_allowed(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda tid: False)

    result = json.loads(browser_tool.browser_get_images(task_id="test"))
    assert result["success"] is True
    assert result["count"] == 1


def test_guard_does_not_block_on_failed_eval(monkeypatch):
    """If the eval itself fails, browser_get_images returns its own error — no guard needed."""
    def _run(task_id, command, args=None, **kwargs):
        return {"success": False, "error": "eval failed"}
    monkeypatch.setattr(browser_tool, "_run_browser_command", _run)

    result = json.loads(browser_tool.browser_get_images(task_id="test"))
    assert result["success"] is False
    assert "eval failed" in result["error"]
