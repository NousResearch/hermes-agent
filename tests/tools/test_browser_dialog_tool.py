"""Tests for tools/browser_dialog_tool.py.

Covers the browser_dialog handler: no supervisor attached, successful
accept/dismiss, and error response from the supervisor.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.browser_dialog_tool import browser_dialog


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear SUPERVISOR_REGISTRY before each test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    with SUPERVISOR_REGISTRY._lock:
        SUPERVISOR_REGISTRY._by_task.clear()
    yield
    with SUPERVISOR_REGISTRY._lock:
        SUPERVISOR_REGISTRY._by_task.clear()


class TestBrowserDialogNoSupervisor:
    """When no CDP supervisor is attached, return a helpful error."""

    def test_no_supervisor_default_task(self):
        result = browser_dialog(action="accept")
        data = json.loads(result)
        assert data["success"] is False
        assert "No CDP supervisor" in data["error"]

    def test_no_supervisor_custom_task_id(self):
        result = browser_dialog(action="dismiss", task_id="my-task")
        data = json.loads(result)
        assert data["success"] is False
        assert "No CDP supervisor" in data["error"]


class TestBrowserDialogSuccess:
    """When the supervisor responds with ok=True, return success JSON."""

    def test_accept_success(self):
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SimpleNamespace(
            respond_to_dialog=lambda action, prompt_text, dialog_id: {
                "ok": True,
                "dialog": {"id": "d1", "type": "alert", "message": "hello"},
            }
        )
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

        result = browser_dialog(action="accept")
        data = json.loads(result)
        assert data["success"] is True
        assert data["action"] == "accept"
        assert data["dialog"]["id"] == "d1"

    def test_dismiss_success(self):
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SimpleNamespace(
            respond_to_dialog=lambda action, prompt_text, dialog_id: {
                "ok": True,
                "dialog": {"id": "d2", "type": "confirm"},
            }
        )
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

        result = browser_dialog(action="dismiss")
        data = json.loads(result)
        assert data["success"] is True
        assert data["action"] == "dismiss"

    def test_prompt_text_forwarded(self):
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        captured = {}

        def fake_respond(action, prompt_text, dialog_id):
            captured["prompt_text"] = prompt_text
            captured["dialog_id"] = dialog_id
            return {"ok": True, "dialog": {}}

        supervisor = SimpleNamespace(respond_to_dialog=fake_respond)
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

        result = browser_dialog(
            action="accept",
            prompt_text="my answer",
            dialog_id="dlg-42",
        )
        data = json.loads(result)
        assert data["success"] is True
        assert captured["prompt_text"] == "my answer"
        assert captured["dialog_id"] == "dlg-42"


class TestBrowserDialogError:
    """When the supervisor responds with ok=False, return error JSON."""

    def test_supervisor_error(self):
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SimpleNamespace(
            respond_to_dialog=lambda action, prompt_text, dialog_id: {
                "ok": False,
                "error": "dialog already handled",
            }
        )
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

        result = browser_dialog(action="accept")
        data = json.loads(result)
        assert data["success"] is False
        assert data["error"] == "dialog already handled"

    def test_supervisor_error_default_message(self):
        """When ok=False but no error key, use 'unknown error'."""
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SimpleNamespace(
            respond_to_dialog=lambda action, prompt_text, dialog_id: {
                "ok": False,
            }
        )
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

        result = browser_dialog(action="dismiss")
        data = json.loads(result)
        assert data["success"] is False
        assert data["error"] == "unknown error"


class TestBrowserDialogCheck:
    """Cover _browser_dialog_check gate function."""

    def test_check_delegates_to_cdp_check_true(self):
        from tools.browser_dialog_tool import _browser_dialog_check

        with patch("tools.browser_cdp_tool._browser_cdp_check", return_value=True):
            assert _browser_dialog_check() is True

    def test_check_delegates_to_cdp_check_false(self):
        from tools.browser_dialog_tool import _browser_dialog_check

        with patch("tools.browser_cdp_tool._browser_cdp_check", return_value=False):
            assert _browser_dialog_check() is False
