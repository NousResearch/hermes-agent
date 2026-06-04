"""Tests for the Hermes Codex-style computer-use MCP adapter."""

from unittest.mock import patch

import computer_use_mcp_server as adapter


class TestListApps:
    def test_list_apps_combines_running_and_installed(self, monkeypatch):
        monkeypatch.setattr(adapter, "_running_apps", lambda: ["WeChat", "Finder"])
        monkeypatch.setattr(adapter, "_installed_apps", lambda limit=100: ["Finder", "Safari", "WeChat"])

        result = adapter.list_apps_impl(limit=10)

        assert result["success"] is True
        assert result["running_apps"] == ["Finder", "WeChat"]
        assert result["installed_apps"] == ["Finder", "Safari", "WeChat"]


class TestGetAppState:
    def test_get_app_state_activates_requested_app_then_screenshots(self, monkeypatch):
        calls = []

        def fake_cc(**kwargs):
            calls.append(kwargs)
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Safari", "window_title": "Docs"}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["app_name"] == "Safari"
        assert result["window_title"] == "Docs"
        assert result["screenshot_path"] == "/tmp/shot.png"
        assert result["accessibility_tree"] == []
        assert [c["action"] for c in calls] == ["activate_app", "frontmost_app", "screenshot"]

    def test_get_app_state_propagates_capture_error(self, monkeypatch):
        def fake_cc(**kwargs):
            if kwargs["action"] == "frontmost_app":
                return '{"success": true, "app_name": "Finder", "window_title": ""}'
            if kwargs["action"] == "screenshot":
                return '{"error": "screen capture blocked"}'
            return '{"success": true}'

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl()

        assert result["success"] is False
        assert "screen capture blocked" in result["error"]


class TestKeyboardTools:
    def test_type_text_impl_maps_to_computer_control(self, monkeypatch):
        seen = {}

        def fake_cc(**kwargs):
            seen.update(kwargs)
            return '{"success": true}'

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.type_text_impl("uwu")

        assert result["success"] is True
        assert seen == {"action": "keystroke", "text": "uwu"}

    def test_press_key_impl_maps_modifiers(self, monkeypatch):
        seen = {}

        def fake_cc(**kwargs):
            seen.update(kwargs)
            return '{"success": true}'

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.press_key_impl("return", ["command", "shift"])

        assert result["success"] is True
        assert seen == {"action": "keystroke", "key": "return", "modifiers": ["command", "shift"]}


class TestUnsupportedActions:
    def test_click_stub_is_explicit(self):
        result = adapter.click_impl(x=10, y=20)
        assert result["success"] is False
        assert result["supported"] is False
        assert "not implemented" in result["error"].lower()

    def test_set_value_stub_is_explicit(self):
        result = adapter.set_value_impl(index=1, value="hello")
        assert result["success"] is False
        assert result["supported"] is False
        assert "not implemented" in result["error"].lower()
