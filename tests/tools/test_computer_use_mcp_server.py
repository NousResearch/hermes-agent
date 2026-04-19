"""Tests for the Hermes Codex-style computer-use MCP adapter."""

from pathlib import Path
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
    def test_get_app_state_activates_requested_app_then_screenshots(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")
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
        assert result["approved"] is True
        assert result["approval_required"] is False
        assert result["app_name"] == "Safari"
        assert result["window_title"] == "Docs"
        assert result["screenshot_path"] == "/tmp/shot.png"
        assert result["accessibility_tree"] == []
        assert result["virtual_cursor"] == {"x": None, "y": None, "detached": True, "visible": True}
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


class TestApprovalAndSessions:
    def test_approve_list_and_revoke_apps_persist_in_store(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)

        assert adapter.list_approved_apps_impl()["approved_apps"] == []

        approved = adapter.approve_app_impl("Safari")
        assert approved["success"] is True
        assert approved["approved_apps"] == ["Safari"]
        assert store_path.exists() is True

        listed = adapter.list_approved_apps_impl()
        assert listed["approved_apps"] == ["Safari"]

        revoked = adapter.revoke_app_impl("Safari")
        assert revoked["success"] is True
        assert revoked["approved_apps"] == []

    def test_get_app_state_requires_explicit_approval_for_requested_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        calls = []

        def fake_cc(**kwargs):
            calls.append(kwargs)
            return '{"success": true}'

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is False
        assert result["approval_required"] is True
        assert result["app_name"] == "Safari"
        assert calls == []

    def test_get_app_state_returns_stable_session_ids_for_approved_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_SESSION_ID", "session-test")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Safari", "window_title": "Docs"}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        first = adapter.get_app_state_impl(app_name="Safari")
        second = adapter.get_app_state_impl(app_name="Safari")

        assert first["success"] is True
        assert first["approval_required"] is False
        assert first["session_id"] == "session-test"
        assert first["session_id"] == second["session_id"]
        assert first["app_session_id"] == second["app_session_id"]
        assert first["app_name"] == "Safari"

    def test_get_app_state_can_resume_by_app_session_id(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })
        adapter.approve_app_impl("Safari")
        calls = []

        def fake_cc(**kwargs):
            calls.append(kwargs)
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Safari", "window_title": "Docs"}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_session_id="app-1")

        assert result["success"] is True
        assert result["app_name"] == "Safari"
        assert result["app_session_id"] == "app-1"
        assert result["virtual_cursor"] == {"x": 11, "y": 22, "detached": True, "visible": True}
        assert [c["action"] for c in calls] == ["activate_app", "frontmost_app", "screenshot"]
        assert calls[0]["app_name"] == "Safari"

    def test_get_app_state_rejects_unknown_app_session_id(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})

        result = adapter.get_app_state_impl(app_session_id="missing")

        assert result["success"] is False
        assert result["session_required"] is True

    def test_list_active_sessions_and_stop_session(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Safari", "window_title": "Docs"}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        started = adapter.get_app_state_impl(app_name="Safari")
        listed = adapter.list_active_sessions_impl()

        assert started["success"] is True
        assert listed["success"] is True
        assert listed["active_sessions"][0]["app_name"] == "Safari"
        assert listed["active_sessions"][0]["app_session_id"] == started["app_session_id"]
        assert listed["active_sessions"][0]["window_title"] == "Docs"
        assert listed["active_sessions"][0]["screenshot_path"] == "/tmp/shot.png"

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["stopped"] is True
        assert stopped["app_session_id"] == started["app_session_id"]
        assert adapter.list_active_sessions_impl()["active_sessions"] == []

    def test_stop_then_reopen_app_creates_new_app_session_id(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Safari", "window_title": "Docs"}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        first = adapter.get_app_state_impl(app_name="Safari")
        adapter.stop_app_session_impl(app_name="Safari")
        second = adapter.get_app_state_impl(app_name="Safari")

        assert first["app_session_id"] != second["app_session_id"]


class TestKeyboardTools:
    def test_type_text_impl_requires_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})

        result = adapter.type_text_impl("uwu")

        assert result["success"] is False
        assert result["session_required"] is True

    def test_type_text_impl_maps_to_computer_control(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
        })
        seen = {}

        def fake_cc(**kwargs):
            seen.update(kwargs)
            return '{"success": true}'

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.type_text_impl("uwu")

        assert result["success"] is True
        assert result["app_session_id"] == "app-1"
        assert seen == {"action": "keystroke", "text": "uwu"}

    def test_type_text_impl_can_target_specific_app_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
            "notes": {"app_name": "Notes", "app_session_id": "app-2", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
        })

        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.type_text_impl("uwu", app_session_id="app-2")

        assert result["success"] is True
        assert result["app_name"] == "Notes"
        assert result["app_session_id"] == "app-2"

    def test_press_key_impl_requires_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})

        result = adapter.press_key_impl("return", ["command"])

        assert result["success"] is False
        assert result["session_required"] is True

    def test_press_key_impl_maps_modifiers(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
        })
        seen = {}

        def fake_cc(**kwargs):
            seen.update(kwargs)
            return '{"success": true}'

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.press_key_impl("return", ["command", "shift"])

        assert result["success"] is True
        assert result["app_session_id"] == "app-1"
        assert seen == {"action": "keystroke", "key": "return", "modifiers": ["command", "shift"]}

    def test_press_key_impl_can_target_specific_app_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
            "notes": {"app_name": "Notes", "app_session_id": "app-2", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.press_key_impl("return", ["command"], app_session_id="app-2")

        assert result["success"] is True
        assert result["app_name"] == "Notes"
        assert result["app_session_id"] == "app-2"


class TestUnsupportedActions:
    def test_click_requires_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})

        result = adapter.click_impl(x=10, y=20)

        assert result["success"] is False
        assert result["session_required"] is True

    def test_click_updates_virtual_cursor_preview_for_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })

        result = adapter.click_impl(x=10, y=20)

        assert result["success"] is False
        assert result["supported"] is False
        assert result["preview_only"] is True
        assert result["app_session_id"] == "app-1"
        assert result["virtual_cursor"] == {"x": 10, "y": 20, "detached": True, "visible": True}

    def test_click_can_target_specific_app_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            },
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            },
        })

        result = adapter.click_impl(x=50, y=60, app_session_id="app-2")

        assert result["app_name"] == "Notes"
        assert result["app_session_id"] == "app-2"
        assert result["virtual_cursor"] == {"x": 50, "y": 60, "detached": True, "visible": True}
        assert adapter._APP_SESSIONS["safari"]["virtual_cursor"] == {"x": None, "y": None, "detached": True, "visible": True}

    def test_drag_updates_virtual_cursor_to_end_position(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 10, "y": 20, "detached": True, "visible": True},
            }
        })

        result = adapter.drag_impl(start_x=10, start_y=20, end_x=30, end_y=40)

        assert result["success"] is False
        assert result["preview_only"] is True
        assert result["virtual_cursor"] == {"x": 30, "y": 40, "detached": True, "visible": True}

    def test_click_stub_is_explicit(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })

        result = adapter.click_impl(x=10, y=20)
        assert result["success"] is False
        assert result["supported"] is False
        assert "not implemented" in result["error"].lower()

    def test_set_value_stub_is_explicit(self):
        result = adapter.set_value_impl(index=1, value="hello")
        assert result["success"] is False
        assert result["supported"] is False
        assert "not implemented" in result["error"].lower()
