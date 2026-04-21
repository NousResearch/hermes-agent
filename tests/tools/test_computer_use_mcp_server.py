"""Tests for the Hermes Codex-style computer-use MCP adapter."""

import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import computer_use_mcp_server as adapter


_REAL_RESOLVE_APPROVAL_TARGET = adapter._resolve_approval_target


@pytest.fixture(autouse=True)
def _stub_approval_resolution(monkeypatch):
    known = {
        "safari": ("com.apple.Safari", "/Applications/Safari.app"),
        "textedit": ("com.apple.TextEdit", "/System/Applications/TextEdit.app"),
        "finder": ("com.apple.finder", "/System/Library/CoreServices/Finder.app"),
        "wechat": ("com.tencent.xinWeChat", "/Applications/WeChat.app"),
    }

    def fake_resolve(app_name: str):
        normalized = adapter._normalize_app_name(app_name)
        if not normalized:
            return None
        bundle_id, bundle_path = known.get(
            normalized.casefold(),
            (f"com.example.{normalized.casefold().replace(' ', '-')}", f"/Applications/{normalized}.app"),
        )
        return {
            "approval_name": normalized,
            "bundle_id": bundle_id,
            "bundle_path": bundle_path,
        }

    monkeypatch.setattr(adapter, "_resolve_approval_target", fake_resolve, raising=False)


class TestApprovalResolution:
    def test_resolve_approval_target_falls_back_to_spotlight_path_when_path_lookup_hangs(self, monkeypatch):
        calls = []

        def fake_run(cmd, check=False, capture_output=False, text=False, timeout=None):
            calls.append((tuple(cmd), timeout))
            if cmd[0] == "osascript" and any("id of application appRef" in part for part in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="com.apple.TextEdit\n", stderr="")
            if cmd[0] == "mdfind":
                return subprocess.CompletedProcess(cmd, 0, stdout="/System/Applications/TextEdit.app\n", stderr="")
            if cmd[0] == "mdls":
                return subprocess.CompletedProcess(cmd, 0, stdout='kMDItemCFBundleIdentifier = "com.apple.TextEdit"\n', stderr="")
            if cmd[0] == "osascript" and any("path to application" in part for part in cmd):
                raise subprocess.TimeoutExpired(cmd, timeout or 5)
            raise AssertionError(cmd)

        monkeypatch.setattr(adapter.subprocess, "run", fake_run)

        resolved = _REAL_RESOLVE_APPROVAL_TARGET("TextEdit")

        assert resolved == {
            "approval_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_path": adapter._normalize_bundle_path("/System/Applications/TextEdit.app"),
        }
        assert not any(cmd[0] == "osascript" and any("path to application" in part for part in cmd) for cmd, _ in calls)

    def test_resolve_approval_target_prefers_later_verified_spotlight_candidate(self, monkeypatch):
        calls = []

        def fake_run(cmd, check=False, capture_output=False, text=False, timeout=None):
            calls.append(tuple(cmd))
            if cmd[0] == "osascript" and any("id of application appRef" in part for part in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="com.apple.TextEdit\n", stderr="")
            if cmd[0] == "mdfind":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="/Applications/FakeTextEdit.app\n/System/Applications/TextEdit.app\n",
                    stderr="",
                )
            if cmd[0] == "mdls" and cmd[-1] == "/Applications/FakeTextEdit.app":
                raise subprocess.TimeoutExpired(cmd, timeout or 5)
            if cmd[0] == "mdls" and cmd[-1] == "/System/Applications/TextEdit.app":
                return subprocess.CompletedProcess(cmd, 0, stdout='kMDItemCFBundleIdentifier = "com.apple.TextEdit"\n', stderr="")
            if cmd[0] == "osascript" and any("path to application" in part for part in cmd):
                raise AssertionError("should not need AppleScript path fallback when a later Spotlight candidate verifies")
            raise AssertionError(cmd)

        monkeypatch.setattr(adapter.subprocess, "run", fake_run)

        resolved = _REAL_RESOLVE_APPROVAL_TARGET("TextEdit")

        assert resolved == {
            "approval_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_path": adapter._normalize_bundle_path("/System/Applications/TextEdit.app"),
        }
        assert any(cmd[0] == "mdls" and cmd[-1] == "/System/Applications/TextEdit.app" for cmd in calls)

    def test_resolve_approval_target_falls_back_to_applescript_path_when_spotlight_candidates_mismatch(self, monkeypatch):
        calls = []

        def fake_run(cmd, check=False, capture_output=False, text=False, timeout=None):
            calls.append(tuple(cmd))
            if cmd[0] == "osascript" and any("id of application appRef" in part for part in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="com.apple.TextEdit\n", stderr="")
            if cmd[0] == "mdfind":
                return subprocess.CompletedProcess(cmd, 0, stdout="/Applications/FakeTextEdit.app\n", stderr="")
            if cmd[0] == "mdls":
                return subprocess.CompletedProcess(cmd, 0, stdout='kMDItemCFBundleIdentifier = "com.example.fake"\n', stderr="")
            if cmd[0] == "osascript" and any("path to application" in part for part in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="/System/Applications/TextEdit.app\n", stderr="")
            raise AssertionError(cmd)

        monkeypatch.setattr(adapter.subprocess, "run", fake_run)

        resolved = _REAL_RESOLVE_APPROVAL_TARGET("TextEdit")

        assert resolved == {
            "approval_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_path": adapter._normalize_bundle_path("/System/Applications/TextEdit.app"),
        }
        assert any(cmd[0] == "osascript" and any("path to application" in part for part in cmd) for cmd in calls)

    def test_resolve_approval_target_backfills_bundle_id_from_path_when_id_lookup_fails(self, monkeypatch):
        calls = []

        def fake_run(cmd, check=False, capture_output=False, text=False, timeout=None):
            calls.append(tuple(cmd))
            if cmd[0] == "osascript" and any("id of application appRef" in part for part in cmd):
                raise subprocess.CalledProcessError(1, cmd, stderr="can't get id")
            if cmd[0] == "osascript" and any("path to application" in part for part in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="/System/Applications/TextEdit.app\n", stderr="")
            if cmd[0] == "mdls":
                return subprocess.CompletedProcess(cmd, 0, stdout='kMDItemCFBundleIdentifier = "com.apple.TextEdit"\n', stderr="")
            if cmd[0] == "mdfind":
                raise AssertionError("should not try Spotlight path resolution without a bundle id")
            raise AssertionError(cmd)

        monkeypatch.setattr(adapter.subprocess, "run", fake_run)

        resolved = _REAL_RESOLVE_APPROVAL_TARGET("TextEdit")

        assert resolved == {
            "approval_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_path": adapter._normalize_bundle_path("/System/Applications/TextEdit.app"),
        }
        assert any(cmd[0] == "mdls" and cmd[-1] == "/System/Applications/TextEdit.app" for cmd in calls)


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
                return '{"success": true, "app_name": "Safari", "bundle_id": "com.apple.Safari", "bundle_name": "Safari", "bundle_path": "/Applications/Safari.app", "process_id": 999, "window_title": "Docs", "window_id": 123, "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480}}'
            if action == "screenshot":
                assert kwargs["window_id"] == 123
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 123}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["approved"] is True
        assert result["approval_required"] is False
        assert result["app_name"] == "Safari"
        assert result["window_title"] == "Docs"
        assert result["window_id"] == 123
        assert result["window_bounds"] == {"x": 1, "y": 2, "width": 640, "height": 480}
        assert result["bundle_id"] == "com.apple.Safari"
        assert result["process_id"] == 999
        assert result["screenshot_path"] == "/tmp/shot.png"
        assert result["accessibility_tree"] == []
        assert result["virtual_cursor"] == {"x": None, "y": None, "detached": True, "visible": True}
        assert [c["action"] for c in calls] == ["activate_app", "frontmost_app", "screenshot"]

    def test_get_app_state_accepts_temporary_session_approval(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.grant_temporary_app_approval_impl("TextEdit", scope="session")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "TextEdit",
                    "bundle_id": "com.apple.TextEdit",
                    "bundle_name": "TextEdit",
                    "bundle_path": "/System/Applications/TextEdit.app",
                    "process_id": 999,
                    "window_title": "Scratch",
                    "window_id": 123,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 123}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "bundle_path": "/System/Applications/TextEdit.app",
            "process_id": 999,
            "window_title": "Scratch",
            "window_id": 123,
            "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
            "accessibility_tree": [],
        })

        result = adapter.get_app_state_impl(app_name="TextEdit")

        assert result["success"] is True
        assert result["approved"] is True
        assert result["approval_scope"] == "session"
        assert result["app_name"] == "TextEdit"

    def test_get_app_state_consumes_once_temporary_approval_after_success(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.grant_temporary_app_approval_impl("TextEdit", scope="once")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "TextEdit",
                    "bundle_id": "com.apple.TextEdit",
                    "bundle_name": "TextEdit",
                    "bundle_path": "/System/Applications/TextEdit.app",
                    "process_id": 999,
                    "window_title": "Scratch",
                    "window_id": 123,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 123}'
            if action == "keystroke":
                return '{"success": true}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "bundle_path": "/System/Applications/TextEdit.app",
            "process_id": 999,
            "window_title": "Scratch",
            "window_id": 123,
            "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
            "accessibility_tree": [],
        })

        first = adapter.get_app_state_impl(app_name="TextEdit")
        second = adapter.type_text_impl("uwu", app_session_id=first["app_session_id"])

        assert first["success"] is True
        assert first["approved"] is True
        assert first["approval_scope"] == "once"
        assert second["success"] is False
        assert second["approval_required"] is True

    def test_get_app_state_includes_accessibility_tree_and_persists_it_in_session_state(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        adapter.approve_app_impl("Safari")
        accessibility_tree = [
            {
                "index": 0,
                "role": "AXWindow",
                "title": "Docs",
                "children": [
                    {"index": 1, "role": "AXTextArea", "value": "dragon", "children": []},
                ],
            },
        ]

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.apple.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "bundle_path": "/Applications/Safari.app",
                    "process_id": 999,
                    "window_title": "Docs",
                    "window_id": 123,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 123}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
            "process_id": 999,
            "window_title": "Docs",
            "window_id": 123,
            "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
            "accessibility_tree": accessibility_tree,
        })

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["accessibility_tree"] == accessibility_tree
        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["accessibility_tree"] == accessibility_tree
        listed = adapter.list_active_sessions_impl()
        assert listed["active_sessions"][0]["accessibility_tree"] == accessibility_tree

    def test_get_app_state_discards_accessibility_tree_when_second_snapshot_does_not_match(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.apple.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "process_id": 999,
                    "window_title": "Docs",
                    "window_id": 123,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 123}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "TextEdit",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "bundle_path": "/System/Applications/TextEdit.app",
            "window_title": "Secrets",
            "window_id": 777,
            "window_bounds": {"x": 40, "y": 50, "width": 500, "height": 300},
            "accessibility_tree": [
                {"index": 0, "role": "AXWindow", "title": "Secrets", "children": []},
            ],
        })

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["approved"] is True
        assert result["accessibility_tree"] == []
        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["accessibility_tree"] == []

    def test_get_app_state_discards_accessibility_tree_for_same_title_different_bounds_without_window_id(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.apple.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "process_id": 999,
                    "window_title": "Docs",
                    "window_id": None,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
            "process_id": 999,
            "window_title": "Docs",
            "window_id": None,
            "window_bounds": {"x": 120, "y": 140, "width": 500, "height": 300},
            "accessibility_tree": [
                {"index": 0, "role": "AXWindow", "title": "Docs", "children": []},
            ],
        })

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["accessibility_tree"] == []

    def test_get_app_state_discards_accessibility_tree_for_same_bounds_different_title(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.apple.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "process_id": 999,
                    "window_title": "Public Doc",
                    "window_id": None,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
            "process_id": 999,
            "window_title": "Secret Doc",
            "window_id": None,
            "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
            "accessibility_tree": [
                {"index": 0, "role": "AXWindow", "title": "Secret Doc", "children": []},
            ],
        })

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["accessibility_tree"] == []

    def test_get_app_state_returns_structured_error_when_accessibility_snapshot_fails(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.apple.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "process_id": 999,
                    "window_title": "Docs",
                    "window_id": 123,
                    "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 123}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        def boom(include_accessibility=True, require_helper_success=False):
            raise RuntimeError("accessibility helper failed")

        monkeypatch.setattr(adapter, "frontmost_window_state", boom)

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is False
        assert "accessibility helper failed" in result["error"]

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
    def test_get_app_state_marks_session_approved_even_if_frontmost_name_is_localized(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("TextEdit")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "文本编辑", "bundle_id": "com.apple.TextEdit", "bundle_name": "TextEdit", "bundle_path": "/System/Applications/TextEdit.app", "process_id": 321, "window_title": "Notes", "window_id": 777, "window_bounds": {"x": 3, "y": 4, "width": 500, "height": 300}}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 777}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="TextEdit")

        assert result["success"] is True
        assert result["approved"] is True
        assert result["app_name"] == "文本编辑"
        assert result["bundle_id"] == "com.apple.TextEdit"

    def test_get_app_state_by_app_session_id_keeps_localized_approved_session_working(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.approve_app_impl("TextEdit")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "文本编辑", "bundle_id": "com.apple.TextEdit", "bundle_name": "TextEdit", "bundle_path": "/System/Applications/TextEdit.app", "process_id": 321, "window_title": "Notes", "window_id": 777, "window_bounds": {"x": 3, "y": 4, "width": 500, "height": 300}}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 777}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        first = adapter.get_app_state_impl(app_name="TextEdit")
        second = adapter.get_app_state_impl(app_session_id=first["app_session_id"])

        assert first["success"] is True
        assert second["success"] is True
        assert second["approval_required"] is False
        assert second["approved"] is True
        assert second["app_session_id"] == first["app_session_id"]
        assert second["app_name"] == "文本编辑"

    def test_get_app_state_by_app_session_id_revocation_marks_same_localized_session_unapproved(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.approve_app_impl("TextEdit")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "文本编辑", "bundle_id": "com.apple.TextEdit", "bundle_name": "TextEdit", "bundle_path": "/System/Applications/TextEdit.app", "process_id": 321, "window_title": "Notes", "window_id": 777, "window_bounds": {"x": 3, "y": 4, "width": 500, "height": 300}}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 777}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        first = adapter.get_app_state_impl(app_name="TextEdit")
        adapter.revoke_app_impl("TextEdit")
        second = adapter.get_app_state_impl(app_session_id=first["app_session_id"])
        typed = adapter.type_text_impl("uwu", app_session_id=first["app_session_id"])

        assert first["success"] is True
        assert second["success"] is False
        assert second["approval_required"] is True
        assert second["approved"] is False
        assert second["app_session_id"] == first["app_session_id"]
        assert len(adapter._APP_SESSIONS) == 1
        assert typed["success"] is False
        assert typed["approval_required"] is True
        assert typed["app_session_id"] == first["app_session_id"]

    def test_get_app_state_without_requested_app_persists_approved_identity_for_localized_frontmost_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        adapter.approve_app_impl("TextEdit")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "文本编辑", "bundle_id": "com.apple.TextEdit", "bundle_name": "TextEdit", "bundle_path": "/System/Applications/TextEdit.app", "process_id": 321, "window_title": "Notes", "window_id": 777, "window_bounds": {"x": 3, "y": 4, "width": 500, "height": 300}}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 777}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        first = adapter.get_app_state_impl()
        second = adapter.get_app_state_impl(app_session_id=first["app_session_id"])

        assert first["success"] is True
        assert first["approved"] is True
        assert second["success"] is True
        assert second["approved"] is True
        assert second["app_session_id"] == first["app_session_id"]
        assert second["app_name"] == "文本编辑"

    def test_get_app_state_does_not_inherit_approval_from_requested_app_when_frontmost_differs(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("TextEdit")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Safari", "bundle_id": "com.apple.Safari", "bundle_name": "Safari", "bundle_path": "/Applications/Safari.app", "process_id": 222, "window_title": "Other", "window_id": 888, "window_bounds": {"x": 5, "y": 6, "width": 700, "height": 400}}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 888}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="TextEdit")

        assert result["success"] is True
        assert result["approved"] is False
        assert result["app_name"] == "Safari"
        assert result["bundle_id"] == "com.apple.Safari"

    def test_get_app_state_redacts_accessibility_tree_for_unapproved_frontmost_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        adapter.approve_app_impl("TextEdit")
        leaked_tree = [
            {
                "index": 0,
                "role": "AXWindow",
                "title": "Safari Secrets",
                "children": [
                    {"index": 1, "role": "AXTextArea", "value": "top secret", "children": []},
                ],
            },
        ]

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.apple.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "process_id": 222,
                    "window_title": "Other",
                    "window_id": 888,
                    "window_bounds": {"x": 5, "y": 6, "width": 700, "height": 400},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/shot.png", "media_tag": "MEDIA:/tmp/shot.png", "window_id": 888}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "frontmost_window_state", lambda include_accessibility=True, require_helper_success=False: {
            "app_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
            "process_id": 222,
            "window_title": "Other",
            "window_id": 888,
            "window_bounds": {"x": 5, "y": 6, "width": 700, "height": 400},
            "accessibility_tree": leaked_tree,
        })

        result = adapter.get_app_state_impl(app_name="TextEdit")

        assert result["success"] is True
        assert result["approved"] is False
        assert result["accessibility_tree"] == []
        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["accessibility_tree"] == []
        listed = adapter.list_active_sessions_impl()
        assert listed["active_sessions"][0]["accessibility_tree"] == []

    def test_get_app_state_does_not_approve_same_display_name_with_different_bundle_id(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return json.dumps({
                    "success": True,
                    "app_name": "Safari",
                    "bundle_id": "com.evil.Safari",
                    "bundle_name": "Safari",
            "bundle_path": "/Applications/Safari.app",
                    "process_id": 999,
                    "window_title": "Trap",
                    "window_id": 456,
                    "window_bounds": {"x": 10, "y": 20, "width": 640, "height": 480},
                })
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/evil-shot.png", "media_tag": "MEDIA:/tmp/evil-shot.png", "window_id": 456}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["approved"] is False
        assert result["bundle_id"] == "com.evil.Safari"

    def test_get_app_state_does_not_trust_bundle_id_suffix_alone_for_approval(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("TextEdit")

        def fake_cc(**kwargs):
            action = kwargs["action"]
            if action == "activate_app":
                return '{"success": true}'
            if action == "frontmost_app":
                return '{"success": true, "app_name": "Evil Notes", "bundle_id": "com.evil.TextEdit", "bundle_name": "Evil Notes", "process_id": 444, "window_title": "Trap", "window_id": 889, "window_bounds": {"x": 8, "y": 9, "width": 710, "height": 410}}'
            if action == "screenshot":
                return '{"success": true, "path": "/tmp/evil-shot.png", "media_tag": "MEDIA:/tmp/evil-shot.png", "window_id": 889}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="TextEdit")

        assert result["success"] is True
        assert result["approved"] is False
        assert result["app_name"] == "Evil Notes"
        assert result["bundle_id"] == "com.evil.TextEdit"

    def test_get_app_state_falls_back_to_display_capture_when_window_capture_fails(self, monkeypatch, tmp_path):
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
                return '{"success": true, "app_name": "Safari", "bundle_id": "com.apple.Safari", "bundle_name": "Safari", "bundle_path": "/Applications/Safari.app", "process_id": 999, "window_title": "Docs", "window_id": 123, "window_bounds": {"x": 1, "y": 2, "width": 640, "height": 480}}'
            if action == "screenshot" and kwargs.get("window_id") == 123:
                return '{"error": "window capture blocked"}'
            if action == "screenshot":
                assert "window_id" not in kwargs
                return '{"success": true, "path": "/tmp/fallback-shot.png", "media_tag": "MEDIA:/tmp/fallback-shot.png"}'
            raise AssertionError(action)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)

        result = adapter.get_app_state_impl(app_name="Safari")

        assert result["success"] is True
        assert result["screenshot_path"] == "/tmp/fallback-shot.png"
        assert [c["action"] for c in calls] == ["activate_app", "frontmost_app", "screenshot", "screenshot"]
        assert calls[2]["window_id"] == 123
        assert "window_id" not in calls[3]


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
        payload = json.loads(store_path.read_text())
        assert payload["approved_apps"] == []
        assert payload["approved_app_entries"] == []

    def test_approve_app_persists_resolved_bundle_identity(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)

        approved = adapter.approve_app_impl("Safari")

        assert approved["success"] is True
        assert approved["approved_apps"] == ["Safari"]
        payload = json.loads(store_path.read_text())
        assert payload["approved_apps"] == ["Safari"]
        assert payload["approved_app_entries"] == [
            {
                "approval_name": "Safari",
                "bundle_id": "com.apple.Safari",
                "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
            }
        ]

    def test_load_approval_entries_backfills_missing_bundle_path_for_legacy_entry(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        store_path.write_text(json.dumps({
            "approved_apps": ["Safari"],
            "approved_app_entries": [
                {"approval_name": "Safari", "bundle_id": "com.apple.Safari", "bundle_path": ""},
            ],
        }))

        entries = adapter._load_approval_entries()

        assert entries == [
            {
                "approval_name": "Safari",
                "bundle_id": "com.apple.Safari",
                "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
            }
        ]

    def test_approve_app_invalidates_stale_same_label_session_when_identity_changes(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        state_root.mkdir()
        state_path = state_root / "app-1.json"
        state_path.write_text("{}")
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: state_path.write_text(json.dumps(adapter._session_payload(session))) or str(state_path), raising=False)
        monkeypatch.setattr(adapter, "_resolve_approval_target", lambda app_name: {
            "approval_name": "Safari",
            "bundle_id": "com.apple.Safari",
            "bundle_path": "/Applications/Safari.app",
        }, raising=False)
        adapter.approve_app_impl("Safari")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "approval_name": "Safari",
                "approval_bundle_id": "com.apple.Safari",
                "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
                "bundle_id": "com.apple.Safari",
                "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "accessibility_tree": [{"index": 0, "role": "AXWindow", "title": "Docs", "children": []}],
                "session_state_path": str(state_path),
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })
        monkeypatch.setattr(adapter, "_resolve_approval_target", lambda app_name: {
            "approval_name": "Safari",
            "bundle_id": "com.example.alt-safari",
            "bundle_path": "/Applications/AltSafari.app",
        }, raising=False)

        approved = adapter.approve_app_impl("Safari")

        assert approved["success"] is True
        assert adapter._APP_SESSIONS["safari"]["approved"] is False
        assert adapter._APP_SESSIONS["safari"]["accessibility_tree"] == []
        payload = json.loads(state_path.read_text())
        assert payload["approved"] is False
        assert payload["accessibility_tree"] == []

    def test_revoke_app_clears_accessibility_tree_from_existing_sessions(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        state_root.mkdir()
        state_path = state_root / "app-1.json"
        state_path.write_text("{}")
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        adapter.approve_app_impl("TextEdit")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "textedit": {
                "app_name": "TextEdit",
                "approval_name": "TextEdit",
                "approval_bundle_id": "com.apple.TextEdit",
                "approval_bundle_path": adapter._normalize_bundle_path("/System/Applications/TextEdit.app"),
                "bundle_id": "com.apple.TextEdit",
                "bundle_path": adapter._normalize_bundle_path("/System/Applications/TextEdit.app"),
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "window_title": "Notes",
                "screenshot_path": "/tmp/shot.png",
                "accessibility_tree": [
                    {"index": 0, "role": "AXWindow", "title": "Notes", "children": []},
                ],
                "session_state_path": str(state_path),
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })

        revoked = adapter.revoke_app_impl("TextEdit")

        assert revoked["success"] is True
        listed = adapter.list_active_sessions_impl()
        assert listed["active_sessions"][0]["approved"] is False
        assert listed["active_sessions"][0]["accessibility_tree"] == []
        payload = json.loads(state_path.read_text())
        assert payload["accessibility_tree"] == []

    def test_revoke_app_clears_session_for_legacy_entry_missing_path(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        state_root.mkdir()
        state_path = state_root / "app-1.json"
        state_path.write_text("{}")
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: state_path.write_text(json.dumps(adapter._session_payload(session))) or str(state_path), raising=False)
        store_path.write_text(json.dumps({
            "approved_apps": ["Safari"],
            "approved_app_entries": [
                {"approval_name": "Safari", "bundle_id": "com.apple.Safari", "bundle_path": ""},
            ],
        }))
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "approval_name": "Safari",
                "approval_bundle_id": "com.apple.Safari",
                "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
                "bundle_id": "com.apple.Safari",
                "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "accessibility_tree": [{"index": 0, "role": "AXWindow", "title": "Docs", "children": []}],
                "session_state_path": str(state_path),
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })

        revoked = adapter.revoke_app_impl("Safari")

        assert revoked["success"] is True
        assert adapter._APP_SESSIONS["safari"]["approved"] is False
        payload = json.loads(state_path.read_text())
        assert payload["approved"] is False
        assert payload["accessibility_tree"] == []

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

    def test_get_app_state_writes_session_state_file(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_SESSION_ID", "session-test")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
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

        result = adapter.get_app_state_impl(app_name="Safari")

        state_path = Path(result["session_state_path"])
        assert state_path.exists() is True
        payload = json.loads(state_path.read_text())
        assert payload["session_id"] == "session-test"
        assert payload["app_session_id"] == result["app_session_id"]
        assert payload["app_name"] == "Safari"
        assert payload["active"] is True

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

    def test_get_app_state_preserves_pending_pointer_action_on_refresh(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "screenshot_path": "/tmp/old-shot.png",
                "pending_pointer_action": {
                    "action_id": "ptr-1",
                    "action_type": "click",
                    "x": 12,
                    "y": 34,
                    "button": "left",
                    "click_count": 1,
                },
                "virtual_cursor": {"x": 12, "y": 34, "detached": True, "visible": True},
            }
        })
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
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

        result = adapter.get_app_state_impl(app_session_id="app-1")

        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["pending_pointer_action"]["action_id"] == "ptr-1"
        assert payload["pending_pointer_action"]["action_type"] == "click"
        assert payload["pending_pointer_action"]["x"] == 12
        assert payload["pending_pointer_action"]["y"] == 34

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

    def test_list_active_sessions_exposes_overlay_screenshot_path(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "window_title": "Docs",
                "screenshot_path": "/tmp/shot.png",
                "overlay_screenshot_path": "/tmp/overlay.png",
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        listed = adapter.list_active_sessions_impl()

        assert listed["success"] is True
        assert listed["active_sessions"][0]["overlay_screenshot_path"] == "/tmp/overlay.png"
        assert listed["active_sessions"][0]["overlay_media_tag"] == "MEDIA:/tmp/overlay.png"

    def test_get_app_state_refreshes_overlay_preview_for_existing_cursor(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "screenshot_path": "/tmp/old-shot.png",
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })
        adapter.approve_app_impl("Safari")

        def fake_sync(session):
            session["overlay_screenshot_path"] = "/tmp/overlay-refresh.png"
            return "/tmp/overlay-refresh.png"

        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", fake_sync, raising=False)

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

        result = adapter.get_app_state_impl(app_session_id="app-1")

        assert result["success"] is True
        assert result["overlay_screenshot_path"] == "/tmp/overlay-refresh.png"
        assert result["overlay_media_tag"] == "MEDIA:/tmp/overlay-refresh.png"
        assert adapter._APP_SESSIONS["safari"]["overlay_screenshot_path"] == "/tmp/overlay-refresh.png"

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

    def test_stop_app_session_removes_overlay_preview_file(self, monkeypatch, tmp_path):
        managed_root = tmp_path / "managed-overlays"
        managed_root.mkdir()
        overlay_path = managed_root / "overlay.png"
        overlay_path.write_text("fake overlay")
        monkeypatch.setattr(adapter, "_overlay_root", lambda: managed_root)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "overlay_screenshot_path": str(overlay_path),
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["overlay_screenshot_path"] == ""
        assert stopped["overlay_media_tag"] == ""
        assert overlay_path.exists() is False

    def test_stop_app_session_removes_session_state_file(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        state_root.mkdir()
        state_path = state_root / "app-1.json"
        state_path.write_text('{"app_session_id": "app-1"}')
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "session_state_path": str(state_path),
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["session_state_path"] == ""
        assert state_path.exists() is False

    def test_stop_app_session_does_not_delete_unmanaged_overlay_path(self, monkeypatch, tmp_path):
        overlay_path = tmp_path / "outside-overlay.png"
        overlay_path.write_text("fake overlay")
        managed_root = tmp_path / "managed-overlays"
        managed_root.mkdir()
        monkeypatch.setattr(adapter, "_overlay_root", lambda: managed_root)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "overlay_screenshot_path": str(overlay_path),
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["overlay_screenshot_path"] == ""
        assert overlay_path.exists() is True

    def test_stop_app_session_does_not_delete_dotdot_escape_path(self, monkeypatch, tmp_path):
        managed_root = tmp_path / "managed-overlays"
        managed_root.mkdir()
        victim = tmp_path / "victim.txt"
        victim.write_text("keep me")
        escaped_path = managed_root / ".." / "victim.txt"
        monkeypatch.setattr(adapter, "_overlay_root", lambda: managed_root)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "overlay_screenshot_path": str(escaped_path),
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["overlay_screenshot_path"] == ""
        assert victim.exists() is True

    def test_stop_app_session_does_not_delete_unmanaged_symlink_to_managed_file(self, monkeypatch, tmp_path):
        managed_root = tmp_path / "managed-overlays"
        managed_root.mkdir()
        managed_overlay = managed_root / "overlay.png"
        managed_overlay.write_text("managed overlay")
        outside_link = tmp_path / "outside-overlay-link.png"
        outside_link.symlink_to(managed_overlay)
        monkeypatch.setattr(adapter, "_overlay_root", lambda: managed_root)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "overlay_screenshot_path": str(outside_link),
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["overlay_screenshot_path"] == ""
        assert outside_link.exists() is True
        assert managed_overlay.exists() is True

    def test_stop_app_session_handles_symlink_loop_overlay_path(self, monkeypatch, tmp_path):
        managed_root = tmp_path / "managed-overlays"
        managed_root.mkdir()
        loop_a = managed_root / "loop-a"
        loop_b = managed_root / "loop-b"
        loop_a.symlink_to(loop_b)
        loop_b.symlink_to(loop_a)
        monkeypatch.setattr(adapter, "_overlay_root", lambda: managed_root)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "overlay_screenshot_path": str(loop_a),
                "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
            }
        })

        stopped = adapter.stop_app_session_impl(app_name="Safari")

        assert stopped["success"] is True
        assert stopped["overlay_screenshot_path"] == ""


class TestOverlayPreviewRenderer:
    def test_pixel_cursor_draw_args_use_block_rectangles(self):
        draw_args = adapter._pixel_cursor_draw_args(10, 20)

        assert "-stroke" not in draw_args
        assert "#8b5cf6" in draw_args
        assert "rectangle 10,20 13,23" in draw_args
        assert any(isinstance(arg, str) and arg.startswith("rectangle ") for arg in draw_args)

    def test_sync_virtual_cursor_overlay_uses_pixel_cursor_draw_args(self, monkeypatch, tmp_path):
        screenshot_path = tmp_path / "shot.png"
        screenshot_path.write_text("fake image bytes")
        overlay_path = tmp_path / "overlay.png"
        seen = {}

        monkeypatch.setattr(adapter, "_overlay_preview_path", lambda session: overlay_path)
        monkeypatch.setattr(adapter, "_pixel_cursor_draw_args", lambda x, y: ["-fill", "#8b5cf6", "-draw", f"rectangle {x},{y} {x + 3},{y + 3}"])
        monkeypatch.setattr(adapter.shutil, "which", lambda name: "/opt/homebrew/bin/magick" if name == "magick" else None)

        def fake_run(cmd, check, capture_output, text):
            seen["cmd"] = cmd
            return None

        monkeypatch.setattr(adapter.subprocess, "run", fake_run)

        session = {
            "app_name": "Safari",
            "app_session_id": "app-1",
            "screenshot_path": str(screenshot_path),
            "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
        }

        result = adapter._sync_virtual_cursor_overlay(session)

        assert result == str(overlay_path)
        assert session["overlay_screenshot_path"] == str(overlay_path)
        assert seen["cmd"] == [
            "/opt/homebrew/bin/magick",
            str(screenshot_path),
            "-fill",
            "#8b5cf6",
            "-draw",
            "rectangle 11,22 14,25",
            str(overlay_path),
        ]

    def test_sync_virtual_cursor_overlay_removes_partial_file_on_render_failure(self, monkeypatch, tmp_path):
        screenshot_path = tmp_path / "shot.png"
        screenshot_path.write_text("fake image bytes")
        managed_root = tmp_path / "managed-overlays"
        managed_root.mkdir()
        overlay_path = managed_root / "overlay.png"

        monkeypatch.setattr(adapter, "_overlay_root", lambda: managed_root)
        monkeypatch.setattr(adapter, "_overlay_preview_path", lambda session: overlay_path)
        monkeypatch.setattr(adapter, "_pixel_cursor_draw_args", lambda x, y: ["-fill", "#8b5cf6", "-draw", f"rectangle {x},{y} {x + 3},{y + 3}"])
        monkeypatch.setattr(adapter.shutil, "which", lambda name: "/opt/homebrew/bin/magick" if name == "magick" else None)

        def fake_run(cmd, check, capture_output, text):
            overlay_path.write_text("partial overlay")
            raise RuntimeError("boom")

        monkeypatch.setattr(adapter.subprocess, "run", fake_run)

        session = {
            "app_name": "Safari",
            "app_session_id": "app-1",
            "screenshot_path": str(screenshot_path),
            "virtual_cursor": {"x": 11, "y": 22, "detached": True, "visible": True},
        }

        result = adapter._sync_virtual_cursor_overlay(session)

        assert result == ""
        assert session["overlay_screenshot_path"] == ""
        assert overlay_path.exists() is False


class TestKeyboardTools:
    def test_type_text_impl_requires_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})

        result = adapter.type_text_impl("uwu")

        assert result["success"] is False
        assert result["session_required"] is True

    def test_type_text_impl_maps_to_computer_control(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "approval_name": "Safari", "approval_bundle_id": "com.apple.Safari", "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "bundle_id": "com.apple.Safari", "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
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

    def test_type_text_impl_rejects_unapproved_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": False, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.type_text_impl("uwu")

        assert result["success"] is False
        assert result["approval_required"] is True
        assert result["approved"] is False
        assert result["app_session_id"] == "app-1"
        assert "not approved" in result["error"]

    def test_type_text_impl_requires_explicit_app_session_when_multiple_sessions_are_active(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
            "notes": {"app_name": "Notes", "app_session_id": "app-2", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.type_text_impl("uwu")

        assert result["success"] is False
        assert result["session_required"] is True
        assert result["multiple_active_sessions"] is True
        assert result["active_sessions"][0]["app_session_id"] == "app-2"
        assert result["active_sessions"][1]["app_session_id"] == "app-1"
        assert "app_session_id" in result["error"]

    def test_type_text_impl_can_target_specific_app_session(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")
        adapter.approve_app_impl("Notes")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "approval_name": "Safari", "approval_bundle_id": "com.apple.Safari", "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "bundle_id": "com.apple.Safari", "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
            "notes": {"app_name": "Notes", "approval_name": "Notes", "approval_bundle_id": "com.example.notes", "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Notes.app"), "bundle_id": "com.example.notes", "bundle_path": adapter._normalize_bundle_path("/Applications/Notes.app"), "app_session_id": "app-2", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
        })

        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.type_text_impl("uwu", app_session_id="app-2")

        assert result["success"] is True
        assert result["app_name"] == "Notes"
        assert result["app_session_id"] == "app-2"

    def test_type_text_impl_blocks_revoke_until_keystroke_dispatch_finishes(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "approval_name": "Safari",
                "approval_bundle_id": "com.apple.Safari",
                "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
                "bundle_id": "com.apple.Safari",
                "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"),
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })
        dispatch_started = threading.Event()
        allow_dispatch_finish = threading.Event()
        revoke_finished = threading.Event()
        calls = []

        def fake_cc(**kwargs):
            if kwargs.get("action") == "keystroke":
                dispatch_started.set()
                allow_dispatch_finish.wait(timeout=2)
                calls.append(kwargs)
                return '{"success": true}'
            raise AssertionError(kwargs)

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        typed_result: dict[str, object] = {}

        def run_type_text():
            typed_result["result"] = adapter.type_text_impl("uwu", app_session_id="app-1")

        def run_revoke():
            adapter.revoke_app_impl("Safari")
            revoke_finished.set()

        type_thread = threading.Thread(target=run_type_text)
        revoke_thread = threading.Thread(target=run_revoke)
        type_thread.start()
        assert dispatch_started.wait(timeout=1) is True
        revoke_thread.start()
        assert revoke_finished.wait(timeout=0.2) is False
        allow_dispatch_finish.set()
        type_thread.join(timeout=2)
        revoke_thread.join(timeout=2)

        assert typed_result["result"]["success"] is True
        assert calls == [{"action": "keystroke", "text": "uwu"}]
        assert revoke_finished.is_set() is True
        assert adapter._APP_SESSIONS["safari"]["approved"] is False

    def test_press_key_impl_requires_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {})

        result = adapter.press_key_impl("return", ["command"])

        assert result["success"] is False
        assert result["session_required"] is True

    def test_press_key_impl_maps_modifiers(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "approval_name": "Safari", "approval_bundle_id": "com.apple.Safari", "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "bundle_id": "com.apple.Safari", "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
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

    def test_press_key_impl_rejects_unapproved_target_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": True, "approved": False, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.press_key_impl("return", ["command"], app_session_id="app-1")

        assert result["success"] is False
        assert result["approval_required"] is True
        assert result["approved"] is False
        assert result["app_session_id"] == "app-1"
        assert "not approved" in result["error"]

    def test_press_key_impl_rejects_inactive_target_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "app_session_id": "app-1", "active": False, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}}
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: '{"success": true}')

        result = adapter.press_key_impl("return", ["command"], app_session_id="app-1")

        assert result["success"] is False
        assert result["session_required"] is True

    def test_press_key_impl_can_target_specific_app_session(self, monkeypatch, tmp_path):
        store_path = tmp_path / "ComputerUseAppApprovals.json"
        monkeypatch.setattr(adapter, "_approval_store_path", lambda: store_path)
        adapter.approve_app_impl("Safari")
        adapter.approve_app_impl("Notes")
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {"app_name": "Safari", "approval_name": "Safari", "approval_bundle_id": "com.apple.Safari", "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "bundle_id": "com.apple.Safari", "bundle_path": adapter._normalize_bundle_path("/Applications/Safari.app"), "app_session_id": "app-1", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
            "notes": {"app_name": "Notes", "approval_name": "Notes", "approval_bundle_id": "com.example.notes", "approval_bundle_path": adapter._normalize_bundle_path("/Applications/Notes.app"), "bundle_id": "com.example.notes", "bundle_path": adapter._normalize_bundle_path("/Applications/Notes.app"), "app_session_id": "app-2", "active": True, "approved": True, "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True}},
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

    def test_click_executes_real_pointer_backend_for_active_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })
        calls = []

        def fake_cc(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "action": "click", "x": 10, "y": 20, "button": "left", "click_count": 1})

        monkeypatch.setattr(adapter, "computer_control", fake_cc)
        monkeypatch.setattr(adapter, "_find_approval_entry", lambda _name: {"approval_name": "Safari"})
        monkeypatch.setattr(adapter, "_session_matches_approval_entry", lambda _session, _entry: True)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        result = adapter.click_impl(x=10, y=20)

        assert result["success"] is True
        assert result["clicked"] is True
        assert result["app_session_id"] == "app-1"
        assert result["virtual_cursor"] == {"x": 10, "y": 20, "detached": True, "visible": True}
        assert result["pending_pointer_action"] is None
        assert result["last_pointer_action_result"]["action_type"] == "click"
        assert result["last_pointer_action_result"]["status"] == "completed"
        assert calls == [{"action": "click", "x": 10, "y": 20, "button": "left", "click_count": 1}]

    def test_click_rechecks_session_activity_before_queueing_pending_pointer_action(self, monkeypatch):
        session = {
            "app_name": "Safari",
            "app_session_id": "app-1",
            "active": True,
            "approved": True,
            "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
        }

        def stale_resolve_session(app_session_id=None):
            session["active"] = False
            return session

        monkeypatch.setattr(adapter, "_resolve_session", stale_resolve_session, raising=False)

        result = adapter.click_impl(x=10, y=20, app_session_id="app-1")

        assert result["success"] is False
        assert result["session_required"] is True
        assert session.get("pending_pointer_action") is None
        assert session["virtual_cursor"] == {"x": None, "y": None, "detached": True, "visible": True}

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
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: json.dumps({"success": True, **kwargs}))
        monkeypatch.setattr(adapter, "_find_approval_entry", lambda _name: {"approval_name": "Notes"})
        monkeypatch.setattr(adapter, "_session_matches_approval_entry", lambda _session, _entry: True)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        result = adapter.click_impl(x=50, y=60, app_session_id="app-2")

        assert result["app_name"] == "Notes"
        assert result["app_session_id"] == "app-2"
        assert result["virtual_cursor"] == {"x": 50, "y": 60, "detached": True, "visible": True}
        assert adapter._APP_SESSIONS["safari"]["virtual_cursor"] == {"x": None, "y": None, "detached": True, "visible": True}

    def test_click_updates_session_state_file_for_target_session(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            },
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: json.dumps({"success": True, **kwargs}))
        monkeypatch.setattr(adapter, "_find_approval_entry", lambda _name: {"approval_name": "Notes"})
        monkeypatch.setattr(adapter, "_session_matches_approval_entry", lambda _session, _entry: True)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        result = adapter.click_impl(x=50, y=60, app_session_id="app-2")

        state_path = Path(result["session_state_path"])
        assert state_path.exists() is True
        payload = json.loads(state_path.read_text())
        assert payload["app_session_id"] == "app-2"
        assert payload["virtual_cursor"] == {"x": 50, "y": 60, "detached": True, "visible": True}
        assert payload["pending_pointer_action"] is None
        assert payload["last_pointer_action_result"]["action_type"] == "click"
        assert payload["last_pointer_action_result"]["status"] == "completed"
        assert payload["last_pointer_action_result"]["x"] == 50
        assert payload["last_pointer_action_result"]["y"] == 60

    def test_click_requires_approved_session(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": False,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            }
        })

        result = adapter.click_impl(x=12, y=34, app_session_id="app-2")

        assert result["success"] is False
        assert result["approval_required"] is True
        assert result["approved"] is False

    def test_scroll_updates_pending_pointer_action_in_session_state(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            },
        })
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        result = adapter.scroll_impl(x=7, y=8, delta_y=240, app_session_id="app-2")

        state_path = Path(result["session_state_path"])
        assert state_path.exists() is True
        payload = json.loads(state_path.read_text())
        assert payload["pending_pointer_action"]["action_type"] == "scroll"
        assert payload["pending_pointer_action"]["x"] == 7
        assert payload["pending_pointer_action"]["y"] == 8
        assert payload["pending_pointer_action"]["delta_y"] == 240

    def test_click_rejects_overwriting_existing_pending_pointer_action(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 10, "y": 20, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 10,
                    "y": 20,
                },
            },
        })
        monkeypatch.setattr(adapter, "_find_approval_entry", lambda _name: {"approval_name": "Notes"})
        monkeypatch.setattr(adapter, "_session_matches_approval_entry", lambda _session, _entry: True)

        result = adapter.click_impl(x=50, y=60, app_session_id="app-2")

        assert result["success"] is False
        assert result["action_pending"] is True
        assert result["pending_pointer_action"]["action_id"] == "ptr-123"
        assert result["virtual_cursor"] == {"x": 10, "y": 20, "detached": True, "visible": True}
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["action_id"] == "ptr-123"
        assert adapter._APP_SESSIONS["notes"]["virtual_cursor"] == {"x": 10, "y": 20, "detached": True, "visible": True}

    def test_new_pointer_action_is_allowed_after_previous_pending_action_is_resolved(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 10, "y": 20, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 10,
                    "y": 20,
                },
            },
        })
        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: json.dumps({"success": True, **kwargs}))
        monkeypatch.setattr(adapter, "_find_approval_entry", lambda _name: {"approval_name": "Notes"})
        monkeypatch.setattr(adapter, "_session_matches_approval_entry", lambda _session, _entry: True)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        cleared = adapter.report_pointer_action_result_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            status="completed",
            x=10,
            y=20,
        )
        result = adapter.click_impl(x=50, y=60, app_session_id="app-2")

        assert cleared["success"] is True
        assert result["success"] is True
        assert result["clicked"] is True
        assert result["pending_pointer_action"] is None
        assert result["last_pointer_action_result"]["action_type"] == "click"
        assert result["last_pointer_action_result"]["x"] == 50
        assert result["last_pointer_action_result"]["y"] == 60

    def test_drag_updates_virtual_cursor_to_end_position(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 10, "y": 20, "detached": True, "visible": True},
            }
        })
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        result = adapter.drag_impl(start_x=10, start_y=20, end_x=30, end_y=40)

        assert result["success"] is False
        assert result["preview_only"] is True
        assert result["virtual_cursor"] == {"x": 30, "y": 40, "detached": True, "visible": True}
        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["pending_pointer_action"]["action_type"] == "drag"
        assert payload["pending_pointer_action"]["start_x"] == 10
        assert payload["pending_pointer_action"]["start_y"] == 20
        assert payload["pending_pointer_action"]["end_x"] == 30
        assert payload["pending_pointer_action"]["end_y"] == 40
    def test_report_pointer_action_result_clears_pending_action_and_records_completion(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                    "button": "left",
                    "click_count": 1,
                },
            },
        })

        claim = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            worker_id="helper-a",
        )
        result = adapter.report_pointer_action_result_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            status="completed",
            claim_token=claim["claim_token"],
            x=55,
            y=65,
        )

        assert result["success"] is True
        assert result["pending_pointer_action"] is None
        assert result["virtual_cursor"] == {"x": 55, "y": 65, "detached": True, "visible": True}
        assert result["last_pointer_action_result"]["action_id"] == "ptr-123"
        assert result["last_pointer_action_result"]["action_type"] == "click"
        assert result["last_pointer_action_result"]["status"] == "completed"
        assert result["last_pointer_action_result"]["reported_by"] == "helper-a"
        assert result["last_pointer_action_result"]["x"] == 55
        assert result["last_pointer_action_result"]["y"] == 65
        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["pending_pointer_action"] is None
        assert payload["last_pointer_action_result"]["action_id"] == "ptr-123"
        assert payload["last_pointer_action_result"]["status"] == "completed"
        assert payload["last_pointer_action_result"]["reported_by"] == "helper-a"

    def test_report_pointer_action_result_rejects_wrong_claim_token_for_claimed_action(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                    "claimed_by": "helper-a",
                    "claimed_at": "2026-04-19T16:55:02+00:00",
                },
                "pending_pointer_claim_token": "claim-secret",
            },
        })

        result = adapter.report_pointer_action_result_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            status="completed",
            claim_token="claim-wrong",
        )

        assert result["success"] is False
        assert result["action_claimed"] is True
        assert result["claimed_by"] == "helper-a"
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["action_id"] == "ptr-123"

    def test_report_pointer_action_result_rejects_claimed_action_when_server_token_is_missing(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                    "claimed_by": "helper-a",
                    "claimed_at": "2026-04-19T16:55:02+00:00",
                },
                "pending_pointer_claim_token": "",
            },
        })

        result = adapter.report_pointer_action_result_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            status="completed",
        )

        assert result["success"] is False
        assert result["action_claimed"] is True
        assert "valid claim_token is required" in result["error"]

    def test_report_pointer_action_result_rejects_expired_claim_token(self, monkeypatch):
        monkeypatch.setattr(adapter, "_utc_now", lambda: adapter.datetime(2026, 4, 19, 17, 0, 0, tzinfo=adapter.timezone.utc), raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                    "claimed_by": "helper-a",
                    "claimed_at": "2026-04-19T16:55:02+00:00",
                    "claim_expires_at": "2026-04-19T16:56:02+00:00",
                },
                "pending_pointer_claim_token": "claim-secret",
            },
        })

        result = adapter.report_pointer_action_result_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            status="completed",
            claim_token="claim-secret",
        )

        assert result["success"] is False
        assert result["claim_expired"] is True
        assert "claim has expired" in result["error"]
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["action_id"] == "ptr-123"

    def test_report_pointer_action_result_rejects_mismatched_action_id(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        result = adapter.report_pointer_action_result_impl(
            app_session_id="app-2",
            action_id="ptr-wrong",
            status="completed",
        )

        assert result["success"] is False
        assert result["action_mismatch"] is True
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["action_id"] == "ptr-123"

    def test_report_pointer_action_result_rejects_blank_app_session_id(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        result = adapter.report_pointer_action_result_impl(
            app_session_id="   ",
            action_id="ptr-123",
            status="completed",
        )

        assert result["success"] is False
        assert result["app_session_required"] is True
        assert "app_session_id is required" in result["error"]
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["action_id"] == "ptr-123"

    def test_list_pending_pointer_actions_returns_only_active_pending_sessions(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
            "safari": {
                "app_name": "Safari",
                "app_session_id": "app-1",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": None, "y": None, "detached": True, "visible": True},
            },
            "finder": {
                "app_name": "Finder",
                "app_session_id": "app-3",
                "active": False,
                "approved": True,
                "virtual_cursor": {"x": 1, "y": 2, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-inactive",
                    "action_type": "scroll",
                    "x": 1,
                    "y": 2,
                },
            },
        })

        result = adapter.list_pending_pointer_actions_impl()

        assert result["success"] is True
        assert len(result["pending_actions"]) == 1
        assert result["pending_actions"][0]["app_session_id"] == "app-2"
        assert result["pending_actions"][0]["pending_pointer_action"]["action_id"] == "ptr-123"

    def test_claim_pending_pointer_action_marks_pending_action_and_persists_state(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            worker_id="helper-a",
        )

        assert result["success"] is True
        assert result["claimed"] is True
        assert result["claim_token"].startswith("claim-")
        assert result["pending_pointer_action"]["claimed_by"] == "helper-a"
        assert result["pending_pointer_action"]["claimed_at"]
        assert result["pending_pointer_action"]["claim_expires_at"]
        state_path = Path(result["session_state_path"])
        payload = json.loads(state_path.read_text())
        assert payload["pending_pointer_action"]["action_id"] == "ptr-123"
        assert payload["pending_pointer_action"]["claimed_by"] == "helper-a"
        assert payload["pending_pointer_action"]["claimed_at"]
        assert payload["pending_pointer_action"]["claim_expires_at"]

    def test_claim_pending_pointer_action_rejects_reclaim_while_claim_is_unexpired(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                    "claimed_by": "helper-a",
                    "claimed_at": "2026-04-19T16:55:02+00:00",
                    "claim_expires_at": "2099-04-19T16:56:02+00:00",
                },
                "pending_pointer_claim_token": "claim-old",
            },
        })

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            worker_id="helper-a",
        )

        assert result["success"] is False
        assert result["action_claimed"] is True
        assert result["claimed_by"] == "helper-a"
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_claim_token"] == "claim-old"

    def test_claim_pending_pointer_action_is_atomic_across_competing_helpers(self, monkeypatch):
        barrier = threading.Barrier(2)
        monkeypatch.setattr(adapter, "_sync_session_artifacts", lambda session: "", raising=False)

        def fake_utc_now():
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
            return adapter.datetime(2026, 4, 19, 17, 0, 0, tzinfo=adapter.timezone.utc)

        monkeypatch.setattr(adapter, "_utc_now", fake_utc_now, raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        results = []
        results_lock = threading.Lock()

        def claim(worker_id: str) -> None:
            payload = adapter.claim_pending_pointer_action_impl(
                app_session_id="app-2",
                action_id="ptr-123",
                worker_id=worker_id,
            )
            with results_lock:
                results.append(payload)

        threads = [
            threading.Thread(target=claim, args=("helper-a",)),
            threading.Thread(target=claim, args=("helper-b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 2
        successes = [payload for payload in results if payload["success"] is True]
        failures = [payload for payload in results if payload["success"] is False]
        assert len(successes) == 1
        assert len(failures) == 1
        assert failures[0]["action_claimed"] is True
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["claimed_by"] == successes[0]["pending_pointer_action"]["claimed_by"]
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_claim_token"] == successes[0]["claim_token"]

    def test_claim_pending_pointer_action_allows_reclaim_after_expiry(self, monkeypatch, tmp_path):
        state_root = tmp_path / "session-state"
        monkeypatch.setattr(adapter, "_session_state_root", lambda: state_root, raising=False)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)
        monkeypatch.setattr(adapter, "_utc_now", lambda: adapter.datetime(2026, 4, 19, 17, 0, 0, tzinfo=adapter.timezone.utc), raising=False)
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                    "claimed_by": "helper-a",
                    "claimed_at": "2026-04-19T16:50:00+00:00",
                    "claim_expires_at": "2026-04-19T16:51:00+00:00",
                },
                "pending_pointer_claim_token": "claim-old",
            },
        })

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            worker_id="helper-b",
        )

        assert result["success"] is True
        assert result["claim_token"].startswith("claim-")
        assert result["claim_token"] != "claim-old"
        assert result["pending_pointer_action"]["claimed_by"] == "helper-b"
        assert result["pending_pointer_action"]["claim_expires_at"] == "2026-04-19T17:01:00+00:00"

    def test_claim_pending_pointer_action_rejects_mismatched_action_id(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-wrong",
            worker_id="helper-a",
        )

        assert result["success"] is False
        assert result["action_mismatch"] is True
        assert adapter._APP_SESSIONS["notes"]["pending_pointer_action"]["action_id"] == "ptr-123"

    def test_claim_pending_pointer_action_rejects_blank_app_session_id(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="   ",
            action_id="ptr-123",
        )

        assert result["success"] is False
        assert result["app_session_required"] is True
        assert "app_session_id is required" in result["error"]

    def test_claim_pending_pointer_action_rechecks_session_activity_before_mutation(self, monkeypatch):
        session = {
            "app_name": "Notes",
            "app_session_id": "app-2",
            "active": True,
            "approved": True,
            "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
            "pending_pointer_action": {
                "action_id": "ptr-123",
                "action_type": "click",
                "x": 50,
                "y": 60,
            },
        }

        def stale_resolve_session(app_session_id=None):
            session["active"] = False
            return session

        monkeypatch.setattr(adapter, "_resolve_session", stale_resolve_session, raising=False)

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            worker_id="helper-a",
        )

        assert result["success"] is False
        assert result["session_required"] is True
        assert session["pending_pointer_action"]["action_id"] == "ptr-123"
        assert "claimed_by" not in session["pending_pointer_action"]
        assert session.get("pending_pointer_claim_token") in {None, ""}

    def test_claim_pending_pointer_action_requires_worker_id(self, monkeypatch):
        monkeypatch.setattr(adapter, "_APP_SESSIONS", {
            "notes": {
                "app_name": "Notes",
                "app_session_id": "app-2",
                "active": True,
                "approved": True,
                "virtual_cursor": {"x": 50, "y": 60, "detached": True, "visible": True},
                "pending_pointer_action": {
                    "action_id": "ptr-123",
                    "action_type": "click",
                    "x": 50,
                    "y": 60,
                },
            },
        })

        result = adapter.claim_pending_pointer_action_impl(
            app_session_id="app-2",
            action_id="ptr-123",
            worker_id="   ",
        )

        assert result["success"] is False
        assert result["worker_required"] is True
        assert "worker_id is required" in result["error"]

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

        monkeypatch.setattr(adapter, "computer_control", lambda **kwargs: json.dumps({"success": True, **kwargs}))
        monkeypatch.setattr(adapter, "_find_approval_entry", lambda _name: {"approval_name": "Safari"})
        monkeypatch.setattr(adapter, "_session_matches_approval_entry", lambda _session, _entry: True)
        monkeypatch.setattr(adapter, "_sync_virtual_cursor_overlay", lambda session: "", raising=False)

        result = adapter.click_impl(x=10, y=20)
        assert result["success"] is True
        assert result["clicked"] is True
        assert result["virtual_cursor"] == {"x": 10, "y": 20, "detached": True, "visible": True}

    def test_set_value_stub_is_explicit(self):
        result = adapter.set_value_impl(index=1, value="hello")
        assert result["success"] is False
        assert result["supported"] is False
        assert "not implemented" in result["error"].lower()
