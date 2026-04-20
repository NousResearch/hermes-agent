"""Tests for the macOS computer-control tool."""

import json
from pathlib import Path

import pytest

import tools.computer_control_tool as cct
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS, resolve_toolset


class TestAvailability:
    def test_check_available_false_off_macos(self, monkeypatch):
        monkeypatch.setattr(cct.sys, "platform", "linux")
        assert cct._check_computer_control_available() is False

    def test_check_available_requires_builtins(self, monkeypatch):
        monkeypatch.setattr(cct.sys, "platform", "darwin")
        monkeypatch.setattr(cct.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cct._check_computer_control_available() is True

        monkeypatch.setattr(
            cct.shutil,
            "which",
            lambda name: None if name == "screencapture" else f"/usr/bin/{name}",
        )
        assert cct._check_computer_control_available() is False


class TestHandlers:
    def test_screenshot_returns_media_tag(self, monkeypatch):
        path = Path("/tmp/hermes-desktop-shot.png")
        seen = {}

        monkeypatch.setattr(cct, "_default_screenshot_path", lambda: path)
        monkeypatch.setattr(cct, "_run_command", lambda cmd: seen.setdefault("cmd", cmd) or "")

        result = json.loads(cct._handle_computer_control({"action": "screenshot"}))

        assert result["success"] is True
        assert result["path"] == str(path)
        assert result["media_tag"] == f"MEDIA:{path}"
        assert seen["cmd"] == ["screencapture", "-x", str(path)]

    def test_screenshot_can_target_specific_window_id(self, monkeypatch):
        path = Path("/tmp/hermes-window-shot.png")
        seen = {}

        monkeypatch.setattr(cct, "_default_screenshot_path", lambda: path)
        monkeypatch.setattr(cct, "_run_command", lambda cmd: seen.setdefault("cmd", cmd) or "")

        result = json.loads(cct._handle_computer_control({"action": "screenshot", "window_id": 321}))

        assert result["success"] is True
        assert result["path"] == str(path)
        assert result["window_id"] == 321
        assert seen["cmd"] == ["screencapture", "-x", "-o", "-l321", str(path)]

    def test_activate_app_uses_osascript(self, monkeypatch):
        scripts = []
        monkeypatch.setattr(cct, "_run_osascript", lambda script: scripts.append(script) or "")

        result = json.loads(cct._handle_computer_control({"action": "activate_app", "app_name": "Safari"}))

        assert result["success"] is True
        assert 'tell application "Safari" to activate' in scripts[0]

    def test_open_uses_open_command(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cct, "_run_command", lambda cmd: seen.setdefault("cmd", cmd) or "")

        result = json.loads(cct._handle_computer_control({"action": "open", "target": "https://example.com"}))

        assert result["success"] is True
        assert seen["cmd"] == ["open", "https://example.com"]

    def test_keystroke_special_key_with_modifiers(self, monkeypatch):
        scripts = []
        monkeypatch.setattr(cct, "_run_osascript", lambda script: scripts.append(script) or "")

        result = json.loads(
            cct._handle_computer_control(
                {"action": "keystroke", "key": "return", "modifiers": ["command", "shift"]}
            )
        )

        assert result["success"] is True
        assert "key code 36 using {command down, shift down}" in scripts[0]

    def test_frontmost_app_parses_result(self, monkeypatch):
        monkeypatch.setattr(cct, "_frontmost_window_info", lambda: {
            "app_name": "Finder",
            "bundle_id": "com.apple.finder",
            "bundle_name": "Finder",
            "process_id": 111,
            "window_title": "Downloads",
            "window_id": 222,
            "window_bounds": {"x": 10, "y": 20, "width": 300, "height": 200},
        })

        result = json.loads(cct._handle_computer_control({"action": "frontmost_app"}))

        assert result["success"] is True
        assert result["app_name"] == "Finder"
        assert result["bundle_id"] == "com.apple.finder"
        assert result["bundle_name"] == "Finder"
        assert result["process_id"] == 111
        assert result["window_title"] == "Downloads"
        assert result["window_id"] == 222
        assert result["window_bounds"] == {"x": 10, "y": 20, "width": 300, "height": 200}

    def test_frontmost_window_info_prefers_large_titled_window_from_helper(self, monkeypatch):
        helper_payload = {
            "app_name": "文本编辑",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "process_id": 111,
            "windows": [
                {"window_title": "", "window_id": 226068, "window_bounds": {"x": 769, "y": 183, "width": 79, "height": 22}},
                {"window_title": "Notes", "window_id": 226062, "window_bounds": {"x": 834, "y": 40, "width": 673, "height": 439}},
            ],
        }

        monkeypatch.setattr(cct, "_ensure_window_helper_binary", lambda: Path("/tmp/mac-window-info"))
        monkeypatch.setattr(cct, "_run_command", lambda cmd: json.dumps(helper_payload, ensure_ascii=False))

        result = cct._frontmost_window_info()

        assert result["app_name"] == "文本编辑"
        assert result["bundle_id"] == "com.apple.TextEdit"
        assert result["window_title"] == "Notes"
        assert result["window_id"] == 226062
        assert result["window_bounds"] == {"x": 834, "y": 40, "width": 673, "height": 439}

    def test_frontmost_window_info_keeps_first_large_window_even_if_later_window_has_title(self, monkeypatch):
        helper_payload = {
            "app_name": "文本编辑",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "process_id": 111,
            "windows": [
                {"window_title": "", "window_id": 226810, "window_bounds": {"x": 833, "y": 264, "width": 673, "height": 439}},
                {"window_title": "Background Notes", "window_id": 226062, "window_bounds": {"x": 834, "y": 40, "width": 640, "height": 420}},
            ],
        }

        monkeypatch.setattr(cct, "_ensure_window_helper_binary", lambda: Path("/tmp/mac-window-info"))
        monkeypatch.setattr(cct, "_run_command", lambda cmd: json.dumps(helper_payload, ensure_ascii=False))

        result = cct._frontmost_window_info()

        assert result["window_title"] == ""
        assert result["window_id"] == 226810
        assert result["window_bounds"] == {"x": 833, "y": 264, "width": 673, "height": 439}

    def test_frontmost_window_info_keeps_smaller_frontmost_dialog_instead_of_larger_background_window(self, monkeypatch):
        helper_payload = {
            "app_name": "文本编辑",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "process_id": 111,
            "windows": [
                {"window_title": "Save", "window_id": 300001, "window_bounds": {"x": 900, "y": 220, "width": 320, "height": 180}},
                {"window_title": "Document", "window_id": 300002, "window_bounds": {"x": 833, "y": 264, "width": 673, "height": 439}},
            ],
        }

        monkeypatch.setattr(cct, "_ensure_window_helper_binary", lambda: Path("/tmp/mac-window-info"))
        monkeypatch.setattr(cct, "_run_command", lambda cmd: json.dumps(helper_payload, ensure_ascii=False))

        result = cct._frontmost_window_info()

        assert result["window_title"] == "Save"
        assert result["window_id"] == 300001
        assert result["window_bounds"] == {"x": 900, "y": 220, "width": 320, "height": 180}

    def test_frontmost_window_info_keeps_compact_titled_frontmost_window(self, monkeypatch):
        helper_payload = {
            "app_name": "文本编辑",
            "bundle_id": "com.apple.TextEdit",
            "bundle_name": "TextEdit",
            "process_id": 111,
            "windows": [
                {"window_title": "Quick Find", "window_id": 300101, "window_bounds": {"x": 920, "y": 200, "width": 150, "height": 60}},
                {"window_title": "Document", "window_id": 300102, "window_bounds": {"x": 833, "y": 264, "width": 673, "height": 439}},
            ],
        }

        monkeypatch.setattr(cct, "_ensure_window_helper_binary", lambda: Path("/tmp/mac-window-info"))
        monkeypatch.setattr(cct, "_run_command", lambda cmd: json.dumps(helper_payload, ensure_ascii=False))

        result = cct._frontmost_window_info()

        assert result["window_title"] == "Quick Find"
        assert result["window_id"] == 300101
        assert result["window_bounds"] == {"x": 920, "y": 200, "width": 150, "height": 60}

    def test_keystroke_requires_text_or_key(self):
        result = json.loads(cct._handle_computer_control({"action": "keystroke"}))
        assert "error" in result
        assert "text or key" in result["error"]

    def test_invalid_action_is_rejected(self):
        result = json.loads(cct._handle_computer_control({"action": "teleport"}))
        assert "error" in result
        assert "Unknown action" in result["error"]

    def test_screenshot_failure_is_humanized_and_cleans_zero_byte_file(self, monkeypatch, tmp_path):
        path = tmp_path / "shot.png"
        path.write_bytes(b"")

        def _boom(cmd):
            raise cct.subprocess.CalledProcessError(1, cmd, stderr="could not create image from display")

        monkeypatch.setattr(cct, "_default_screenshot_path", lambda: path)
        monkeypatch.setattr(cct, "_run_command", _boom)

        result = json.loads(cct._handle_computer_control({"action": "screenshot"}))

        assert "error" in result
        assert "Screen Recording permission" in result["error"] or "display" in result["error"]
        assert not path.exists()


class TestToolsetRegistration:
    def test_toolset_contains_computer_control(self):
        assert "computer" in TOOLSETS
        assert "computer_control" in TOOLSETS["computer"]["tools"]
        assert "computer_control" in _HERMES_CORE_TOOLS
        assert "computer_control" in resolve_toolset("hermes-telegram")
