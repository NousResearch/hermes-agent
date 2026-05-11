import json
import subprocess
from types import SimpleNamespace

from tools import computer_use_tool as tool


def test_desktop_status_handles_timeouts(monkeypatch):
    def fake_osascript(script, timeout=10):
        raise subprocess.TimeoutExpired(["osascript"], timeout)

    def fake_run(cmd, timeout=20):
        return SimpleNamespace(returncode=0, stdout="12,34\n", stderr="")

    monkeypatch.setattr(tool, "_osascript", fake_osascript)
    monkeypatch.setattr(tool, "_run", fake_run)
    monkeypatch.setattr(tool, "_cliclick", lambda: "/usr/local/bin/cliclick")
    monkeypatch.setattr(tool.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = json.loads(tool._desktop_status({}))
    assert result["success"] is True
    assert result["mouse_position"] == "12,34"
    assert "osascript timed out" in " ".join(result["status_errors"])


def test_desktop_action_click_builds_cliclick_command(monkeypatch):
    calls = []

    def fake_run(cmd, timeout=20):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool, "_cliclick", lambda: "/opt/homebrew/bin/cliclick")
    monkeypatch.setattr(tool, "_run", fake_run)

    result = json.loads(tool._desktop_action({"action": "click", "x": 10, "y": 20}))
    assert result["success"] is True
    assert calls[-1] == ["/opt/homebrew/bin/cliclick", "c:10,20"]


def test_desktop_action_hotkey_builds_modifier_sequence(monkeypatch):
    calls = []

    def fake_run(cmd, timeout=20):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool, "_cliclick", lambda: "/opt/homebrew/bin/cliclick")
    monkeypatch.setattr(tool, "_run", fake_run)

    result = json.loads(tool._desktop_action({"action": "hotkey", "modifiers": ["cmd", "shift"], "key": "4"}))
    assert result["success"] is True
    assert calls[-1] == ["/opt/homebrew/bin/cliclick", "kd:cmd,shift", "kp:4", "ku:cmd,shift"]


def test_desktop_screenshot_returns_error_when_screencapture_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(tool.shutil, "which", lambda name: "/usr/sbin/screencapture" if name == "screencapture" else None)
    monkeypatch.setattr(tool, "_run", lambda cmd, timeout=20: SimpleNamespace(returncode=1, stdout="", stderr="no permission"))

    result = json.loads(tool._desktop_screenshot({"path": str(tmp_path / "shot.png")}))
    assert result["error"] == "no permission"
