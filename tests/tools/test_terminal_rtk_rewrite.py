from __future__ import annotations

from types import SimpleNamespace

from tools.terminal_tool import _apply_rtk_rewrite


class _CompletedProcess:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def test_rtk_off_keeps_raw_command(monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be called when RTK is off")

    monkeypatch.setattr("tools.terminal_tool.subprocess.run", fake_run)
    monkeypatch.setattr("tools.terminal_tool.shutil.which", lambda name: "/usr/bin/rtk")

    command, error, used_rtk = _apply_rtk_rewrite("git status", "off", "rtk")

    assert command == "git status"
    assert error is None
    assert used_rtk is False
    assert called is False


def test_rtk_auto_rewrites_simple_command(monkeypatch):
    monkeypatch.setattr("tools.terminal_tool.shutil.which", lambda name: "/usr/bin/rtk")
    monkeypatch.setattr(
        "tools.terminal_tool.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(0, "git status --short"),
    )

    command, error, used_rtk = _apply_rtk_rewrite("git status", "auto", "rtk")

    assert command == "git status --short"
    assert error is None
    assert used_rtk is True


def test_rtk_auto_falls_back_for_complex_commands(monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be called for compound commands")

    monkeypatch.setattr("tools.terminal_tool.subprocess.run", fake_run)
    monkeypatch.setattr("tools.terminal_tool.shutil.which", lambda name: "/usr/bin/rtk")

    command, error, used_rtk = _apply_rtk_rewrite("git status && git diff", "auto", "rtk")

    assert command == "git status && git diff"
    assert error is None
    assert used_rtk is False
    assert called is False


def test_rtk_force_requires_binary(monkeypatch):
    monkeypatch.setattr("tools.terminal_tool.shutil.which", lambda name: None)

    command, error, used_rtk = _apply_rtk_rewrite("git status", "force", "")

    assert command == "git status"
    assert used_rtk is False
    assert error == "RTK is enabled in force mode, but the `rtk` binary was not found."
