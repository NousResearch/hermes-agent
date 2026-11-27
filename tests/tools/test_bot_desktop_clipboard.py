"""Copy-out reads the screen's X CLIPBOARD with xclip on the screen's own display, capped like paste-in."""

from __future__ import annotations

import subprocess

from tools.bot_desktop import clipboard, runtime


def _host_screen(monkeypatch, *, which="/usr/bin/xclip"):
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":23", "XAUTHORITY": "/x/auth", "DBUS": "no"})
    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: which)


def test_reads_the_clipboard_on_the_screens_display(monkeypatch):
    _host_screen(monkeypatch)
    seen = {}

    def run(argv, env, **kw):
        seen.update(argv=argv, env=env)
        return subprocess.CompletedProcess(argv, 0, "héllo".encode(), b"")

    monkeypatch.setattr(clipboard.subprocess, "run", run)
    assert clipboard.read_text() == clipboard.ClipboardRead("héllo")
    assert seen["argv"] == ["xclip", "-o", "-selection", "clipboard"]
    assert seen["env"]["DISPLAY"] == ":23" and seen["env"]["XAUTHORITY"] == "/x/auth" and "DBUS" not in seen["env"]


def test_nothing_owning_the_clipboard_is_empty(monkeypatch):
    _host_screen(monkeypatch)
    monkeypatch.setattr(clipboard.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, b"", b"Error: target STRING not available"))
    assert clipboard.read_text() == clipboard.ClipboardRead(None, "empty")


def test_over_the_cap_is_refused_not_truncated(monkeypatch):
    _host_screen(monkeypatch)
    big = b"a" * (clipboard.MAX_TEXT + 1)
    monkeypatch.setattr(clipboard.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, big, b""))
    assert clipboard.read_text() == clipboard.ClipboardRead(None, "too_large")


def test_without_xclip_or_a_screen_it_says_why(monkeypatch):
    _host_screen(monkeypatch, which=None)
    assert clipboard.read_text().reason == "unavailable"
    monkeypatch.setattr(runtime, "published_env", lambda: {})
    assert clipboard.read_text().reason == "not_running"


def test_a_hung_clipboard_owner_does_not_hang_the_gateway(monkeypatch):
    _host_screen(monkeypatch)

    def hang(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(clipboard.subprocess, "run", hang)
    assert clipboard.read_text().reason == "unavailable"
