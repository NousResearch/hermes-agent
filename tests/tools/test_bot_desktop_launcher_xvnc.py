"""The Bot Desktop's Xvnc is started with the RFB options the lease design relies on."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def test_xvnc_never_sends_the_holders_clipboard_to_watchers(tmp_path):
    """Whoever holds control may paste INTO the screen (AcceptCutText), but the screen's clipboard must
    not be pushed to every connected viewer (SendCutText off): watchers are not the holder."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "xvnc-argv"
    (bindir / "Xvnc").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_log}"\nexec sleep 3\n', encoding="utf-8")
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "dbus-run-session", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "grep", "seq", "sleep", "kill"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env = {
        "PATH": str(bindir), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    argv = argv_log.read_text(encoding="utf-8-sig").split("\n")
    assert "-SendCutText=0" in argv, argv
    assert not any(a.startswith("-AcceptCutText") for a in argv), "paste into the screen must keep working"
    # Xvnc's own cut-text cap and the bridge filter's must agree, or one side drops a paste the other admits.
    from tools.bot_desktop.rfb_filter import _MAX_CUT_TEXT
    assert int(argv[argv.index("-MaxCutText") + 1]) == _MAX_CUT_TEXT, argv
    assert not os.path.exists(tmp_path / "rfb.sock")  # stub never bound it; nothing leaked


def test_xauth_cookie_survives_a_shadowing_od_on_path(tmp_path):
    """Regression for issue #127688: a bare `od` on the gateway's PATH can be
    shadowed by an unrelated CLI of the same name (the real-world case: a
    `~/.local/bin/od` rejecting `-An`, producing an empty Xauthority cookie
    and a broken `xauth add` command that aborted screen startup before Xvnc
    ever ran). `command -p od`/`command -p tr` must use bash's own guaranteed
    default utility path instead, so the cookie generation succeeds even
    when $PATH's own `od` is broken -- confirmed here by never even putting
    a working `od`/`tr` on $PATH at all.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    xvnc_log = tmp_path / "xvnc-argv"
    xauth_log = tmp_path / "xauth-argv"
    (bindir / "Xvnc").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{xvnc_log}"\nexec sleep 3\n', encoding="utf-8")
    (bindir / "xauth").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{xauth_log}"\ncat >> "{xauth_log}"\nexit 0\n', encoding="utf-8")
    # The exact real-world collision: an unrelated `od` that rejects the real
    # utility's flags, producing empty output instead of failing loudly.
    (bindir / "od").write_text('#!/bin/sh\necho "not the coreutils od" >&2\nexit 1\n', encoding="utf-8")
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "dbus-run-session"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    # Deliberately no `tr`/coreutils-`od` symlinked onto $PATH: command -p
    # must resolve both from bash's own default path, not from here.
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "awk", "grep", "seq", "sleep", "kill"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env = {
        "PATH": str(bindir), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert xauth_log.exists(), "xauth was never invoked -- the cookie pipeline aborted"
    cookie_input = xauth_log.read_text(encoding="utf-8")
    assert "MIT-MAGIC-COOKIE-1" in cookie_input
    # A real 16-byte hex cookie, not the empty string the shadowed `od` would produce.
    cookie_hex = cookie_input.split("MIT-MAGIC-COOKIE-1", 1)[1].split()[0]
    assert len(cookie_hex) == 32 and all(c in "0123456789abcdef" for c in cookie_hex), repr(cookie_hex)
