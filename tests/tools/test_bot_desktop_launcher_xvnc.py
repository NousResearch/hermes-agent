"""The Bot Desktop's Xvnc is started with the RFB options the lease design relies on."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.linux_only


def test_xvnc_never_sends_the_holders_clipboard_to_watchers(tmp_path):
    """Whoever holds control may paste INTO the screen (AcceptCutText), but the screen's clipboard must
    not be pushed to every connected viewer (SendCutText off): watchers are not the holder."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "xvnc-argv"
    (bindir / "Xvnc").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_log}"\nexec sleep 3\n', encoding="utf-8")
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "dbus-run-session"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "seq", "sleep", "kill"):
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
    argv = argv_log.read_text(encoding="utf-8").split("\n")
    assert "-SendCutText=0" in argv, argv
    assert not any(a.startswith("-AcceptCutText") for a in argv), "paste into the screen must keep working"
    assert not os.path.exists(tmp_path / "rfb.sock")  # stub never bound it; nothing leaked
