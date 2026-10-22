"""Apps on the Bot Desktop keep their accessibility tree on: computer_use reads elements through AT-SPI on
Linux, so a session that switches GTK's bridge off leaves every capture of a dock app with a bare window."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def test_the_session_every_screen_app_inherits_leaves_the_accessibility_bridge_on(tmp_path):
    """xfwm4, xfdesktop and xfce4-panel run inside dbus-run-session, and everything the dock starts inherits
    the panel's environment, so the environment dbus-run-session gets is the one every screen app gets."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    session_env = tmp_path / "session-env"
    (bindir / "Xvnc").write_text("#!/bin/sh\nexec sleep 3\n", encoding="utf-8")
    (bindir / "dbus-run-session").write_text(f'#!/bin/sh\nenv -0 > "{session_env}"\n', encoding="utf-8")
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "chmod", "od", "tr", "grep",
                 "seq", "sleep", "kill", "env"):
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
    seen = dict(item.split("=", 1) for item in session_env.read_text(encoding="utf-8-sig").split("\0") if "=" in item)
    assert seen.get("DISPLAY") == ":99", "the stub did not see the session environment"
    # GTK 3's and GTK 4's documented off switches for AT-SPI.
    assert seen.get("NO_AT_BRIDGE") != "1", "GTK 3 apps on the screen would publish no accessibility tree"
    assert seen.get("GTK_A11Y") != "none", "GTK 4 apps on the screen would publish no accessibility tree"
