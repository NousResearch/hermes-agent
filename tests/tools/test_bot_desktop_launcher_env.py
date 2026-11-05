"""The Bot Desktop's Xfce session gets a data-dir search path xfce4-panel can load its plugins from."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def test_session_drops_login_flatpak_data_dirs_and_keeps_the_rest(tmp_path):
    """A login session's flatpak exports dirs make xfce4-panel 4.20 fail every plugin ("Plugin loading
    failure"), so the session must not see them; the host's other data dirs keep their order."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    seen = tmp_path / "session-xdg-data-dirs"
    (bindir / "Xvnc").write_text("#!/bin/sh\nexec sleep 3\n", encoding="utf-8")
    (bindir / "dbus-run-session").write_text(f'#!/bin/sh\nprintf "%s" "$XDG_DATA_DIRS" > "{seen}"\n', encoding="utf-8")
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "od", "tr", "awk", "grep", "seq", "sleep", "kill"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env = {
        "PATH": str(bindir), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
        "XDG_DATA_DIRS": f"/usr/share/gnome:{tmp_path}/.local/share/flatpak/exports/share:"
                         "/var/lib/flatpak/exports/share:/usr/local/share/:/usr/share/",
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    assert seen.read_text(encoding="utf-8") == "/usr/share/gnome:/usr/local/share/:/usr/share/"
