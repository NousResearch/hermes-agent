"""The Bot Desktop's Xfce session does not inherit desktop identity from the host login session."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def _session_env(tmp_path: Path, inherited: dict[str, str]) -> dict[str, str]:
    """Run the launcher with a stub Xvnc that records the environment the screen's session starts in."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    env_log = tmp_path / "xvnc-env"
    (bindir / "Xvnc").write_text(
        f'#!/bin/sh\nenv > "{env_log}"\nexec sleep 3\n', encoding="utf-8"
    )
    for stub in (
        "xdpyinfo",
        "setxkbmap",
        "xsetroot",
        "xset",
        "dbus-run-session",
        "xauth",
    ):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in (
        "mkdir",
        "sed",
        "cat",
        "printf",
        "dirname",
        "bash",
        "sh",
        "rm",
        "ln",
        "touch",
        "chmod",
        "od",
        "tr",
        "awk",
        "grep",
        "seq",
        "sleep",
        "kill",
        "env",
    ):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env = {
        **inherited,
        "PATH": str(bindir),
        "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t",
        "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"),
        "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"),
        "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
    )
    pairs = (
        line.split("=", 1)
        for line in env_log.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    return dict(pairs)


@pytest.mark.parametrize("host_prefix", ["hyprland-", "gnome-", None])
def test_menu_prefix_is_xfce_whatever_the_host_session_exports(tmp_path, host_prefix):
    """uwsm/Hyprland exports XDG_MENU_PREFIX=hyprland-; inherited, the Xfce panel looks for
    hyprland-applications.menu, which no Xfce install ships, and pops a modal error on every start."""
    inherited = {"XDG_MENU_PREFIX": host_prefix} if host_prefix else {}
    session = _session_env(tmp_path, inherited)
    assert session.get("XDG_MENU_PREFIX") == "xfce-"
    # The prefix must agree with the desktop identity the launcher already pins.
    assert session.get("XDG_CURRENT_DESKTOP") == "XFCE"
