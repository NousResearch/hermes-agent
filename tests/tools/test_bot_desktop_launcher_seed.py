"""Bot Desktop launcher seeds: the dock only points at programs that exist, the look is applied."""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def _seed(tmp_path: Path, fake_bins: list[str], browser_exec: str = "", profile_dir: str = "") -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for name in fake_bins:
        exe = bindir / name
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o755)
    # The launcher only needs xauth to accept the generated cookie; the test host need not ship X11 tools.
    xauth = bindir / "xauth"
    xauth.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    xauth.chmod(0o755)
    # The script's own tooling (mkdir, sed, cat, awk...) symlinked in, so PATH need not contain the
    # host's /usr/bin where a real chrome/thunar would leak into the dock under test.
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "grep"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    cfg = tmp_path / "xdg"
    env = {
        "PATH": str(bindir),
        "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(cfg),
        "HERMES_BD_SEED_ONLY": "1",
        **({"HERMES_BD_BROWSER_EXEC": browser_exec, "HERMES_BD_BROWSER_EXEC_LINE": f"Exec={browser_exec} --user-data-dir={profile_dir or f'{tmp_path}/bp'}"} if browser_exec else {}),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    return cfg


def test_dock_lists_only_programs_present_on_path(tmp_path):
    chrome = tmp_path / "bin" / "chrome"  # the browser is the one runtime.py resolved, never a PATH scan
    cfg = _seed(tmp_path, ["xfce4-terminal", "chrome", "firefox"], browser_exec=str(chrome))
    panel = ET.parse(cfg / "xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml")  # well-formed or this raises
    launcher_ids = [str(p.get("name")) for p in panel.iter("property") if p.get("value") == "launcher"]
    execs = sorted(
        line.split("=", 1)[1]
        for pid in launcher_ids
        for line in (cfg / "xfce4/panel" / pid.replace("plugin-", "launcher-") / "hermes.desktop").read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("Exec=")
    )
    assert execs == [f"{chrome} --user-data-dir={tmp_path}/bp", "xfce4-terminal"]


def test_browser_launcher_follows_the_profile_without_reseeding_the_panel(tmp_path):
    """The panel layout is seeded once (the human may have rearranged it), but the Browser entry's Exec=
    binds the bot's user-data-dir by absolute path: after `hermes profile rename` the dock must open the
    renamed profile's cookie jar, the same one agent-browser now drives, while the layout stays untouched."""
    chrome = tmp_path / "bin" / "chrome"
    cfg = _seed(tmp_path, ["xfce4-terminal", "chrome"], browser_exec=str(chrome), profile_dir="/old name/bp")
    panel_xml = cfg / "xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml"
    layout_before = panel_xml.read_text(encoding="utf-8-sig")
    shutil.rmtree(tmp_path / "bin")
    _seed(tmp_path, ["xfce4-terminal", "chrome"], browser_exec=str(chrome), profile_dir="/new name/bp")
    execs = [
        line for d in (cfg / "xfce4/panel").glob("launcher-*/hermes.desktop")
        for line in d.read_text(encoding="utf-8-sig").splitlines() if line.startswith("Exec=") and "user-data-dir" in line
    ]
    assert execs == [f"Exec={chrome} --user-data-dir=/new name/bp"]
    assert panel_xml.read_text(encoding="utf-8-sig") == layout_before


def test_browser_launcher_seeded_before_the_marker_still_follows_a_rename(tmp_path):
    """Profiles seeded by an earlier launcher have the Browser entry but no `.hermes-browser-launcher`;
    a rename must still reach their Exec= (the marker is recovered from our own entry), not leave them
    opening the old profile's empty cookie jar forever."""
    chrome = tmp_path / "bin" / "chrome"
    cfg = _seed(tmp_path, ["xfce4-terminal", "chrome"], browser_exec=str(chrome), profile_dir="/old name/bp")
    (cfg / "xfce4/panel/.hermes-browser-launcher").unlink()
    shutil.rmtree(tmp_path / "bin")
    _seed(tmp_path, ["xfce4-terminal", "chrome"], browser_exec=str(chrome), profile_dir="/new name/bp")
    execs = [
        line for d in (cfg / "xfce4/panel").glob("launcher-*/hermes.desktop")
        for line in d.read_text(encoding="utf-8-sig").splitlines() if line.startswith("Exec=") and "user-data-dir" in line
    ]
    assert execs == [f"Exec={chrome} --user-data-dir=/new name/bp"]


def test_look_is_seeded_with_wallpaper_and_theme(tmp_path):
    cfg = _seed(tmp_path, ["xfce4-terminal"])
    desktop = (cfg / "xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml").read_text(encoding="utf-8-sig")
    xsettings = (cfg / "xfce4/xfconf/xfce-perchannel-xml/xsettings.xml").read_text(encoding="utf-8-sig")
    assert str(LAUNCHER.with_name("wallpaper.png")) in desktop
    assert "PLACEHOLDER" not in desktop + xsettings
    assert os.path.isfile(LAUNCHER.with_name("wallpaper.png"))
