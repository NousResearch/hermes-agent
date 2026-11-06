"""A persistent profile is never opened by a Chromium older than the one that last wrote it: Chromium does not
support downgrades and crashes on the newer data at startup (``tools/browser_launcher_version.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import browser_launcher as launcher
from tools.bot_desktop import browser, runtime

pytestmark = pytest.mark.platforms("linux")


def _chromium(path: Path, version: str) -> str:
    """A ``chromium`` that reports ``version`` and, started as a browser, notes it in ``<udd>/spawns`` and
    crashes with SIGTRAP's status, as an older Chromium does on a newer profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'[ "$1" = --version ] && {{ echo "Chromium {version} built on Debian"; exit 0; }}\n'
        'for a; do case $a in --user-data-dir=*) touch "${a#--user-data-dir=}/spawns" ;; esac; done\n'
        "exit 133\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


def _profile(udd: Path, last_version: str) -> Path:
    udd.mkdir(parents=True, exist_ok=True)
    (udd / "Last Version").write_text(last_version, encoding="utf-8")
    return udd


def test_a_persistent_profile_is_not_launched_with_an_older_chromium(tmp_path):
    udd = _profile(tmp_path / "profile", "154.0.8037.57")
    exe = _chromium(tmp_path / "pw" / "chrome", "145.0.7632.0")

    browser_, err = launcher.launch_chromium(f"profile:{udd}", executable=exe, user_data_dir=str(udd),
                                             ephemeral=False)

    assert browser_ is None
    assert err and "Chromium 154" in err and "Chromium 145" in err
    assert not (udd / "spawns").exists()  # refused before Chromium could touch the profile


def test_bot_desktop_keeps_the_chromium_that_can_open_its_profile(tmp_path, monkeypatch):
    """pm's packaged build is preferred, but not over a profile a newer system Chromium already wrote."""
    managed = _chromium(tmp_path / "pw" / "chrome-linux" / "chrome", "145.0.7632.0")
    system = _chromium(tmp_path / "bin" / "chromium", "155.0.8059.79")
    monkeypatch.setattr("tools.browser_session_cloud.executable_path", lambda: "")
    monkeypatch.setattr("tools.browser_session_cloud.user_data_dir", lambda: "")
    monkeypatch.setattr("hermes_cli.browser_runtime.chromium_executable", lambda *, allow_override=True: managed)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: system if name == "chromium" else None)
    monkeypatch.setattr(browser, "_userns_restricted", lambda: False)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")

    assert browser.executable() == managed  # a fresh profile: the packaged build, as before

    _profile(tmp_path / "bot-desktop" / "browser-profile", "154.0.8037.57")
    assert browser.executable() == system
