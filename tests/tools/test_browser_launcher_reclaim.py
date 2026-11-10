"""The reaper never deletes the throwaway profile of a Chromium that is still running on it.

A Hermes process SIGKILLed mid-launch can leave a live Chromium whose launch record is missing or cannot
prove the pid (``tools/browser_launcher.py``: ``_reclaim_throwaway_dirs``, ``_reap_record``). Deleting the dir
under it corrupts a running browser, so the dir waits until that Chromium is gone.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.live_system_guard_bypass]  # the fake browsers are ours to stop if a leg failed

_REPO = Path(__file__).resolve().parents[2]


def _dir_of_a_dead_owner() -> Path:
    """A throwaway profile dir named for a Hermes process that has exited (it died mid-launch)."""
    from hermes_constants import hermes_home_key
    code = ("import sys\n"
            "from tools.browser_launcher import _default_user_data_dir, _prepare_dir\n"
            "udd = _default_user_data_dir('browser-exec-default', sys.argv[1])\n"
            "_prepare_dir(udd, True)\n"
            "print(udd, flush=True)\n")
    return Path(subprocess.run([sys.executable, "-c", code, hermes_home_key()], cwd=_REPO, env=dict(os.environ),
                               capture_output=True, text=True, check=True).stdout.strip())


def _start_browser_on(script: Path, udd: Path, log: Path) -> subprocess.Popen:
    with open(log, "wb") as out:
        proc = subprocess.Popen([str(script), f"--user-data-dir={udd}"], start_new_session=True,
                                stdout=out, stderr=out)
    wait_until(lambda: (udd / "DevToolsActivePort").exists() or proc.poll() is not None, timeout=15,
               what="the fake browser to write DevToolsActivePort")
    assert proc.poll() is None, f"the fake browser exited during startup: {log.read_text()[-2000:]}"
    return proc


def _stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)


def test_an_unrecorded_live_browsers_throwaway_profile_survives_the_reclaim(tmp_path):
    """Owner SIGKILLed before its record existed, its Chromium still up: reclaim waits for that Chromium."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd = _dir_of_a_dead_owner()
    proc = _start_browser_on(script, udd, tmp_path / "chromium.log")
    try:
        launcher.reap_orphaned_browsers()
        assert udd.is_dir(), "the reaper deleted the profile of a Chromium still running on it"
        assert proc.poll() is None

        _stop(proc)
        launcher.reap_orphaned_browsers()
        assert not udd.exists()
    finally:
        kill_fake(proc.pid, script)


def test_a_live_browsers_profile_survives_when_its_record_cannot_prove_the_pid(tmp_path):
    """A record whose start time no longer proves the pid is never signalled; its dir must not go either while a
    Chromium still runs on it."""
    from hermes_constants import hermes_home_key
    script = write_fake_chromium(tmp_path / "chromium")
    udd = _dir_of_a_dead_owner()
    proc = _start_browser_on(script, udd, tmp_path / "chromium.log")
    try:
        unprovable = launcher.LaunchedBrowser(key="lost", pid=proc.pid, start_time=1, cdp_url="", http_url="",
                                              user_data_dir=str(udd), owned=True, ephemeral=True,
                                              home_key=hermes_home_key())
        launcher._write_record(unprovable)  # this process's record of a browser it no longer tracks

        assert launcher.reap_orphaned_browsers() == 0
        assert proc.poll() is None, "signalled on an unproven pid"
        assert udd.is_dir(), "the reaper deleted the profile of a Chromium still running on it"
    finally:
        kill_fake(proc.pid, script)
