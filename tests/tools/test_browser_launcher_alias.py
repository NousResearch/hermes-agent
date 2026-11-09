"""A shared profile reached through a symlink is the same browser launch: adopting it keeps its launch identity.

A Hermes process attaching through another spelling of the profile (``/alias -> /profile``) after the owner
crashed takes the launch record over. The record must still describe the browser as launched (the
``--user-data-dir`` its argv carries), or the reaper cannot verify it once this process crashes too: it would
leave Chromium running and delete its only record (``tools/browser_launcher.py``). Runs the real launcher against a
fake Chromium that takes the profile singleton and serves ``/json/version`` like the real one.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, pids, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.live_system_guard_bypass]  # the fake browser is ours to stop if a leg failed

_REPO = Path(__file__).resolve().parents[2]

# A Hermes process on a persistent profile: launches (or attaches), prints "<pid> <owned>", then waits to be killed.
_HOLDER = textwrap.dedent("""\
    import os, sys
    import tools.browser_session  # registers the exit teardown
    from tools import browser_launcher as launcher
    launcher.needs_chromium_sandbox_bypass = lambda: False
    exe, udd = sys.argv[1], sys.argv[2]
    browser, err = launcher.launch_chromium("profile:" + udd, executable=exe, user_data_dir=udd, ephemeral=False,
                                            headed=False, env=dict(os.environ), timeout=30)
    print(f"{browser.pid} {browser.owned}" if browser else f"ERR {err!r}", flush=True)
    sys.stdin.readline()
""")


def _crashing_holder(script: Path, udd: Path) -> tuple[int, str]:
    """A holder that launches or attaches on ``udd``, then is SIGKILLed (no teardown runs)."""
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(script), str(udd)], cwd=_REPO, env=dict(os.environ),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().split()
        assert line and line[0] != "ERR", proc.stderr.read()[-2000:] if not line else line
    finally:
        proc.kill()
        proc.wait(timeout=10)
    return int(line[0]), line[1]


def test_a_browser_adopted_through_a_symlink_is_still_reaped_after_its_adopter_crashes(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "profile"
    alias = tmp_path / "alias"
    alias.symlink_to(udd)
    try:
        pid, owned = _crashing_holder(script, udd)
        assert owned == "True"
        adopted, owned = _crashing_holder(script, alias)
        assert (adopted, owned) == (pid, "False")

        assert launcher.reap_orphaned_browsers() == 1
        assert wait_until(lambda: not launcher._pid_alive(pid), timeout=15, what="the orphaned browser to stop")
        state = launcher._shared_state_dir(str(udd))
        assert not list(state.glob("*.json")), "a launch record outlived its reaped browser"
    finally:
        for spawned in pids(udd, "spawns"):
            kill_fake(spawned, script)
