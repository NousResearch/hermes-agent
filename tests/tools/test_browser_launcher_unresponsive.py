"""A live browser whose DevTools endpoint stops answering for a while is still the profile's browser.

Chromium paused (SIGSTOP, a debugger, a swapped-out host) keeps its profile singleton; a second launch on the
profile would only wipe the live browser's ``DevToolsActivePort`` and lose it for every later attach (its
singleton refuses the next launch with "SingletonLock exists"). The launcher waits for the live holder and then
refuses with a clear error, leaving its profile untouched (``tools/browser_launcher.py``). Runs the real launcher
against a fake Chromium that takes the profile singleton and serves ``/json/version`` like the real one.
"""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, pids, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.live_system_guard_bypass]  # the fake browser is ours to stop if a leg failed


def _start_fake(script, udd) -> subprocess.Popen:
    proc = subprocess.Popen([str(script), "--remote-debugging-port=0", f"--user-data-dir={udd}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_until(lambda: (udd / "DevToolsActivePort").exists(), timeout=15, what="the fake to come up")
    return proc


def test_a_paused_shared_browser_is_refused_then_attached_once_it_answers_again(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    browser = _start_fake(script, udd)
    port_file = (udd / "DevToolsActivePort").read_text()
    key = f"profile:{udd}"

    def attach():
        return launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd), ephemeral=False,
                                        headed=False, env=dict(os.environ), timeout=3)

    try:
        os.kill(browser.pid, signal.SIGSTOP)
        try:
            attached, err = attach()
        finally:
            os.kill(browser.pid, signal.SIGCONT)
        assert attached is None and err and str(browser.pid) in err, err
        assert (udd / "DevToolsActivePort").read_text() == port_file, "the live browser's endpoint file was wiped"
        assert pids(udd, "spawns") == [browser.pid], "a second Chromium was started on the live browser's profile"

        attached, err = attach()
        assert err is None and attached.pid == browser.pid and not attached.owned, err
        launcher.stop_browser(key)
    finally:
        for pid in pids(udd, "spawns"):
            kill_fake(pid, script)
        browser.wait(timeout=10)
