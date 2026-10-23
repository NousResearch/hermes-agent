"""The real-profile snapshot (the overlay of the user's Cookies / Login Data onto the profile copy) is part of the
launch step under the copy dir's cross-process launcher lock: two Hermes processes of one profile never copy into
the same snapshot dir at once, nor over a copy-browser that just came up (the later one attaches instead), and a
different profile's copy never waits on it (``tools/browser_session_real_profile.py``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, pids, write_fake_chromium

pytestmark = pytest.mark.platforms("linux")

_REPO = Path(__file__).resolve().parents[2]

# One Hermes process resolving its real-profile browser. The snapshot is the real copier on a fixture profile,
# wrapped only to log when it runs and to stay inside long enough for a rival process to show up.
_SESSION = textwrap.dedent("""\
    import json, os, sys, time
    from unittest.mock import patch
    import hermes_cli.browser_connect as bc
    from tools import browser_lightpanda, browser_session, browser_session_cloud
    from tools import browser_session_real_profile as rp
    chrome, src, log, gate = sys.argv[1:5]
    real_snapshot = bc.snapshot_real_profile

    def note(event):
        with open(log, "a") as f:
            f.write(f"{event} {os.getpid()}\\n")

    def snapshot(browser, _src=None):
        note("enter")
        deadline = time.monotonic() + 2.0  # wait for a rival's copy to start, if it can
        while open(log).read().count("enter") < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            return real_snapshot(browser, src)
        finally:
            note("exit")

    while not os.path.exists(gate):
        time.sleep(0.01)
    with patch.object(browser_session_cloud, "use_real_profile", return_value=True), \\
         patch.object(browser_lightpanda, "using_lightpanda_engine", return_value=False), \\
         patch.object(browser_session, "ensure_screen_for_headed_chromium", lambda: None), \\
         patch.object(bc, "detect_default_chromium", return_value="chrome"), \\
         patch.object(bc, "chromium_executable", return_value=chrome), \\
         patch.object(bc, "snapshot_real_profile", side_effect=snapshot):
        cdp, err = rp.real_profile_cdp()
        from tools.browser_launcher import get_browser
        running = get_browser(rp.real_profile_key())
        print(json.dumps({"cdp": cdp, "err": err, "owned": bool(running and running.owned)}), flush=True)
        sys.stdin.readline()
        rp.terminate_real_profile_browser()
""")


def _profile(root: Path) -> Path:
    (root / "Default").mkdir(parents=True)
    (root / "Local State").write_text('{"os_crypt": {}}', encoding="utf-8")
    with closing(sqlite3.connect(root / "Default" / "Cookies")) as conn, conn:
        conn.execute("create table marker(value)")
        conn.execute("insert into marker values('signed-in')")
    return root


def _overlapped(events: list[str]) -> bool:
    inside = 0
    for event in events:
        inside += 1 if event == "enter" else -1
        if inside > 1:
            return True
    return False


@pytest.mark.live_system_guard_bypass  # the fake browsers are ours to stop if a leg failed
@pytest.mark.parametrize("second", ["same-profile", "other-profile"])
def test_only_one_process_of_a_profile_copies_its_real_profile_at_a_time(tmp_path, second):
    chrome = write_fake_chromium(tmp_path / "chromium")
    src, log, gate = _profile(tmp_path / "real-profile"), tmp_path / "snapshots.log", tmp_path / "go"
    homes = [tmp_path / "home-a", tmp_path / "home-a" if second == "same-profile" else tmp_path / "home-b"]
    sessions = []
    for home in homes:
        home.mkdir(exist_ok=True)
        sessions.append(subprocess.Popen(
            [sys.executable, "-c", _SESSION, str(chrome), str(src), str(log), str(gate)], cwd=_REPO,
            env={**os.environ, "HERMES_HOME": str(home)}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
    try:
        gate.touch()
        results = [json.loads(session.stdout.readline()) for session in sessions]
        events = [line.split()[0] for line in log.read_text().splitlines()]

        assert all(r["err"] is None and r["cdp"] for r in results), results
        if second == "same-profile":
            assert not _overlapped(events), f"two processes copied into one snapshot dir at once: {events}"
            assert events == ["enter", "exit"], "the later process re-copied over the running copy-browser"
            assert results[0]["cdp"] == results[1]["cdp"] and sorted(r["owned"] for r in results) == [False, True]
        else:
            assert _overlapped(events), f"another profile's copy waited on this one: {events}"
            assert results[0]["cdp"] != results[1]["cdp"]
        copy = homes[0] / "browser-profile" / "chrome" / "Default" / "Cookies"
        with closing(sqlite3.connect(copy)) as conn:
            assert conn.execute("select value from marker").fetchone()[0] == "signed-in"
    finally:
        for session in sessions:
            session.kill()
        for home in set(homes):
            for pid in pids(home / "browser-profile" / "chrome", "launches"):
                kill_fake(pid, chrome)
