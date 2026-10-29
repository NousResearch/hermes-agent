"""A pid in launcher state means something only inside the PID namespace and boot that wrote it.

One ``HERMES_HOME`` can be shared by a container gateway and a host CLI: a record, lease or throwaway profile dir
written in another PID namespace (or before a reboot), or one that does not say where it was written, names a
pid that may be dead HERE while its process lives THERE. It is never judged dead: never reaped, deleted,
dropped or handed over. Same-namespace state of a process that is gone is still cleaned up
(``tools/browser_launcher.py``, ``tools/browser_launcher_holders.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = pytest.mark.platforms("linux")

_REPO = Path(__file__).resolve().parents[2]
_FOREIGN = "4026531836:00000000-0000-0000-0000-000000000000"  # another pid namespace / boot
_FOREIGN_TAG = "-nffffffffffff"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _home_state() -> tuple[Path, Path]:
    from hermes_constants import get_hermes_home
    browser = Path(get_hermes_home()) / "cache" / "browser"
    records, throwaway = browser / "chromium-records", browser / "chromium"
    for path in (records, throwaway):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return records, throwaway


def _write(path: Path, data: dict, ns) -> Path:
    path.write_text(json.dumps({**data, **({"ns": ns} if ns is not None else {})}))
    return path


@pytest.mark.parametrize("ns", [_FOREIGN, None], ids=["other-namespace", "no-identity"])
def test_state_from_another_namespace_is_never_judged_dead(ns):
    records, throwaway = _home_state()
    owner, browser, holder = _dead_pid(), _dead_pid(), _dead_pid()
    suffix = _FOREIGN_TAG if ns else ""
    profile = throwaway / f"{'a' * 16}-{browser}-1{suffix}"
    (profile / "Default").mkdir(parents=True)
    orphan_dir = throwaway / f"{'b' * 16}-{owner}-1{suffix}"  # its owner died before writing a record
    orphan_dir.mkdir()
    record = _write(records / f"{'a' * 16}-{browser}{suffix}.json",
                    {"owner_pid": owner, "owner_start_time": None, "pid": browser, "start_time": 1,
                     "user_data_dir": str(profile), "home_key": "x", "ephemeral": True}, ns)
    lease = _write(records / f"{'c' * 16}-{browser}{suffix}@{holder}{suffix}.lease",
                   {"holder_pid": holder, "holder_start_time": None, "pid": browser, "start_time": 1}, ns)

    assert launcher.reap_orphaned_browsers() == 0

    assert record.exists(), "a record from another namespace was dropped"
    assert (profile / "Default").is_dir(), "the throwaway profile named by another namespace's record was deleted"
    assert orphan_dir.is_dir(), "another namespace's throwaway profile was reclaimed"
    assert lease.exists(), "another namespace's lease was dropped"


_HOLDER = textwrap.dedent("""\
    import os, sys
    from tools import browser_launcher as launcher
    exe, udd = sys.argv[1], sys.argv[2]
    browser, err = launcher.launch_chromium("profile:" + udd, executable=exe, user_data_dir=udd, ephemeral=False,
                                            headed=False, env=dict(os.environ), timeout=30)
    print(f"{browser.pid} {browser.owned}" if browser else f"ERR {err}", flush=True)
    sys.stdin.readline()
""")


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_a_holder_in_another_namespace_keeps_the_shared_browser_up(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    mine, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd), ephemeral=False,
                                         headed=False, env=dict(os.environ), timeout=30)
    assert err is None and mine.owned
    try:
        record, gone_here = launcher._record_path(mine), _dead_pid()
        foreign = _write(record.with_name(f"{record.stem}@{gone_here}{_FOREIGN_TAG}.lease"),
                         {"holder_pid": gone_here, "holder_start_time": None, "pid": mine.pid,
                          "start_time": mine.start_time, "user_data_dir": str(udd)}, _FOREIGN)

        launcher.stop_browser(key)
        assert launcher._pid_alive(mine.pid), "stopped a browser a holder in another namespace may still drive"
        assert launcher.reap_orphaned_browsers() == 0 and launcher._pid_alive(mine.pid)
        assert foreign.exists() and record.exists()
    finally:
        kill_fake(mine.pid, script)


_FOREIGN_ATTACHER = textwrap.dedent("""\
    import sys
    from tools import browser_launcher_holders as holders
    holders.ns_identity = lambda: sys.argv[3]  # this process writes as if from another pid namespace
""") + _HOLDER + 'launcher.stop_browser("profile:" + udd)\n'


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_an_attacher_from_another_namespace_is_seen_by_the_owner(tmp_path):
    """A host CLI attaching to a container gateway's shared Chromium (one lock dir, one network) leases the
    browser under the OWNER's record identity, its own namespace tag only on the holder part: the owner's stop
    finds that lease and keeps the browser up. Once the attacher lets go, nothing of it is left behind."""
    import hashlib
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    mine, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd), ephemeral=False,
                                         headed=False, env=dict(os.environ), timeout=30)
    assert err is None and mine.owned
    holder = subprocess.Popen([sys.executable, "-c", _FOREIGN_ATTACHER, str(script), str(udd), _FOREIGN],
                              cwd=_REPO, env=dict(os.environ), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True)
    try:
        assert holder.stdout.readline().split() == [str(mine.pid), "False"]
        record = launcher._record_path(mine)
        foreign_tag = f"-n{hashlib.sha256(_FOREIGN.encode()).hexdigest()[:12]}"
        assert [p.name for p in record.parent.glob("*.lease")] == [f"{record.stem}@{holder.pid}{foreign_tag}.lease"]

        launcher.stop_browser(key)
        assert launcher._pid_alive(mine.pid), "the owner stopped a browser another namespace's holder still drives"
        assert launcher.reap_orphaned_browsers() == 0 and launcher._pid_alive(mine.pid)

        holder.stdin.write("\n")
        holder.stdin.flush()
        assert holder.wait(timeout=15) == 0
        assert not list(record.parent.glob("*.lease")), "the attacher left its lease behind"
        assert launcher.reap_orphaned_browsers() == 1
        assert wait_until(lambda: not launcher._pid_alive(mine.pid), timeout=15, what="the browser to stop")
    finally:
        holder.kill()
        kill_fake(mine.pid, script)


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_a_dead_holder_in_this_namespace_is_still_dropped(tmp_path):
    """Same namespace, behaviour unchanged: the lease of a holder that crashed is dropped, so the launcher's
    stop is the last one out and stops the browser."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    mine, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd), ephemeral=False,
                                         headed=False, env=dict(os.environ), timeout=30)
    assert err is None and mine.owned
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(script), str(udd)], cwd=_REPO,
                              env=dict(os.environ), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().split() == [str(mine.pid), "False"]
        record = launcher._record_path(mine)
        assert list(record.parent.glob("*.lease"))
        holder.kill()
        holder.wait(timeout=10)

        launcher.stop_browser(key)
        assert wait_until(lambda: not launcher._pid_alive(mine.pid), timeout=15, what="the browser to stop")
        assert not list(record.parent.glob("*.lease")) and not record.exists()
    finally:
        holder.kill()
        kill_fake(mine.pid, script)
