"""One pinned browser profile, one launcher state root — whichever ``$HOME`` a Hermes process is given.

A container gateway runs on ``HOME={HERMES_HOME}`` and the tool children it spawns run on ``HOME={HERMES_HOME}/home``
(``hermes_constants.get_subprocess_home``). While the shared-profile root resolved from ``$HOME``, those two
processes kept one profile's launch record, leases and lock in two directories: the second attached, but its lease
landed where the first could not see it, so the first holder's stop read itself as the last holder and stopped the
browser the second was still driving; and a browser a crashed launcher left behind was invisible to the reaper on
the other side.

State a PRE-FIX launcher left under a profile ``$HOME`` is not migrated: ``_legacy_guard`` inspects it read-only and,
while it still names a live launcher, holder or browser, a launch/attach/stop on that profile is refused instead of
entering a second ownership domain.

Runs the real launcher against a fake Chromium that takes the profile singleton and serves ``/json/version``
(``tests/tools/browser_launcher_fakes.py``): every assertion is about a browser process living or dying — or about a
refusal — never about a path existing for its own sake.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from gateway import status as gateway_status
from tests.tools.browser_launcher_fakes import kill_fake, pids, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher
from tools.browser_launcher_holders import ns_identity

pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.live_system_guard_bypass]  # the fake browsers are ours to stop if a leg failed

_REPO = Path(__file__).resolve().parents[2]

# A Hermes process on a persistent profile: launch (or attach), print "<pid> <owned>" or "ERR <reason>", wait for a
# line on stdin, then exit normally — the session layer's exit teardown is what lets go of its browsers.
_PROFILE_USER = textwrap.dedent("""\
    import os, sys
    import tools.browser_session  # registers the exit teardown
    from tools import browser_launcher as launcher
    launcher.needs_chromium_sandbox_bypass = lambda: False
    exe, udd = sys.argv[1], sys.argv[2]
    browser, err = launcher.launch_chromium("profile:" + udd, executable=exe, user_data_dir=udd, ephemeral=False,
                                            headed=False, env=dict(os.environ), timeout=30)
    print(f"{browser.pid} {browser.owned}" if browser else f"ERR {err}", flush=True)
    sys.stdin.readline()
""")

_REAPER = textwrap.dedent("""\
    from tools import browser_launcher as launcher
    print(launcher.reap_orphaned_browsers(), flush=True)
""")


def _env(hermes_home: Path, home: Path, *, lock_dir: Path | None = None) -> dict[str, str]:
    """A child whose shared-profile root comes from its own ``HOME`` (unless a leg pins one explicitly).

    ``HERMES_REAL_HOME`` is what a container child carries (``apply_subprocess_home_env``); without it,
    ``get_real_home()`` falls back to the account's passwd home and leaves the tmp tree. The suite's own
    ``HERMES_GATEWAY_LOCK_DIR`` is dropped: an explicit override legitimately collapses the roots, which is the very
    split under test.
    """
    env = {**os.environ, "HERMES_HOME": str(hermes_home), "HOME": str(home),
           "HERMES_REAL_HOME": str(hermes_home)}
    env.pop("HERMES_GATEWAY_LOCK_DIR", None)
    env.pop("XDG_STATE_HOME", None)
    if lock_dir is not None:
        env["HERMES_GATEWAY_LOCK_DIR"] = str(lock_dir)
    return env


def _user(hermes_home: Path, home: Path, script: Path, udd: Path, *,
          lock_dir: Path | None = None) -> tuple[subprocess.Popen, tuple[int, bool] | str]:
    """A launcher process: ``(proc, (pid, owned))`` when it launched or attached, ``(proc, error)`` when refused."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", _PROFILE_USER, str(script), str(udd)], cwd=_REPO,
                            env=_env(hermes_home, home, lock_dir=lock_dir), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if line.startswith("ERR "):
        return proc, line[4:]
    parts = line.split()
    assert len(parts) == 2 and parts[0].isdigit(), (
        (proc.stderr.read() if proc.stderr is not None else "")[-2000:] if proc.poll() is not None else line)
    return proc, (int(parts[0]), parts[1] == "True")


def _reap(hermes_home: Path, home: Path, *, lock_dir: Path | None = None) -> int:
    done = subprocess.run([sys.executable, "-c", _REAPER], check=False, cwd=_REPO,
                          env=_env(hermes_home, home, lock_dir=lock_dir),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return int(done.stdout.strip().splitlines()[-1])


def _gone(pid: int) -> bool:
    return wait_until(lambda: not launcher._pid_alive(pid), timeout=15, what=f"pid {pid} to exit")


def _state_name(udd: Path) -> str:
    return launcher._profile_digest(str(udd))[:32]


def _pre_fix_state_dir(hermes_home: Path, udd: Path) -> Path:
    """The directory a pre-fix launcher kept this profile's state in: its ``{HERMES_HOME}/home`` root."""
    return (hermes_home / "home" / ".local" / "state" / "hermes" / "gateway-locks"
            / "browser-shared-profiles" / _state_name(udd))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _pre_fix_record(udd: Path, *, pid: int, owner_pid: int, same_ns: bool = True) -> dict:
    """A launch record as a launcher on a profile ``HOME`` wrote it (its own PID namespace unless asked otherwise)."""
    return {"owner_pid": owner_pid, "owner_start_time": launcher._start_time(owner_pid) if owner_pid else None,
            "pid": pid, "start_time": 7, "user_data_dir": str(udd), "home_key": "pre-fix-install",
            "ephemeral": False, "ns": ns_identity() if same_ns else "4026531836:another-boot", "devtools": None}


def _write_pre_fix(hermes_home: Path, udd: Path, files: dict[str, dict]) -> Path:
    state = _pre_fix_state_dir(hermes_home, udd)
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    for name, payload in files.items():
        (state / name).write_text(json.dumps(payload))
    return state


def test_a_peer_stop_keeps_the_browser_the_other_home_is_still_driving(tmp_path):
    """A holder's stop must hand over, not tear down, while a holder on another ``$HOME`` still drives it."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    a, first = _user(hermes_home, hermes_home, script, udd)                        # gateway-side HOME
    b = None
    pid = None
    try:
        assert isinstance(first, tuple) and first[1] is True, first
        pid = first[0]
        b, second = _user(hermes_home, hermes_home / "home", script, udd)          # tool-child HOME
        assert isinstance(second, tuple), second
        assert second == (pid, False), "the second launcher must attach to the one browser"

        a.communicate("\n", timeout=60)   # the first holder lets go while the other still drives it
        assert launcher._pid_alive(pid), "a peer's stop stopped a browser the other home still holds"
        assert pids(udd, "launches") == [pid]
    finally:
        for proc in (a, b):
            if proc is not None:
                proc.kill()
        if pid is not None:
            kill_fake(pid, script)


def test_the_last_holder_stop_releases_the_browser(tmp_path):
    """With one holder, a stop still releases the browser: the hand-over above must not become a leak."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    a, first = _user(hermes_home, hermes_home, script, udd)
    pid = None
    try:
        assert isinstance(first, tuple) and first[1] is True, first
        pid = first[0]
        a.communicate("\n", timeout=60)
        assert _gone(pid), "the last holder's stop left the browser running"
    finally:
        a.kill()
        if pid is not None:
            kill_fake(pid, script)


def test_a_crashed_launchers_browser_is_reaped_from_the_other_home(tmp_path):
    """The canonical root is one, so a browser whose launcher died on the other ``$HOME`` is reaped."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    owner, first = _user(hermes_home, hermes_home, script, udd)
    assert isinstance(first, tuple), first
    pid = first[0]
    try:
        owner.kill()
        owner.wait(timeout=10)
        assert launcher._pid_alive(pid), "the orphan must outlive its launcher"

        assert _reap(hermes_home, hermes_home / "home") == 1, "invisible orphan on the repointed HOME"
        assert _gone(pid)
    finally:
        kill_fake(pid, script)


def test_pre_fix_state_with_a_live_owner_blocks_the_launch(tmp_path):
    """A launcher from before the fix that is still alive keeps the profile out of this version's hands."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    state = _write_pre_fix(hermes_home, udd, {
        f"{launcher._profile_digest(str(udd))[:16]}-{os.getpid()}.json":
            _pre_fix_record(udd, pid=os.getpid(), owner_pid=os.getpid())})
    before = {p.name: p.read_bytes() for p in state.iterdir()}
    proc = None
    try:
        proc, outcome = _user(hermes_home, hermes_home / "home", script, udd)
        assert isinstance(outcome, str), f"expected a refusal, got {outcome}"
        assert str(state) in outcome, outcome
        assert pids(udd, "launches") == [], "a browser was started over a live pre-fix owner"
        assert {p.name: p.read_bytes() for p in state.iterdir()} == before, "the guard modified pre-fix state"
    finally:
        if proc is not None:
            proc.kill()
        for launched in pids(udd, "launches"):
            kill_fake(launched, script)


def test_a_pre_fix_foreign_record_still_refuses_the_profile(tmp_path):
    """The same-hostname case: a launch record from another PID namespace in the pre-fix root still refuses."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    pid = _dead_pid()
    os.symlink(f"{socket.gethostname()}-{pid}", udd / "SingletonLock")   # this hostname: no refusal from that alone
    _write_pre_fix(hermes_home, udd, {
        f"{launcher._profile_digest(str(udd))[:16]}-{pid}-nffffffffffff.json":
            _pre_fix_record(udd, pid=pid, owner_pid=pid + 1, same_ns=False)})
    proc = None
    try:
        proc, outcome = _user(hermes_home, hermes_home / "home", script, udd)
        assert isinstance(outcome, str), f"expected the foreign-holder refusal, got {outcome}"
        assert "another environment" in outcome, outcome
        assert pids(udd, "launches") == []
    finally:
        if proc is not None:
            proc.kill()
        for launched in pids(udd, "launches"):
            kill_fake(launched, script)


def test_dead_pre_fix_state_does_not_stop_a_live_holder(tmp_path):
    """Pre-fix state naming only dead processes neither blocks a launch nor gets a live browser killed."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    dead = _dead_pid()
    state = _write_pre_fix(hermes_home, udd, {
        f"{launcher._profile_digest(str(udd))[:16]}-{dead}.json":
            _pre_fix_record(udd, pid=dead, owner_pid=dead),
        f"{launcher._profile_digest(str(udd))[:16]}-{dead}@{dead}.lease": {
            "holder_pid": dead, "holder_start_time": 1, "pid": dead, "start_time": 7,
            "user_data_dir": str(udd), "ns": ns_identity()}})
    a = b = None
    pid = None
    try:
        a, first = _user(hermes_home, hermes_home, script, udd)
        assert isinstance(first, tuple) and first[1] is True, first
        pid = first[0]
        b, second = _user(hermes_home, hermes_home / "home", script, udd)
        assert isinstance(second, tuple) and second == (pid, False), second

        a.communicate("\n", timeout=60)   # the other holder is still live: this must not kill the browser
        assert launcher._pid_alive(pid), "dead pre-fix state caused a live holder's browser to be stopped"
        assert sorted(p.name for p in state.iterdir()), "the guard removed pre-fix state"

        b.communicate("\n", timeout=60)
        assert _gone(pid)
    finally:
        for proc in (a, b):
            if proc is not None:
                proc.kill()
        if pid is not None:
            kill_fake(pid, script)


def test_an_explicit_root_does_not_probe_pre_fix_state(tmp_path):
    """``HERMES_GATEWAY_LOCK_DIR`` keeps its meaning: it is the one root, and pre-fix state is not consulted."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    _write_pre_fix(hermes_home, udd, {
        f"{launcher._profile_digest(str(udd))[:16]}-{os.getpid()}.json":
            _pre_fix_record(udd, pid=os.getpid(), owner_pid=os.getpid())})   # would block, if it were consulted
    proc = None
    pid = None
    try:
        proc, outcome = _user(hermes_home, hermes_home / "home", script, udd,
                              lock_dir=tmp_path / "explicit-locks")
        assert isinstance(outcome, tuple) and outcome[1] is True, outcome
        pid = outcome[0]
    finally:
        if proc is not None:
            proc.kill()
        if pid is not None:
            kill_fake(pid, script)


def _start_fake(script: Path, udd: Path) -> subprocess.Popen:
    """A Chromium this process did not launch, the way another environment's browser looks (fakes, no launcher)."""
    proc = subprocess.Popen([str(script), "--remote-debugging-port=0", f"--user-data-dir={udd}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_until(lambda: (udd / "DevToolsActivePort").exists(), timeout=15, what="the fake to come up")
    return proc


def test_a_reachable_foreign_browser_is_not_leased_over_live_pre_fix_state(tmp_path):
    """A foreign endpoint that answers would be attached and leased — unless pre-fix state still names a holder.

    Every path that writes a lease passes the guard, so an attach is refused here instead of putting a second
    launcher in charge of the browser.
    """
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    fake = _start_fake(script, udd)
    lock = udd / "SingletonLock"
    lock.unlink(missing_ok=True)
    # Another environment's lock: another hostname, and a pid number that names nobody here (its endpoint is still
    # reachable, which is what would normally allow an attach — and a lease).
    os.symlink(f"another-container-{_dead_pid()}", lock)
    _write_pre_fix(hermes_home, udd, {
        f"{launcher._profile_digest(str(udd))[:16]}-{os.getpid()}.json":
            _pre_fix_record(udd, pid=os.getpid(), owner_pid=os.getpid())})
    proc = None
    try:
        proc, outcome = _user(hermes_home, hermes_home / "home", script, udd)
        assert isinstance(outcome, str), f"expected a refusal, got {outcome}"
        assert "another environment" in outcome, outcome
        state = launcher._shared_state_dir(str(udd)) if launcher._canonical_shared_root() else None
        leases = sorted(state.glob("*.lease")) if state is not None else []
        assert leases == [], f"an attach leased a browser while pre-fix state was live: {leases}"
    finally:
        if proc is not None:
            proc.kill()
        fake.kill()
        fake.wait(timeout=10)


def test_a_live_pre_fix_lease_blocks_both_the_stop_and_the_reaper(tmp_path):
    """A holder from before this version still leasing the profile is not stopped over, and not reaped."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])   # a live same-namespace pid
    a = None
    pid = None
    try:
        a, first = _user(hermes_home, hermes_home, script, udd)     # nothing to guard yet: the launch is allowed
        assert isinstance(first, tuple) and first[1] is True, first
        pid = first[0]

        # A holder from before this version leases the same browser (its own pid, this namespace, still alive).
        _write_pre_fix(hermes_home, udd, {
            f"{launcher._profile_digest(str(udd))[:16]}-{pid}@{holder.pid}.lease": {
                "holder_pid": holder.pid, "holder_start_time": launcher._start_time(holder.pid),
                "pid": pid, "start_time": launcher._start_time(pid), "user_data_dir": str(udd),
                "ns": ns_identity()}})

        a.communicate("\n", timeout=60)          # its stop must not stop a browser a pre-fix holder leases
        assert launcher._pid_alive(pid), "a live pre-fix lease did not protect the browser from the stop"

        assert _reap(hermes_home, hermes_home) == 0, "the reaper collected a browser a pre-fix holder leases"
        assert launcher._pid_alive(pid)
    finally:
        holder.kill()
        holder.wait(timeout=10)
        if a is not None:
            a.kill()
        if pid is not None:
            kill_fake(pid, script)


def test_stopping_one_key_keeps_the_lease_another_key_of_this_process_holds(tmp_path, monkeypatch):
    """The guard's own stop path must not drop a lease that another key of this process still needs."""
    from tools import browser_launcher_holders as holders
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    alias = tmp_path / "pinned-profile-alias"
    udd.mkdir(parents=True, exist_ok=True)
    os.symlink(udd, alias)
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: False)

    first, err = launcher.launch_chromium("key-one", executable=str(script), user_data_dir=str(udd),
                                         ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
    assert err is None and first is not None and first.owned is True, err
    pid = first.pid
    try:
        second, err = launcher.launch_chromium("key-two", executable=str(script), user_data_dir=str(alias),
                                              ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
        assert err is None and second is not None and second.pid == pid and second.owned is False, err

        _write_pre_fix(hermes_home, udd, {
            f"{launcher._profile_digest(str(udd))[:16]}-{os.getpid()}.json":
                _pre_fix_record(udd, pid=os.getpid(), owner_pid=os.getpid())})   # guard: pre-fix state is live

        record = launcher._record_path(second)
        assert record is not None
        lease = record.with_name(f"{record.stem}@{os.getpid()}{holders.ns_tag()}.lease")
        assert lease.exists(), "precondition: this process leases the browser"

        launcher.stop_browser("key-one")     # blocked by the guard, and still needed by "key-two"
        assert lease.exists(), "the guard's stop path dropped a lease another key still holds"
        assert launcher._pid_alive(pid)
    finally:
        launcher.stop_browser("key-two")
        kill_fake(pid, script)


@pytest.mark.parametrize("payload", [b"{not json at all", b'{"user_data_dir": "\xff\xfe"}'],
                         ids=["invalid-json", "invalid-utf8"])
def test_unreadable_pre_fix_state_refuses_the_launch(tmp_path, payload):
    """State that exists but cannot be read is not "nothing to fear": it is refused, and left untouched."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    udd.mkdir(parents=True, exist_ok=True)
    corrupt = _pre_fix_state_dir(hermes_home, udd) / f"{launcher._profile_digest(str(udd))[:16]}-1.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(corrupt.parent, 0o700)
    corrupt.write_bytes(payload)
    before = corrupt.read_bytes()
    proc = None
    try:
        proc, outcome = _user(hermes_home, hermes_home / "home", script, udd)
        assert isinstance(outcome, str), f"expected a refusal, got {outcome}"
        assert "cannot be read or confirmed" in outcome, outcome
        assert pids(udd, "launches") == []
        assert corrupt.read_bytes() == before, "the guard touched an unreadable pre-fix file"
    finally:
        if proc is not None:
            proc.kill()
        for launched in pids(udd, "launches"):
            kill_fake(launched, script)


def test_an_unresolvable_os_user_home_refuses_the_profile(tmp_path, monkeypatch):
    """No stable per-user root means no persistent launch and no reaping — never a silent ``$HOME`` fallback."""
    import tempfile

    monkeypatch.setenv("HOME", str(tmp_path / "profile-home"))
    monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr("hermes_constants.get_real_home", lambda env=None: tempfile.gettempdir())

    with pytest.raises(RuntimeError):
        gateway_status._get_lock_dir(os_user=True)
    assert launcher._canonical_shared_root() is None
    assert launcher._shared_state_dir(str(tmp_path / "pinned-profile")) is None
    assert launcher.reap_orphaned_browsers() == 0

    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    env = {**_env(hermes_home, hermes_home), "HERMES_REAL_HOME": tempfile.gettempdir()}
    proc = subprocess.Popen([sys.executable, "-c", _PROFILE_USER, str(script), str(udd)], cwd=_REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    try:
        line = proc.stdout.readline().strip()
        assert line.startswith("ERR "), f"expected a refusal, got {line!r}"
        assert "cannot serialise" in line, line
        assert pids(udd, "launches") == []
    finally:
        proc.kill()
        for launched in pids(udd, "launches"):
            kill_fake(launched, script)


def test_a_browser_whose_state_dir_became_unusable_is_not_stopped(tmp_path, monkeypatch):
    """A persistent browser with no usable state dir cannot be proven the last holder's to stop.

    The ephemeral path stops an owned browser without a lock because nobody else can be driving it. A persistent
    profile is the opposite case: if its state dir is gone or not private, this process can neither read the
    holders nor take the lock, so it leaves the browser running instead of guessing.
    """
    script = write_fake_chromium(tmp_path / "chromium")
    udd, hermes_home = tmp_path / "pinned-profile", tmp_path / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: False)

    browser, err = launcher.launch_chromium("state-dir", executable=str(script), user_data_dir=str(udd),
                                           ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
    assert err is None and browser is not None and browser.owned is True, err
    pid = browser.pid
    state = launcher._shared_state_dir(str(udd))
    assert state is not None, "precondition: the state dir exists while the profile runs"
    try:
        os.chmod(state, 0o777)               # group/other writable: no longer a private dir, so no lock
        assert launcher._record_path(browser) is None, "precondition: no record path is reachable"

        launcher.stop_browser("state-dir")
        assert launcher._pid_alive(pid), "a browser with no usable state dir was killed without a lock"
    finally:
        os.chmod(state, 0o700)
        kill_fake(pid, script)


def test_the_lock_dir_keeps_its_existing_semantics(tmp_path, monkeypatch):
    """Role state still follows ``$HOME``; the opt-in scope follows the account home; overrides still win."""
    account, profile_home = tmp_path / "account", tmp_path / "account" / "home"
    for path in (account, profile_home):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(profile_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(account))
    monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    assert gateway_status._get_lock_dir() == profile_home / ".local" / "state" / "hermes" / "gateway-locks"
    assert gateway_status._get_lock_dir(os_user=True) == account / ".local" / "state" / "hermes" / "gateway-locks"

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert gateway_status._get_lock_dir() == tmp_path / "xdg" / "hermes" / "gateway-locks"
    assert gateway_status._get_lock_dir(os_user=True) == tmp_path / "xdg" / "hermes" / "gateway-locks"

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "explicit"))
    assert gateway_status._get_lock_dir() == tmp_path / "explicit"
    assert gateway_status._get_lock_dir(os_user=True) == tmp_path / "explicit"
