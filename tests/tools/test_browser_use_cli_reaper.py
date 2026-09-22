"""Browser Use harness daemon lifecycle (``tools/browser_use_cli_reaper.py``).

The harness daemon is DETACHED on purpose (``start_new_session=True``) so a named session
survives between ``browser_exec`` calls — ``ppid=1`` is the design. These tests pin the
reclamation contract that replaces the missing parent link: an owner claim, a dead-owner
sweep, an idle grace for live owners, and teardown of our own claims at exit.

Daemons are simulated with real child processes and a real AF_UNIX server speaking the
``browser_harness._ipc`` wire protocol, so the IPC-first stop path is exercised for real
rather than mocked.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from tools import browser_use_cli_reaper as reaper

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="harness IPC uses AF_UNIX on POSIX"
)


# ---------------------------------------------------------------- helpers


class _FakeDaemon:
    """A real process plus a real AF_UNIX server answering the harness meta protocol."""

    def __init__(self, runtime_dir, stem, *, answer_shutdown=True, respond=True):
        self.runtime_dir = runtime_dir
        self.stem = stem
        self.answer_shutdown = answer_shutdown
        self.respond = respond
        self.sock_path = str(runtime_dir / f"{stem}.sock")
        self.shutdown_requested = threading.Event()
        self._stop = threading.Event()

        # A real, signalable process whose cmdline identifies it as a harness daemon, so
        # the signal path's identity check runs against a genuine /proc-equivalent entry.
        self.proc = subprocess.Popen(
            [sys.executable, "-c",
             "import sys, time\n"
             "sys.argv[0] = 'browser_harness.daemon'\n"
             "time.sleep(300)\n"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        (runtime_dir / f"{stem}.pid").write_text(str(self.proc.pid), encoding="utf-8")

        if respond:
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(self.sock_path)
            self._server.listen(8)
            self._server.settimeout(0.2)
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                try:
                    data = conn.recv(1 << 16)
                    req = json.loads(data.decode() or "{}")
                except (OSError, ValueError):
                    continue
                meta = req.get("meta")
                if meta == "ping":
                    resp = {"pong": True, "pid": self.proc.pid}
                elif meta == "shutdown":
                    self.shutdown_requested.set()
                    if self.answer_shutdown:
                        resp = {"ok": True}
                        # A real daemon exits after acknowledging.
                        self.proc.terminate()
                    else:
                        resp = {"error": "stale-session recovery did not stop"}
                else:
                    resp = {"error": "unsupported"}
                with_newline = (json.dumps(resp) + "\n").encode()
                try:
                    conn.sendall(with_newline)
                except OSError:
                    pass

    def claim(self, owner_pid):
        (self.runtime_dir / f"{self.stem}.owner_pid").write_text(str(owner_pid), encoding="utf-8")

    def alive(self):
        return self.proc.poll() is None

    def cleanup(self):
        self._stop.set()
        if self.respond:
            try:
                self._server.close()
            except OSError:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)


def _dead_pid():
    """A PID that is certainly not running: spawn a process and reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    return proc.pid


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch):
    """An isolated BH_RUNTIME_DIR, with the reaper's claim registry reset per test."""
    d = tmp_path / "rt"
    d.mkdir()
    monkeypatch.setenv("BH_RUNTIME_DIR", str(d))
    monkeypatch.setattr(reaper, "_claimed_runtime_dirs", set())
    return d


@pytest.fixture
def daemons(runtime_dir):
    made = []

    def _make(stem="bu-probe", **kwargs):
        d = _FakeDaemon(runtime_dir, stem, **kwargs)
        made.append(d)
        return d

    yield _make
    for d in made:
        d.cleanup()


# ---------------------------------------------------------------- runtime dir resolution


def test_runtime_dir_follows_the_harness_precedence_order(tmp_path):
    """BH_RUNTIME_DIR > BH_TMP_DIR > BH_HOME/runtime, matching browser_harness._ipc."""
    explicit = {"BH_RUNTIME_DIR": str(tmp_path / "a"), "BH_TMP_DIR": str(tmp_path / "b"),
                "BH_HOME": str(tmp_path / "c")}
    assert reaper.harness_runtime_dir(explicit) == (tmp_path / "a").resolve()

    assert reaper.harness_runtime_dir(
        {"BH_TMP_DIR": str(tmp_path / "b"), "BH_HOME": str(tmp_path / "c")}
    ) == (tmp_path / "b").resolve()

    assert reaper.harness_runtime_dir({"BH_HOME": str(tmp_path / "c")}) == \
        (tmp_path / "c").resolve() / "runtime"


def test_file_stem_matches_the_harness_shared_dir_rule():
    """A caller-supplied runtime dir isolates one daemon ("bu"); BH_RUNTIME_DIR_SHARED=1
    puts the name back in the filename ("bu-<NAME>"), as _ipc._runtime_stem does."""
    assert reaper._runtime_stem({"BH_RUNTIME_DIR": "/x"}, "alpha") == "bu"
    assert reaper._runtime_stem({"BH_RUNTIME_DIR": "/x", "BH_RUNTIME_DIR_SHARED": "1"}, "alpha") == "bu-alpha"
    assert reaper._runtime_stem({}, "alpha") == "bu-alpha"
    assert reaper._runtime_stem({}, "") == "bu-default"


# ---------------------------------------------------------------- claim


def test_claim_records_this_process_and_refreshes_the_activity_marker(runtime_dir):
    """The claim is rewritten on every call, so its mtime is a restart-proof activity
    marker — the socket's mtime never moves once the daemon is listening."""
    env = {"BH_RUNTIME_DIR": str(runtime_dir), "BH_RUNTIME_DIR_SHARED": "1"}
    reaper.claim_session(env, "alpha")
    claim = runtime_dir / "bu-alpha.owner_pid"
    assert claim.read_text().strip() == str(os.getpid())

    first = claim.stat().st_mtime
    time.sleep(0.05)
    os.utime(claim, (first - 10_000, first - 10_000))
    reaper.claim_session(env, "alpha")
    assert claim.stat().st_mtime > first - 10_000


def test_claim_failure_does_not_raise(tmp_path, monkeypatch, runtime_dir):
    """An unwritable runtime dir degrades to the idle-only path, never breaks the tool call."""
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    try:
        reaper.claim_session({"BH_RUNTIME_DIR": str(blocked), "BH_RUNTIME_DIR_SHARED": "1"}, "alpha")
    finally:
        blocked.chmod(0o700)


# ---------------------------------------------------------------- reap decisions


def test_dead_owner_daemon_is_stopped_immediately(runtime_dir, daemons):
    """The actual leak: the hermes process that spawned the daemon is gone, so nobody
    would ever reclaim it. No idle grace applies — it can never be used again."""
    daemon = daemons()
    daemon.claim(_dead_pid())

    reaper.reap_orphaned_harness_daemons()

    assert daemon.shutdown_requested.is_set(), "must stop over IPC, not by signal"
    daemon.proc.wait(timeout=10)
    assert not daemon.alive()
    assert not (runtime_dir / "bu-probe.sock").exists()
    assert not (runtime_dir / "bu-probe.pid").exists()


def test_live_owner_daemon_within_grace_is_left_alone(runtime_dir, daemons):
    """A session this process is actively using must survive the sweep."""
    daemon = daemons()
    daemon.claim(os.getpid())

    reaper.reap_orphaned_harness_daemons()

    assert not daemon.shutdown_requested.is_set()
    assert daemon.alive()


def test_live_owner_daemon_idle_past_grace_is_stopped(runtime_dir, daemons, monkeypatch):
    """A leaked claim must not make a daemon immortal — the same escape hatch the
    agent-browser lane uses when an owner is alive but the session is abandoned."""
    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "BROWSER_ORPHAN_GRACE_SECONDS", 1)

    daemon = daemons()
    daemon.claim(os.getpid())
    old = time.time() - 3600
    for suffix in (".owner_pid", ".pid", ".sock"):
        os.utime(runtime_dir / f"bu-probe{suffix}", (old, old))

    reaper.reap_orphaned_harness_daemons()

    assert daemon.shutdown_requested.is_set()
    daemon.proc.wait(timeout=10)
    assert not daemon.alive()


def test_unclaimed_daemon_is_idle_gated_not_trusted(runtime_dir, daemons, monkeypatch):
    """A legacy daemon (no claim file) is reaped on idle, but never inside the grace."""
    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "BROWSER_ORPHAN_GRACE_SECONDS", 3600)

    daemon = daemons()
    reaper.reap_orphaned_harness_daemons()
    assert daemon.alive(), "fresh unclaimed daemon is within grace"

    monkeypatch.setattr(bt, "BROWSER_ORPHAN_GRACE_SECONDS", 1)
    old = time.time() - 3600
    for suffix in (".pid", ".sock"):
        os.utime(runtime_dir / f"bu-probe{suffix}", (old, old))

    reaper.reap_orphaned_harness_daemons()
    daemon.proc.wait(timeout=10)
    assert not daemon.alive()


def test_dead_daemon_endpoint_files_are_cleared(runtime_dir):
    """Stale bu-*.sock/.pid pairs left by a crashed daemon are removed, so the next call
    does not connect to an endpoint whose process is gone."""
    (runtime_dir / "bu-ghost.pid").write_text(str(_dead_pid()), encoding="utf-8")
    (runtime_dir / "bu-ghost.sock").write_text("", encoding="utf-8")
    (runtime_dir / "bu-ghost.owner_pid").write_text(str(_dead_pid()), encoding="utf-8")

    reaper.reap_orphaned_harness_daemons()

    assert not (runtime_dir / "bu-ghost.pid").exists()
    assert not (runtime_dir / "bu-ghost.sock").exists()
    assert not (runtime_dir / "bu-ghost.owner_pid").exists()


# ---------------------------------------------------------------- stop mechanics


def test_ipc_shutdown_is_preferred_over_signalling(runtime_dir, daemons):
    """meta:shutdown is the only stop that lets the daemon release a billable Browser Use
    cloud browser first, so it must be tried before any signal."""
    daemon = daemons()
    daemon.claim(_dead_pid())

    reaper.reap_orphaned_harness_daemons()

    assert daemon.shutdown_requested.is_set()


def test_signal_fallback_when_the_socket_is_gone(runtime_dir, daemons):
    """A daemon whose socket vanished cannot be asked politely; it is still ours to stop."""
    daemon = daemons(respond=False)
    daemon.claim(_dead_pid())
    assert not (runtime_dir / "bu-probe.sock").exists()

    reaper.reap_orphaned_harness_daemons()

    daemon.proc.wait(timeout=10)
    assert not daemon.alive()


def test_signal_fallback_when_shutdown_is_refused(runtime_dir, daemons):
    """A daemon that answers ping but refuses shutdown still gets stopped."""
    daemon = daemons(answer_shutdown=False)
    daemon.claim(_dead_pid())

    reaper.reap_orphaned_harness_daemons()

    assert daemon.shutdown_requested.is_set()
    daemon.proc.wait(timeout=10)
    assert not daemon.alive()


def test_refuses_to_signal_a_process_that_is_not_a_harness_daemon(runtime_dir):
    """The pid file is a plain file: a planted or recycled PID must not turn the reaper
    into an arbitrary-process killer. Identity is verified before any signal."""
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        (runtime_dir / "bu-imposter.pid").write_text(str(victim.pid), encoding="utf-8")
        (runtime_dir / "bu-imposter.owner_pid").write_text(str(_dead_pid()), encoding="utf-8")

        reaper.reap_orphaned_harness_daemons()

        assert victim.poll() is None, "unrelated process must not be signalled"
    finally:
        victim.kill()
        victim.wait(timeout=10)


def test_out_of_range_pid_file_is_never_signalled(runtime_dir):
    """0 and negatives address a process GROUP (os.kill(-1) hits everything the user owns);
    values past signed 32-bit make os.kill raise. All are rejected at read time."""
    probe = runtime_dir / "probe.pid"
    for bad in ("0", "-1", "99999999999999", "not-a-pid", ""):
        probe.write_text(bad, encoding="utf-8")
        assert reaper._read_pid(probe) is None

    probe.write_text(str(os.getpid()), encoding="utf-8")
    assert reaper._read_pid(probe) == os.getpid()


def test_wedged_daemon_is_escalated_to_sigkill(runtime_dir, daemons, monkeypatch):
    """A daemon that ignores SIGTERM must still be reclaimed — otherwise one wedged
    process keeps its CDP connection and memory forever."""
    monkeypatch.setattr(reaper, "_SHUTDOWN_EXIT_WAIT_S", 1.0)

    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import signal, sys, time\n"
         "sys.argv[0] = 'browser_harness.daemon'\n"
         "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
         "time.sleep(300)\n"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)  # let the handler install before we signal
        (runtime_dir / "bu-wedged.pid").write_text(str(proc.pid), encoding="utf-8")
        (runtime_dir / "bu-wedged.owner_pid").write_text(str(_dead_pid()), encoding="utf-8")

        reaper.reap_orphaned_harness_daemons()

        proc.wait(timeout=10)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------- endpoint-less daemons


def test_daemon_outliving_its_endpoint_files_is_reclaimed(runtime_dir, daemons, monkeypatch):
    """Observed live: a daemon kept running after its bu-*.sock/.pid were unlinked. It is
    unreachable over IPC and invisible to every file-based scan, so it can never be reused
    — a permanent leak only the process table can see."""
    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "BROWSER_ORPHAN_GRACE_SECONDS", 0)

    daemon = daemons(stem="bu-vanished")
    pid = daemon.proc.pid
    for suffix in (".pid", ".sock"):
        (runtime_dir / f"bu-vanished{suffix}").unlink()
    assert not reaper._sessions_in(runtime_dir), "precondition: invisible to the file sweep"

    reaper.reap_orphaned_harness_daemons()

    daemon.proc.wait(timeout=10)
    assert daemon.proc.poll() is not None, f"endpoint-less daemon PID {pid} must be reclaimed"


def test_endpoint_less_sweep_spares_a_daemon_inside_the_grace(runtime_dir, daemons, monkeypatch):
    """A daemon starting up has not published its pid file yet; killing it would break
    every cold start. The grace is what separates 'starting' from 'abandoned'."""
    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "BROWSER_ORPHAN_GRACE_SECONDS", 3600)

    daemon = daemons(stem="bu-starting")
    for suffix in (".pid", ".sock"):
        (runtime_dir / f"bu-starting{suffix}").unlink()

    reaper.reap_orphaned_harness_daemons()

    assert daemon.alive(), "a just-started daemon must not be reaped"


def test_endpoint_less_sweep_spares_indexed_and_foreign_processes(runtime_dir, daemons, monkeypatch):
    """The process-table sweep must not touch a daemon a pid file still points at (it has
    an owner and an idle clock), nor any non-harness process."""
    import tools.browser_tool as bt
    monkeypatch.setattr(bt, "BROWSER_ORPHAN_GRACE_SECONDS", 0)

    indexed = daemons(stem="bu-indexed")
    indexed.claim(os.getpid())

    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        reaper._sweep_unindexed()

        assert indexed.alive(), "a pid-file-indexed daemon is the file sweep's business"
        assert victim.poll() is None, "unrelated process must never be signalled"
    finally:
        victim.kill()
        victim.wait(timeout=10)


# ---------------------------------------------------------------- exit teardown


def test_exit_teardown_stops_only_our_own_claims(runtime_dir, daemons):
    """Several hermes processes share one runtime dir; at exit we stop what WE claimed and
    leave another live process's daemon running."""
    mine = daemons(stem="bu-mine")
    theirs = daemons(stem="bu-theirs")
    mine.claim(os.getpid())

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        theirs.claim(other.pid)

        reaper.shutdown_owned_harness_daemons()

        mine.proc.wait(timeout=10)
        assert not mine.alive()
        assert theirs.alive(), "another live process's daemon must survive our exit"
        assert not theirs.shutdown_requested.is_set()
    finally:
        other.kill()
        other.wait(timeout=10)


def test_atexit_hook_reaches_the_harness_sweep(monkeypatch):
    """The exit path must actually call the harness teardown — a detached daemon has no
    other owner, so a missing wire-up is the whole bug."""
    from tools import browser_tool_lifecycle as lifecycle
    import tools.browser_tool as bt

    called = []
    monkeypatch.setattr(lifecycle, "_shutdown_owned_harness_daemons", lambda: called.append(True))
    monkeypatch.setattr(lifecycle, "_reap_orphaned_browser_sessions", lambda: None)
    monkeypatch.setattr(lifecycle, "_stop_all_lightpanda", lambda: None)
    monkeypatch.setattr(lifecycle, "cleanup_all_browsers", lambda: None)
    monkeypatch.setattr(bt, "_cleanup_done", False)

    lifecycle._emergency_cleanup_all_sessions()

    assert called == [True]


def test_janitor_cycle_reaches_the_harness_sweep(monkeypatch):
    """The periodic orphan reap must sweep harness daemons too, so a crashed hermes
    process is cleaned up by a LIVE one without waiting for its own exit."""
    from tools import browser_tool_lifecycle as lifecycle
    from tools import browser_use_cli_reaper

    called = []
    monkeypatch.setattr(browser_use_cli_reaper, "reap_orphaned_harness_daemons",
                        lambda: called.append(True))
    monkeypatch.setattr(lifecycle, "_stop_all_lightpanda", lambda: None)

    lifecycle._reap_orphaned_browser_sessions()

    assert called == [True]
