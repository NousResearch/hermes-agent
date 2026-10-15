"""Idle lane closers: browsers a lane launches itself are torn down after the lane goes idle.

Regression coverage for the leak where a long-lived gateway never reaches the atexit hook, and the
lanes that own their browser process (real-profile Chrome, Browser Use harness daemon) have no
``_active_sessions`` entry for the janitor to reach — leaving Chrome and daemons running for days
after the run that started them.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest
import tools.browser_tool as browser_tool
import tools.browser_tool_lifecycle as lifecycle
from hermes_constants import get_hermes_home
from tools import browser_tool_real_profile as real_profile
from tools import browser_use_cli

REAL_PROFILE_KEY = f"real-profile:{browser_tool._REAL_PROFILE_SESSION}"


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "_idle_lane_closers", {})
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_real_profile_cdp_cache", {})
    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))


def _age(key: str, seconds: Optional[float] = None):
    """Push a lane's clock past its own window (default) or by ``seconds``."""
    entry = lifecycle._idle_lane_closers[key]
    entry["last"] -= entry["timeout"] + 1 if seconds is None else seconds


def _record_teardown(monkeypatch):
    """Stub the three real teardowns and return the call log, in the order they run."""
    calls = []
    monkeypatch.setattr(real_profile, "_agent_browser_close_session",
                        lambda name: calls.append(("close-session", name)))
    monkeypatch.setattr(real_profile, "_terminate_real_profile_chrome",
                        lambda: calls.append(("chrome", None)))
    monkeypatch.setattr(browser_use_cli, "stop_harness_daemons",
                        lambda cdp=None: calls.append(("harness", cdp)))
    return calls


def _write_claim(tmp_path, pid: int):
    """Write the socket-dir ``owner_pid`` claim the real-profile lane publishes for its daemon."""
    socket_dir = Path(tmp_path) / f"agent-browser-{browser_tool._REAL_PROFILE_SESSION}"
    socket_dir.mkdir(exist_ok=True)
    (socket_dir / f"{browser_tool._REAL_PROFILE_SESSION}.owner_pid").write_text(str(pid))


def _dead_pid() -> int:
    """A PID that really is dead, whatever the host's ``pid_max`` is."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def _run_one_worker_cycle(monkeypatch):
    """Drive the real cleanup-thread worker once, so the sweep the thread itself performs is tested."""
    monkeypatch.setattr(browser_tool, "_cleanup_running", True)
    monkeypatch.setattr(lifecycle, "_reap_orphaned_browser_sessions", lambda: None)
    monkeypatch.setattr(lifecycle, "_cleanup_inactive_browser_sessions", lambda: None)

    def _stop_after_one_cycle(_seconds):
        browser_tool._cleanup_running = False

    monkeypatch.setattr(lifecycle.time, "sleep", _stop_after_one_cycle)
    lifecycle._browser_cleanup_thread_worker()


def test_worker_sweep_fires_a_due_closer_once_and_keeps_a_failing_one(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    fired, failed = [], []

    def _boom():
        failed.append(1)
        raise RuntimeError("closer failed")

    lifecycle.register_idle_lane_closer("lane", lambda: fired.append(1), 60.0)
    lifecycle.register_idle_lane_closer("broken", _boom, 60.0)

    _run_one_worker_cycle(monkeypatch)
    assert fired == [], "must not close a lane that was just used"

    _age("lane")
    _age("broken")
    _run_one_worker_cycle(monkeypatch)  # a failing closer must not escape the thread

    assert fired == [1], "a fired closer must not fire again every window"
    assert "lane" not in lifecycle._idle_lane_closers, "a fired closer is disarmed"
    assert failed == [1]
    retry = lifecycle._idle_lane_closers["broken"]
    assert time.time() - retry["last"] < retry["timeout"], "a retry waits a full window"


def test_closers_run_under_their_own_profile_scope(monkeypatch, tmp_path):
    """The cleanup thread is process-global; each closer must re-enter its owner profile's scope."""
    _isolate(monkeypatch, tmp_path)
    home_a, home_b = str(tmp_path / "A"), str(tmp_path / "B")
    Path(home_a).mkdir()
    Path(home_b).mkdir()
    seen = []
    homes = iter([home_a, home_b, home_a])  # A -> B -> A, per the profile-scope rule
    monkeypatch.setattr(lifecycle, "get_hermes_home", lambda: next(homes))

    for index in range(3):
        lifecycle.register_idle_lane_closer(
            f"lane:{index}", lambda: seen.append(str(get_hermes_home())), 60.0)
        _age(f"lane:{index}")

    lifecycle._run_due_idle_lane_closers()

    assert seen == [home_a, home_b, home_a], "a closer must see its own home, not the launch profile's"
    assert str(get_hermes_home()) not in (home_a, home_b), "the scope must be restored afterwards"


@pytest.mark.parametrize("case,closes", [
    ("idle", True),
    ("claimed-by-own-pid", True),
    ("claimed-by-dead-pid", True),
    ("claimed-by-live-foreign-pid", False),
    ("real-profile-session-live", False),
    ("resolve-in-flight", False),
    ("lane-used-again", False),
])
def test_real_profile_closer_closes_only_an_idle_unclaimed_lane(monkeypatch, tmp_path, case, closes):
    """The attach daemon is one session name host-wide, and a live user session outranks our clock."""
    _isolate(monkeypatch, tmp_path)
    calls = _record_teardown(monkeypatch)
    browser_tool._real_profile_cdp_cache["cdp"] = "http://127.0.0.1:1234"
    held = False

    if case == "claimed-by-own-pid":
        _write_claim(tmp_path, os.getpid())
    elif case == "claimed-by-dead-pid":
        _write_claim(tmp_path, _dead_pid())
    elif case == "claimed-by-live-foreign-pid":
        _write_claim(tmp_path, os.getppid())
    elif case == "real-profile-session-live":
        browser_tool._active_sessions["task-1"] = {"features": {"local": True, "real_profile": True}}
    elif case == "resolve-in-flight":
        held = browser_tool._real_profile_cdp_lock.acquire(blocking=False)
        assert held
    elif case == "lane-used-again":
        real_profile._mark_real_profile_in_use()

    try:
        real_profile._close_real_profile_browser()
    finally:
        if held:
            browser_tool._real_profile_cdp_lock.release()

    if not closes:
        assert calls == [], f"{case} must keep the browser"
        return
    assert [call[0] for call in calls] == ["harness", "close-session", "chrome"]
    assert calls[0][1] == "http://127.0.0.1:1234", "only this lane's daemon may be reloaded"
    assert "cdp" not in browser_tool._real_profile_cdp_cache


def test_real_profile_closer_closes_after_a_live_session_goes_away(monkeypatch, tmp_path):
    """The janitor drops the session record when the user stops browsing; then the lane is idle."""
    _isolate(monkeypatch, tmp_path)
    calls = _record_teardown(monkeypatch)
    browser_tool._active_sessions["task-1"] = {"features": {"local": True, "real_profile": True}}

    real_profile._mark_real_profile_in_use()  # armed the way a real call arms it
    _age(REAL_PROFILE_KEY)
    lifecycle._run_due_idle_lane_closers()
    assert calls == [], "a live real-profile session must keep its browser"
    assert REAL_PROFILE_KEY in lifecycle._idle_lane_closers, "and the closer must stay armed"

    browser_tool._active_sessions.clear()
    browser_tool._real_profile_cdp_cache["cdp"] = "http://127.0.0.1:1234"
    _age(REAL_PROFILE_KEY)
    lifecycle._run_due_idle_lane_closers()

    assert [call[0] for call in calls] == ["harness", "close-session", "chrome"]


def test_real_profile_closer_rechecks_the_lane_after_taking_the_lock(monkeypatch, tmp_path):
    """A call that re-arms the lane between the sweep and the acquire must still win.

    Encoded by re-arming from inside the acquire itself, which is exactly where the unlocked check
    used to sit.
    """
    _isolate(monkeypatch, tmp_path)
    calls = _record_teardown(monkeypatch)
    browser_tool._real_profile_cdp_cache["cdp"] = "http://127.0.0.1:1234"

    class _LockProbe:
        def __init__(self, inner, on_acquire):
            self._inner, self._on_acquire = inner, on_acquire

        def acquire(self, blocking=True, timeout=-1):
            acquired = self._inner.acquire(blocking, timeout) if blocking else self._inner.acquire(False)
            if acquired:
                self._on_acquire()
            return acquired

        def release(self):
            self._inner.release()

    monkeypatch.setattr(browser_tool, "_real_profile_cdp_lock",
                        _LockProbe(browser_tool._real_profile_cdp_lock,
                                   real_profile._mark_real_profile_in_use))

    real_profile._close_real_profile_browser()

    assert calls == [], "the call that re-armed the lane under the lock keeps its browser"


def test_idle_close_and_its_deferrals_are_observable(monkeypatch, tmp_path, caplog):
    """A close that logs nothing cannot be verified from the running gateway, and neither can a deferral."""
    _isolate(monkeypatch, tmp_path)
    browser_tool._real_profile_cdp_cache["cdp"] = "http://127.0.0.1:1234"

    real_stop = browser_use_cli.stop_harness_daemons  # the recorder below stands in for the closer's use
    calls = _record_teardown(monkeypatch)
    browser_tool._active_sessions["task-1"] = {"features": {"local": True, "real_profile": True}}
    with caplog.at_level("INFO", logger="tools.browser_tool"):
        real_profile._close_real_profile_browser()
    assert calls == []
    assert any("deferred" in record.getMessage() for record in caplog.records)

    caplog.clear()
    browser_tool._active_sessions.clear()
    _age(REAL_PROFILE_KEY)  # the deferral re-armed the lane; its next window passes
    with caplog.at_level("INFO", logger="tools.browser_tool"):
        lifecycle._run_due_idle_lane_closers()
    assert [c[0] for c in calls] == ["harness", "close-session", "chrome"]
    assert any("closing the lane" in record.getMessage() for record in caplog.records)

    caplog.clear()
    reloaded = []
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: "/bin/true")
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: reloaded.append(env["BU_NAME"]))
    monkeypatch.setattr(browser_use_cli, "_driven_daemons", {"default": "127.0.0.1:1234"})
    with caplog.at_level("INFO", logger="tools.browser_use_cli"):
        real_stop("http://127.0.0.1:1234")
    assert reloaded == ["default"]
    assert any("harness daemon" in record.getMessage() for record in caplog.records)


def test_harness_close_leaves_a_lane_that_was_used_again_alone(monkeypatch, tmp_path):
    """A reload takes the browser off the call that owns it, so a re-armed lane is never reloaded."""
    _isolate(monkeypatch, tmp_path)
    reloaded = []
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: "/bin/true")
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: reloaded.append(env["BU_NAME"]))
    monkeypatch.setattr(browser_use_cli, "_driven_daemons", {"default": "127.0.0.1:1234"})

    browser_use_cli._mark_harness_in_use()   # a call arms the lane
    browser_use_cli._close_harness_lane()    # the sweep's already-selected closer then runs
    assert reloaded == [], "the call that re-armed the lane keeps its daemon"

    _age(browser_use_cli._HARNESS_LANE)
    lifecycle._run_due_idle_lane_closers()
    assert reloaded == ["default"], "an idle lane is reloaded once its window passes"
    assert browser_use_cli._driven_daemons == {}


def test_browser_exec_records_which_lane_its_daemon_serves(monkeypatch, tmp_path):
    """The endpoint recorded at dispatch is what later lets one lane be torn down on its own."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(browser_use_cli, "_mark_harness_in_use", lambda: None)
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: "/bin/true")
    monkeypatch.setattr(browser_use_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(browser_use_cli, "_workspace_dir", lambda task_id: None)
    monkeypatch.setattr(browser_use_cli, "_driven_daemons", {})

    def _fake_route(env, session, task_id, local, **kwargs):
        env["BU_CDP_WS"] = "ws://127.0.0.1:4321/devtools/browser/x"

    monkeypatch.setattr(browser_use_cli, "_route_backend", _fake_route)
    monkeypatch.setattr(browser_use_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: subprocess.CompletedProcess(cmd, 0, "ok", ""))

    browser_use_cli.browser_exec("print('x')")

    assert browser_use_cli._driven_daemons == {"default": "127.0.0.1:4321"}

def test_lane_state_carries_the_countdown_across_a_restart(monkeypatch, tmp_path):
    """A gateway restart must not restart the hour: the lane's own record holds the last use."""
    _isolate(monkeypatch, tmp_path)
    calls = _record_teardown(monkeypatch)
    copy_dir = str(tmp_path / "profile-copy")
    Path(copy_dir).mkdir()
    state = real_profile._lane_state_path()

    real_profile._write_lane_state(time.time() - 3900, copy_dir)
    assert Path(state).exists(), "the record must be written next to the lane's other state"
    assert real_profile._lane_last_used() is not None

    real_profile._declare_real_profile_lane()          # a fresh process, nothing used this lane
    lifecycle._run_due_idle_lane_closers()

    assert [call[0] for call in calls] == ["close-session", "chrome"], \
        "the record says the lane has been idle 65 minutes, so the closer must run"
    assert not Path(state).exists(), "a closed lane leaves no record claiming a live browser"


def test_lane_state_declines_to_close_a_lane_that_was_used_recently(monkeypatch, tmp_path):
    """The record is the countdown: used 10 minutes ago means the closer must not run yet."""
    _isolate(monkeypatch, tmp_path)
    calls = _record_teardown(monkeypatch)
    real_profile._write_lane_state(time.time() - 600, str(tmp_path))
    real_profile._declare_real_profile_lane()

    lifecycle._run_due_idle_lane_closers()

    assert calls == [], "an hour has not passed since the recorded use"


def test_reclaim_terminates_only_orphaned_chromes_on_our_own_copy_dir(monkeypatch, tmp_path):
    """Only a Chrome on OUR data dir whose parent is gone may be killed; a live owner's is left alone."""
    _isolate(monkeypatch, tmp_path)
    copy_dir = "/tmp/copy-dir"
    listing = "\n".join([
        f"111 1 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir={copy_dir}",
        f"222 999 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir={copy_dir}",
        "333 1 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir=/tmp/somewhere-else",
        f"444 1 /usr/bin/other-tool --user-data-dir={copy_dir}",
    ])
    monkeypatch.setattr(real_profile.subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 0, listing, ""))
    monkeypatch.setattr(real_profile, "_pid_alive", lambda pid: False)
    signalled = []
    monkeypatch.setattr(real_profile.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    reclaimed = real_profile._reclaim_orphaned_real_profile_chrome(copy_dir)

    assert reclaimed == 1
    assert signalled == [(111, real_profile.signal.SIGTERM)], signalled

def test_lane_declarations_run_once_without_recursing(monkeypatch, tmp_path):
    """Declaring a lane starts the cleanup thread, which runs the declarations — that must not recurse."""
    real_start = lifecycle._start_browser_cleanup_thread  # _isolate() stubs it; this test needs it
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", real_start)
    monkeypatch.setattr(lifecycle, "_declarations_done", False)
    monkeypatch.setattr(lifecycle, "_declared_lane_factories", [])

    class _AliveThread:  # so the real starter does not spawn a thread for this test
        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(browser_tool, "_cleanup_thread", _AliveThread())
    runs = []

    def _factory():  # a declaration that starts the thread, the way a real lane does
        runs.append(1)
        lifecycle.register_idle_lane_closer("lane", lambda: None, 60.0)

    lifecycle.add_idle_lane_declaration(_factory)
    lifecycle._start_browser_cleanup_thread()
    lifecycle._start_browser_cleanup_thread()

    assert runs == [1], "a declaration must run exactly once per process"

def test_ensure_idle_lane_cleanup_is_idempotent_and_starts_the_thread(monkeypatch, tmp_path):
    """The gateway's timer calls this: it must start the thread and never spawn a second one."""
    starts = []
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: starts.append(1))

    lifecycle.ensure_idle_lane_cleanup()
    lifecycle.ensure_idle_lane_cleanup()

    assert starts == [1, 1], "one call starts the thread; a second must be a cheap no-op"
