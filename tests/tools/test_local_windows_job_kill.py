"""Windows: terminal commands must not leave MSYS pipeline stages running after a kill.

Git Bash emulates fork/exec through short-lived stub processes, so a pipeline stage like
``find / ... | head`` ends up with a dead parent PID and ``taskkill /T`` (which walks parent
PIDs) never reaches it; neither does a psutil descendant snapshot. Those orphans used to keep
scanning the disk at full CPU until Hermes exited. LocalEnvironment now contains each command
in its own job object, terminates the job on timeout/interrupt, and hands it to the process
registry when the command yields to background (#69033, follow-up named on #69076).
"""

import time
import types
import uuid

import pytest

pytestmark = pytest.mark.platforms("windows")


def _alive(tag):
    """Probe processes carrying ``tag`` (name filter first: reading every cmdline is slow)."""
    import psutil
    out = []
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info["name"] or "").lower() in ("find.exe", "head.exe", "bash.exe") \
                    and tag in " ".join(p.cmdline()):
                out.append(p)
        except (psutil.Error, OSError):
            pass
    return out


def _wait_gone(tag, seconds=5.0):
    deadline = time.monotonic() + seconds
    while _alive(tag) and time.monotonic() < deadline:
        time.sleep(0.2)
    return _alive(tag)


def _kill_all(tag):
    import psutil
    for p in _alive(tag):
        try:
            p.kill()
        except (psutil.Error, OSError):
            pass


def _spawn_pipeline(tag):
    """A _run_bash-style spawn of an MSYS pipeline whose stages outlive their stub parent."""
    import subprocess
    from hermes_cli._subprocess_compat import windows_hide_flags
    from tools.environments.local import _find_bash, _popen_in_job_windows

    proc = _popen_in_job_windows(
        [_find_bash(), "-c", f"find /c/Windows -name {tag} 2>/dev/null | head -1; echo never"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, creationflags=windows_hide_flags())
    deadline = time.monotonic() + 10
    while not _alive(tag) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _alive(tag), "probe pipeline never started"
    return proc


def test_timeout_kills_msys_orphaned_pipeline_stage(tmp_path):
    from tools.environments.local import LocalEnvironment

    tag = f"zz_hermes_jobkill_{uuid.uuid4().hex[:8]}"
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        result = env.execute(f"find /c/Windows -name {tag} 2>/dev/null | head -1; echo never", timeout=3)
        assert result.get("returncode", result.get("exit_code")) == 124
        survivors = _wait_gone(tag)
        assert not survivors, f"orphaned pipeline stage survived timeout: {[p.pid for p in survivors]}"
    finally:
        _kill_all(tag)
        env.cleanup()


def test_registry_kill_reaches_orphans_of_a_yielded_command():
    """A yielded foreground command keeps its job; ProcessRegistry._signal_kill terminates it."""
    from tools.process_registry import ProcessRegistry

    tag = f"zz_hermes_jobkill_{uuid.uuid4().hex[:8]}"
    proc = _spawn_pipeline(tag)
    try:
        assert getattr(proc, "_hermes_job", None) is not None
        session = types.SimpleNamespace(_pty=None, process=proc, host_start_time=None)
        registry = object.__new__(ProcessRegistry)  # this branch of _signal_kill only reads the session
        try:
            ProcessRegistry._signal_kill(registry, session, False)
        except OSError:
            pass  # the guarded taskkill may refuse without a start time; the job kill already ran
        survivors = _wait_gone(tag)
        assert not survivors, f"registry kill missed MSYS orphans: {[p.pid for p in survivors]}"
    finally:
        _kill_all(tag)


def test_release_drops_the_job_without_killing():
    """Normal completion releases the handle only (no KILL_ON_JOB_CLOSE), so `cmd &` survives."""
    from tests.waiting import wait_until
    from tools.environments.local import _release_command_job

    tag = f"zz_hermes_jobkill_{uuid.uuid4().hex[:8]}"
    proc = _spawn_pipeline(tag)
    try:
        _release_command_job(proc)
        assert getattr(proc, "_hermes_job", None) is None
        # A kill-on-close job would take the tree down here; this must time out instead.
        with pytest.raises(AssertionError):
            wait_until(lambda: not _alive(tag), timeout=1.0, what="job release killed its processes")
        assert _alive(tag), "releasing the job must not terminate its processes"
    finally:
        _kill_all(tag)
        proc.kill()
