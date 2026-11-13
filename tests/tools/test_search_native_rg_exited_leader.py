"""search_files' native rg lane keeps its output when rg is exiting as it is spawned (#123930).

macOS answers ``getpgid`` with ESRCH for a process that has begun exiting, while
``poll()`` still reports it running. A one-file content search is the fastest rg run
there is, so rg could already be exiting when the lane read its group back right after
``Popen``; the read failed, no group was recorded, the bounded kill's own ``getpgid``
failed the same way and re-raised, and ``[Errno 3] No such process`` replaced the lines
rg had already printed. The group is rg's pid by construction (``start_new_session``),
so the lane records that instead of reading it back.

The window lasts microseconds for a real rg, so the child here holds it open with the
kernel's own mechanism: it exits holding the last descriptor of a pty whose output
nobody reads, and closing a tty waits for its output to drain. ``Popen`` is wrapped only
so it returns once the kernel reports the child exiting, and the teardown only so the
pty is drained once it has run; the lookups, the signals and the race are real. Both
states here are macOS kernel behaviour, so these tests run there.
"""

import os
import shlex
import subprocess
import sys
import textwrap
import threading
import time

import pytest

pytestmark = pytest.mark.platforms("macos")

_CHILD = textwrap.dedent("""
    import os, sys
    tty = os.open(sys.argv[1], os.O_WRONLY | os.O_NOCTTY)
    os.write(tty, b"held until the master reads it\\n")
    os.write(1, b"m1\\nm2\\nm3\\nm4\\n")
    os._exit(0)  # closing the pty's last descriptor waits for that output to drain
""")


def _exiting(proc) -> bool:
    """Begun exiting, not yet a zombie: ``getpgid`` refuses it while ``waitpid`` still waits."""
    try:
        os.getpgid(proc.pid)
    except ProcessLookupError:
        return proc.poll() is None
    return False


def test_native_rg_lane_keeps_its_output_when_the_child_is_already_exiting(tmp_path, monkeypatch):
    from hermes_agent.tools.environments.local import LocalEnvironment
    from hermes_agent.tools.file_operations import ShellFileOperations

    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    master, slave = os.openpty()
    slave_name = os.ttyname(slave)
    os.close(slave)  # the child's descriptor is then the last one, so its close drains
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    spawned = []
    real_popen = subprocess.Popen

    class ReturnsOnceExiting(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spawned.append(self)
            deadline = time.monotonic() + 10
            while not _exiting(self):
                assert self.returncode is None, "child finished exiting before the window was held"
                assert time.monotonic() < deadline, "child never entered the exiting state"
                time.sleep(0.001)

    import hermes_agent.tools.environments.local as local_mod
    real_kill = local_mod._kill_process_group_posix
    teardown_done = threading.Event()

    def kill_then_release(proc):
        try:
            real_kill(proc)
        finally:
            teardown_done.set()

    def release() -> None:
        # The child stays exiting until the lane's teardown has run against it; then its
        # pty output is drained so it can finish and the lane's own wait() returns.
        teardown_done.wait(timeout=20)
        while spawned and spawned[0].poll() is None:
            try:
                os.read(master, 4096)
            except OSError:
                break

    releaser = threading.Thread(target=release, daemon=True)
    argv = [shlex.quote(sys.executable), shlex.quote(str(script)), shlex.quote(slave_name)]
    monkeypatch.setattr(subprocess, "Popen", ReturnsOnceExiting)
    monkeypatch.setattr(local_mod, "_kill_process_group_posix", kill_then_release)
    releaser.start()
    try:
        result = ops._run_rg_native(argv, 3, timeout=30)
    finally:
        monkeypatch.setattr(subprocess, "Popen", real_popen)
        teardown_done.set()
        releaser.join()
        os.close(master)
        for proc in spawned:
            proc.wait(timeout=10)
    assert (result.exit_code, result.stdout) == (0, "m1\nm2\nm3\n"), result


def test_native_rg_lane_keeps_its_output_when_the_child_is_a_zombie_at_the_kill(tmp_path, monkeypatch):
    """rg finished between the lane's ``poll()`` and its group kill: an unreaped zombie. macOS
    then answers ``getpgid`` with ESRCH and ``killpg`` on the zombie-only group with EPERM
    (#116855, #104696); neither may replace the drained lines with a tool error."""
    psutil = pytest.importorskip("psutil")
    from hermes_agent.tools.environments.local import LocalEnvironment
    from hermes_agent.tools.file_operations import ShellFileOperations

    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    real_popen = subprocess.Popen
    spawned = []

    class ZombieOnFirstPoll(real_popen):
        """The race, held open: the first ``poll()`` reports rg running, and by then it is a zombie."""
        _first = True

        def poll(self):
            if ZombieOnFirstPoll._first:
                ZombieOnFirstPoll._first = False
                spawned.append(self)
                deadline = time.monotonic() + 10
                while psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE:  # never reaps
                    assert time.monotonic() < deadline, "child never became a zombie"
                    time.sleep(0.005)
                return None
            return super().poll()

    monkeypatch.setattr(subprocess, "Popen", ZombieOnFirstPoll)
    try:
        result = ops._run_rg_native(["printf", shlex.quote("m1\\nm2\\nm3\\n")], 50, timeout=20)
    finally:
        monkeypatch.setattr(subprocess, "Popen", real_popen)
    assert spawned, "the lane never reached its kill check; the race was not exercised"
    assert (result.exit_code, result.stdout) == (0, "m1\nm2\nm3\n"), result
