"""Foreground Git Bash timeouts reap jobs orphaned by fork/exec (#69033)."""

import shlex
import subprocess
import sys
from pathlib import Path
import uuid

import psutil
import pytest

from hermes_cli._subprocess_compat import windows_hide_flags
from tests.waiting import wait_until
from tools.environments import local


def _matching_processes(name, argument):
    matches = []
    for child in psutil.process_iter(["name", "cmdline"]):
        try:
            if (child.info["name"] or "").lower() == name and argument in (child.info["cmdline"] or []):
                matches.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def _reap_processes(name, argument, bin_dir):
    survivors = {child.pid for child in _matching_processes(name, argument)}
    if survivors:
        output = subprocess.run(
            [str(bin_dir / "ps.exe"), "-W", "-l"], capture_output=True,
            encoding="utf-8", errors="replace", timeout=2,
            creationflags=windows_hide_flags(), check=True,
        ).stdout.splitlines()
        header = output[0].split()
        for line in output[1:]:
            row = line.split()
            if row[0] in ("I", "O", "S"):
                row = row[1:]
            if int(row[header.index("WINPID")]) in survivors:
                subprocess.run(
                    [str(bin_dir / "kill.exe"), "-KILL", row[header.index("PID")]],
                    capture_output=True, encoding="utf-8", errors="replace",
                    timeout=2, creationflags=windows_hide_flags(), check=True,
                )
    wait_until(lambda: not _matching_processes(name, argument), timeout=5,
               what=f"test process {name} {argument} to be reaped")


@pytest.fixture
def git_bash_bin():
    try:
        bash = Path(local._find_bash())
    except RuntimeError:
        pytest.skip("Git Bash is not installed")
    bin_dir = bash.parent
    if bin_dir.parent.name.lower() != "usr":
        bin_dir = bin_dir.parent / "usr" / "bin"
    if not all((bin_dir / name).is_file() for name in ("bash.exe", "ps.exe", "kill.exe", "sleep.exe")):
        pytest.skip("Git Bash MSYS utilities are not installed")
    return bin_dir


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("topology", ["pipeline", "background-and-pipeline", "exec-sh", "orphaned-pipeline"])
def test_timeout_kills_command_sleep_jobs(tmp_path, git_bash_bin, topology):
    count = 1 if topology in ("pipeline", "exec-sh") else 2
    durations = [f"58.69076{uuid.uuid4().int % 1_000_000:06d}" for _ in range(count)]
    command = {
        "pipeline": f"echo x; sleep {durations[0]} | cat",
        "background-and-pipeline": f"echo x; sleep {durations[0]} & sleep {durations[-1]} | cat",
        "exec-sh": f"exec /usr/bin/sh -c 'sleep {durations[0]} | cat'",
        "orphaned-pipeline": f"bash -c 'sleep {durations[0]} | cat &' ; sleep {durations[-1]} | cat",
    }[topology]
    print("default bash:", local._find_bash(), "unique sleep arguments:", durations)
    environment = local.LocalEnvironment(cwd=str(tmp_path), timeout=4)
    proc = None
    try:
        proc = environment._run_bash(command)
        for duration in durations:
            wait_until(lambda: _matching_processes("sleep.exe", duration), timeout=4,
                       what=f"command sleep {duration} to start")
        print("launcher:", proc.pid, "direct children:",
              [child.pid for child in psutil.Process(proc.pid).children(recursive=False)])
        result = environment._wait_for_process(proc, timeout=4)
        assert result["returncode"] == 124
        assert result["hermes_timed_out"] is True
        assert "timed out" in result["output"]
        for duration in durations:
            wait_until(lambda: not _matching_processes("sleep.exe", duration), timeout=4,
                       what=f"timed-out command sleep {duration} to exit")
    finally:
        try:
            for duration in durations:
                _reap_processes("sleep.exe", duration, git_bash_bin)
        finally:
            if proc is not None:
                environment._kill_process(proc)
                proc.wait(timeout=5)
                proc.stdout.close()
            environment.cleanup()


@pytest.mark.platforms("windows")
def test_timeout_kills_native_python_child_and_grandchild(tmp_path, git_bash_bin):
    durations = [f"58.69076{uuid.uuid4().int % 1_000_000:06d}" for _ in range(2)]
    grandchild = f"import time; time.sleep({durations[0]})"
    child = ("import subprocess,sys,time; "
             f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
             f"time.sleep({durations[1]})")
    scripts = [child, grandchild]
    python = shlex.quote(Path(sys.executable).as_posix())
    # Keep bash alive so both native generations remain in its Windows tree.
    command = f"echo x; {python} -c {shlex.quote(child)}; :"
    print("default bash:", local._find_bash(), "unique Python sleeps:", durations)
    environment = local.LocalEnvironment(cwd=str(tmp_path), timeout=4)
    proc = None
    try:
        proc = environment._run_bash(command)
        for script in scripts:
            wait_until(lambda: _matching_processes("python.exe", script), timeout=4,
                       what="native command Python to start")
        print("native Python PIDs:",
              [[child.pid for child in _matching_processes("python.exe", script)] for script in scripts])
        result = environment._wait_for_process(proc, timeout=4)
        assert result["returncode"] == 124
        assert result["hermes_timed_out"] is True
        assert "timed out" in result["output"]
        for script in scripts:
            wait_until(lambda: not _matching_processes("python.exe", script), timeout=4,
                       what="timed-out native Python to exit")
    finally:
        try:
            for script in scripts:
                for child in _matching_processes("python.exe", script):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                wait_until(lambda: not _matching_processes("python.exe", script), timeout=5,
                           what="native test Python to be reaped")
        finally:
            if proc is not None:
                environment._kill_process(proc)
                proc.wait(timeout=5)
                proc.stdout.close()
            environment.cleanup()
