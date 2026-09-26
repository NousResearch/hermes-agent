"""Kill-on-job-close for terminal bash trees and background pipe sessions."""

import ctypes
import gc
import io
import logging
import os
import subprocess
import sys
import time

import psutil
import pytest

from hermes_cli.local_runtime import processes


def _wait(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.03)
    return bool(predicate())


def _same_process(pid: int, created: float) -> bool:
    try:
        proc = psutil.Process(pid)
        if proc.create_time() != created:
            return False
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def test_spawn_contained_without_job_is_plain_popen(monkeypatch):
    seen = {}

    def popen(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return "proc"

    def explode():
        raise AssertionError("job created")

    monkeypatch.setattr(processes.subprocess, "Popen", popen)
    monkeypatch.setattr(processes, "_WindowsJob", explode)
    kwargs = {"stdout": subprocess.PIPE, "start_new_session": True, "cwd": "C:/work"}
    assert processes.spawn_contained(["echo", "hi"], job=None, **kwargs) == "proc"
    assert seen["cmd"] == ["echo", "hi"]
    assert seen["kwargs"] == kwargs
    assert (seen["kwargs"].get("creationflags", 0) & 0x4) == 0


@pytest.mark.skipif(sys.platform == "win32", reason="process-lifetime job is Windows-only")
def test_kill_on_exit_job_is_none_off_windows(monkeypatch):
    def explode():
        raise AssertionError("job created")

    monkeypatch.setattr(processes, "_WindowsJob", explode)
    assert processes.kill_on_exit_job() is None


@pytest.mark.windows_only
def test_job_creation_failure_spawns_and_warns_once(monkeypatch, caplog):
    calls = {"n": 0}

    def fail_job():
        calls["n"] += 1
        raise ctypes.WinError(5)

    flags = []

    def popen(cmd, **kwargs):
        flags.append(kwargs.get("creationflags", 0))
        return object()

    monkeypatch.setattr(processes, "_kill_on_exit_job", None)
    monkeypatch.setattr(processes, "_kill_on_exit_failed", False)
    monkeypatch.setattr(processes, "_WindowsJob", fail_job)
    monkeypatch.setattr(processes.subprocess, "Popen", popen)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.local_runtime.processes"):
        for _ in range(2):
            processes.spawn_contained(
                ["x"], job=processes.kill_on_exit_job(), creationflags=0x08000000)
    assert calls["n"] == 1
    assert flags == [0x08000000, 0x08000000]
    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "kill-on-exit job creation failed" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "WinError" in warnings[0].getMessage()


@pytest.mark.windows_only
def test_assign_failure_resumes_and_warns_once(monkeypatch, caplog):
    order = []

    class Job:
        def assign(self, proc):
            order.append("assign")
            raise ctypes.WinError(5)

        def close(self):
            order.append("close")

    class Proc:
        def __init__(self, pid):
            self.pid = pid

        def resume(self):
            order.append("resume")

    flags = []

    def popen(cmd, **kwargs):
        flags.append(kwargs.get("creationflags", 0))
        proc = Proc(0)
        proc.pid = 4242
        return proc

    monkeypatch.setattr(processes, "_assign_warned", False)
    monkeypatch.setattr(processes.subprocess, "Popen", popen)
    monkeypatch.setattr(processes.psutil, "Process", Proc)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.local_runtime.processes"):
        first = processes.spawn_contained(["x"], job=Job(), creationflags=0x08000000)
        second = processes.spawn_contained(["x"], job=Job(), creationflags=0x08000000)
    assert first.pid == 4242 and second.pid == 4242
    assert order == ["assign", "resume", "assign", "resume"]
    assert flags == [0x08000004, 0x08000004]
    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "kill-on-exit job assign failed" in r.getMessage()
    ]
    assert len(warnings) == 1


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_closing_job_kills_grandchild(tmp_path):
    pidfile = tmp_path / "grand.pid"
    ready = tmp_path / "ready"
    code = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "pidfile, ready = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "pidfile.write_text(str(child.pid))\n"
        "ready.write_text('1')\n"
        "time.sleep(30)\n"
    )
    job = processes._WindowsJob()
    proc = None
    gpid = None
    created = None
    try:
        proc = processes.spawn_contained(
            [sys.executable, "-c", code, str(pidfile), str(ready)],
            job=job,
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert _wait(ready.exists, 2.0), "grandchild did not start"
        gpid = int(pidfile.read_text().strip())
        created = psutil.Process(gpid).create_time()
        assert _same_process(gpid, created)
        job.close()
        assert _wait(lambda: not _same_process(gpid, created), 2.0), "grandchild survived job close"
    finally:
        try:
            job.close()
        except Exception:
            pass
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        if gpid is not None and created is not None and _same_process(gpid, created):
            try:
                psutil.Process(gpid).kill()
            except psutil.Error:
                pass


@pytest.mark.windows_only
def test_dropping_run_bash_popen_keeps_grandchild(tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment

    job = processes._WindowsJob()
    def owned_job():
        return job

    # _run_bash -> _spawn_kill_on_exit imports this name on each call.
    monkeypatch.setattr(processes, "kill_on_exit_job", owned_job)
    pidfile = tmp_path / "grand.pid"
    ready = tmp_path / "ready"
    code = (
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        f"pidfile = Path({str(pidfile)!r})\n"
        f"ready = Path({str(ready)!r})\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "pidfile.write_text(str(child.pid))\n"
        "ready.write_text('1')\n"
    )
    monkeypatch.setattr("tools.environments.local._find_bash", lambda: sys.executable)
    monkeypatch.setattr(
        "tools.environments.local._make_run_env", lambda env: os.environ.copy())
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    proc = None
    gpid = None
    created = None
    try:
        proc = env._run_bash(code)
        assert _wait(ready.exists, 2.0), "grandchild did not start"
        proc.wait(timeout=2)
        gpid = int(pidfile.read_text().strip())
        created = psutil.Process(gpid).create_time()
        held = proc
        proc = None
        del held
        gc.collect()
        time.sleep(0.3)
        assert _same_process(gpid, created), "dropping the foreground Popen killed the grandchild"
        job.close()
        assert _wait(lambda: not _same_process(gpid, created), 5.0), "grandchild survived job close"
        assert processes._kill_on_exit_job is not job  # the test-owned job never leaks into the singleton
    finally:
        try:
            job.close()
        except Exception:
            pass
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        if gpid is not None and created is not None and _same_process(gpid, created):
            try:
                psutil.Process(gpid).kill()
            except psutil.Error:
                pass


@pytest.mark.windows_only
def test_spawn_local_session_job_closes_on_kill(tmp_path, monkeypatch):
    import tools.process_registry as pr

    closed = []
    captured = {}

    class FakeJob:
        def close(self):
            closed.append(1)

    class FakeProc:
        def __init__(self):
            self.pid = 2**31 - 1
            self.stdout = io.StringIO("")
            self.stderr = None
            self.stdin = None
            self.returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_spawn(cmd, *, job, **kwargs):
        captured["job"] = job
        return FakeProc()

    monkeypatch.setattr(processes, "_WindowsJob", FakeJob)
    monkeypatch.setattr(processes, "spawn_contained", fake_spawn)
    monkeypatch.setattr(pr.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(pr.os, "kill", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr.ProcessRegistry, "_safe_host_start_time", staticmethod(lambda pid: None))
    registry = pr.ProcessRegistry()
    monkeypatch.setattr(
        registry, "_scope_argv", lambda *args, **kwargs: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(
        registry, "_track_started",
        lambda session, reader_target, reader_name, extra_args=(): registry._running.__setitem__(
            session.id, session))
    monkeypatch.setattr(registry, "_post_kill_survivors", lambda session: [])
    monkeypatch.setattr(registry, "_move_to_finished", lambda session: True)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda *args, **kwargs: None)

    session = registry.spawn_local("echo hi", cwd=str(tmp_path))
    assert isinstance(session.job, FakeJob)
    assert captured["job"] is session.job
    assert closed == []
    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert closed
