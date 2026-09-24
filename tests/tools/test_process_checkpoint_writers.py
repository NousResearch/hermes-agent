"""Every process on a profile shares processes.json; recovery must only take jobs whose writer is gone."""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from tools.process_registry import ProcessRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

# A Hermes process (gateway turn, CLI, cron worker) that starts one background job and either
# dies without any shutdown (crash) or stays up holding it.
_WRITER = textwrap.dedent('''
    import json, os, subprocess, sys
    from tools.process_registry import process_registry
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
    session = process_registry.adopt_local(job, command="sleep 600", cwd=None, task_id=sys.argv[1])
    print(json.dumps({"session_id": session.id, "pid": job.pid, "start": session.host_start_time}), flush=True)
    if sys.argv[2] == "crash":
        os._exit(137)
    sys.stdin.readline()
''')


def _alive(pid):
    return ProcessRegistry._is_host_pid_alive(pid)


def _wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _start_writer(task_id, mode):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.Popen([sys.executable, "-c", _WRITER, task_id, mode], cwd=REPO_ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    line = []
    reader = threading.Thread(target=lambda: line.append(proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(60)
    assert line and line[0], f"{mode} writer printed nothing (rc={proc.poll()})"
    return proc, json.loads(line[0])


@pytest.mark.live_system_guard_bypass
def test_gateway_recovery_takes_only_dead_writers_jobs_and_its_shutdown_spares_live_ones():
    crashed_writer = live_writer = None
    jobs = []
    try:
        crashed_writer, crashed = _start_writer("gateway-turn", "crash")
        jobs.append(crashed)
        assert crashed_writer.wait(timeout=30) == 137
        live_writer, live = _start_writer("cli-turn", "stay")
        jobs.append(live)

        gateway = ProcessRegistry()  # the restarted gateway: recovery at startup, kill_all at shutdown
        gateway.recover_from_checkpoint()
        adopted = {s for s in (crashed["session_id"], live["session_id"]) if gateway.get(s) is not None}
        gateway.kill_all()

        assert _alive(live["pid"]), "gateway shutdown killed a job another live process still owns"
        assert adopted == {crashed["session_id"]}
        assert _wait_until(lambda: not _alive(crashed["pid"]), timeout=10)
    finally:
        for job in jobs:  # the start time makes this a no-op on a recycled PID
            ProcessRegistry._terminate_host_pid(job["pid"], job["start"])
        for writer in (crashed_writer, live_writer):
            if writer is not None and writer.poll() is None:
                writer.stdin.close()
                writer.wait(timeout=30)


def test_detached_finish_never_replaces_the_producers_saved_result():
    """A detached session (here: a second registry recovering the producer's entry, as after an
    in-place exec) has no output or exit code, so its finish must leave the real receipt alone."""
    job = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('RESULT-' + sys.stdin.readline().strip(), flush=True)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    try:
        producer = ProcessRegistry()
        session = producer.adopt_local(job, command="review", cwd=None, task_id="cli-turn",
                                       notify_on_complete=False)
        adopter = ProcessRegistry()
        assert adopter.recover_from_checkpoint() == 1

        job.stdin.write("OK-42\n")
        job.stdin.close()
        assert session._completion_event.wait(timeout=30)
        assert adopter.get(session.id).exited

        from hermes_constants import get_hermes_home
        receipt = json.loads((get_hermes_home() / "logs" / "process-results" / f"{session.id}.json")
                             .read_text(encoding="utf-8"))
        assert receipt["exit_code"] == 0
        assert "RESULT-OK-42" in receipt["output"]
    finally:
        if job.poll() is None:
            job.kill()
        job.wait(timeout=30)
