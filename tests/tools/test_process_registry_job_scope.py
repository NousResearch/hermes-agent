"""Real Linux job custody survives primary exit and reaps escaped children on Stop."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until


_PAYLOAD = r'''
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

root, mode = Path(sys.argv[1]), sys.argv[2]
deadline = time.monotonic() + 12

def record(name):
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    data = {'pid': os.getpid(), 'birth': int(fields[19])}
    target = root / (name + '.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(target)

def finite_wait(gate=None):
    event = threading.Event()
    while time.monotonic() < deadline:
        if gate is not None and gate.exists():
            return
        event.wait(min(0.02, max(0, deadline - time.monotonic())))

if mode == 'normal':
    print('PRIMARY-STDOUT', flush=True)
    print('PRIMARY-STDERR', file=sys.stderr, flush=True)
    sys.exit(23)

if mode == 'crash':
    record('primary')
    finite_wait()
    sys.exit(0)

if mode == 'detached':
    intermediate = os.fork()
    if intermediate == 0:
        os.setsid()
        if os.fork():
            os._exit(0)
        for fd in (0, 1, 2):
            replacement = os.open(os.devnull, os.O_RDWR)
            os.dup2(replacement, fd)
            if replacement > 2:
                os.close(replacement)
        record('detached')
        finite_wait(root / 'release')
        os._exit(0)
    os.waitpid(intermediate, 0)
    print('PRIMARY-EXITED', flush=True)
    sys.exit(7)

if mode == 'stop':
    def late_fork(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if os.fork() == 0:
            os.setsid()
            record('late')
            finite_wait()
            os._exit(0)
    signal.signal(signal.SIGTERM, late_fork)
    if os.fork() == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.setsid()
        record('stubborn')
        finite_wait()
        os._exit(0)
    record('primary')
    finite_wait()
'''


def _stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "birth": int(fields[19]), "state": fields[0]}


def _same_process(identity):
    current = _stat(identity["pid"])
    return current is not None and current["birth"] == identity["birth"]


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class _Job:
    def __init__(self, proc, receipt, root, session):
        self.proc, self.receipt, self.root, self.session = proc, receipt, root, session
        self.identity = wait_until(
            lambda: _read_json(receipt), timeout=5, what="initial native scope receipt"
        )
        assert self.identity["pid"] == proc.pid
        assert _same_process(self.identity)
        self.session.pid = proc.pid
        self.session.host_start_time = self.identity["birth"]

    def evidence(self, predicate):
        def matching():
            receipt = _read_json(self.receipt)
            return receipt if receipt and predicate(receipt) else None

        return wait_until(matching, timeout=5, what="native scope receipt transition")

    def fixture_identity(self, name):
        return wait_until(
            lambda: _read_json(self.root / f"{name}.json"),
            timeout=5, what=f"{name} fixture PID and kernel birth",
        )

    def stop(self):
        from tools.process_registry import ProcessRegistry
        from tools.process_registry_job import request_job_stop

        assert request_job_stop(ProcessRegistry(), self.session)


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = tmp_path / "payload.py"
    payload.write_text(_PAYLOAD)
    started = []

    def launch(mode, *, grace=0.4):
        from tools.process_registry_job import job_argv, job_receipt_path

        root = tmp_path / str(len(started))
        root.mkdir()
        session = SimpleNamespace(id=f"proc_jobscope{len(started)}", native_job_scope=False)
        argv = job_argv(
            session,
            [sys.executable, "-I", "-S", "-B", str(payload), str(root), mode],
            grace,
        )
        assert session.native_job_scope
        receipt = job_receipt_path(session.id)
        output = (root / "output").open("w+")
        proc = subprocess.Popen(
            argv,
            env=os.environ.copy(), cwd=root, stdout=output, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        started.append((proc, receipt, root, output))
        return _Job(proc, receipt, root, session)

    yield launch

    for proc, receipt, root, output in reversed(started):
        try:
            identity = _read_json(receipt)
            if proc.poll() is None and identity and _same_process(identity):
                os.kill(proc.pid, signal.SIGTERM)
            # Every payload has a 12-second upper bound, even on failed assertions.
            proc.wait(timeout=15)
            for path in root.glob("*.json"):
                identity = _read_json(path)
                assert not _same_process(identity), f"unreaped fixture: {identity}"
        finally:
            output.close()


@pytest.mark.platforms("linux")
def test_normal_primary_exit_preserves_output_and_native_quiescent_receipt(jobs):
    job = jobs("normal")
    receipt = job.evidence(lambda item: item["quiescent"])
    assert job.proc.wait(timeout=5) == 23
    assert {key: value for key, value in receipt.items() if key != 'stdout'} == {
        "version": 1, "pid": job.proc.pid, "birth": job.identity["birth"],
        "exit_code": 23, "quiescent": True,
    }
    output = (job.root / 'output').stat()
    assert receipt['stdout'] == {'device': output.st_dev, 'inode': output.st_ino, 'bytes': output.st_size}
    assert "PRIMARY-STDOUT" in (job.root / "output").read_text()
    assert "PRIMARY-STDERR" in (job.root / "output").read_text()


@pytest.mark.platforms("linux")
def test_primary_exit_retains_double_fork_setsid_child_until_it_is_reaped(jobs):
    job = jobs("detached")
    child = job.fixture_identity("detached")
    primary = job.evidence(lambda item: item["exit_code"] == 7)
    assert not primary["quiescent"]
    assert _same_process(child)
    assert _stat(child["pid"])["state"] != "Z"
    assert job.proc.poll() is None
    assert "PRIMARY-EXITED" in (job.root / "output").read_text()
    (job.root / "release").touch()
    final = job.evidence(lambda item: item["quiescent"])
    assert final["exit_code"] == 7
    assert job.proc.wait(timeout=5) == 7
    assert not _same_process(child), "receipt preceded kernel reaping"


@pytest.mark.platforms("linux")
def test_stop_reaps_term_ignoring_and_late_fork_children_without_touching_neighbor(jobs):
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_job import request_job_stop

    neighbor = jobs("detached")
    neighbor_child = neighbor.fixture_identity("detached")
    job = jobs("stop")
    primary = job.fixture_identity("primary")
    stubborn = job.fixture_identity("stubborn")
    assert _same_process(stubborn)
    wrong_birth = SimpleNamespace(**vars(job.session))
    wrong_birth.host_start_time += 1
    assert not request_job_stop(ProcessRegistry(), wrong_birth)
    assert _same_process(primary)
    job.stop()
    late = job.fixture_identity("late")
    receipt = job.evidence(lambda item: item["quiescent"])
    job.proc.wait(timeout=5)
    assert receipt["pid"] == job.identity["pid"]
    assert receipt["birth"] == job.identity["birth"]
    assert receipt["exit_code"] == -signal.SIGKILL, "payload expired instead of being stopped"
    for identity in (primary, stubborn, late):
        assert not _same_process(identity), f"scope receipt preceded reaping: {identity}"
    assert _same_process(neighbor.identity)
    assert _same_process(neighbor_child)
    assert _stat(neighbor_child["pid"])["state"] != "Z"
    (neighbor.root / "release").touch()
    neighbor.evidence(lambda item: item["quiescent"])


@pytest.mark.platforms("linux")
def test_missing_receipt_is_not_quiescence_proof_after_real_helper_exit(jobs):
    from tools.process_registry_job import job_quiescent

    job = jobs("normal")
    job.evidence(lambda item: item["quiescent"])
    job.proc.wait(timeout=5)
    assert job_quiescent(job.session)
    job.receipt.unlink()
    assert not job_quiescent(job.session)


@pytest.mark.platforms("linux")
def test_helper_crash_retains_nonquiescent_receipt_and_refuses_stop_success(jobs):
    import ctypes

    from tools.process_registry import ProcessRegistry
    from tools.process_registry_job import job_quiescent, request_job_stop

    # Adopt this fixture's orphan ourselves, so even the intentional helper crash
    # leaves no child or zombie for the host's init to collect.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    primary = None
    try:
        job = jobs("crash")
        primary = job.fixture_identity("primary")
        assert _same_process(primary)
        assert _same_process(job.identity)
        os.kill(job.identity["pid"], signal.SIGKILL)
        assert job.proc.wait(timeout=5) == -signal.SIGKILL
        receipt = _read_json(job.receipt)
        assert receipt["quiescent"] is False
        assert receipt["exit_code"] is None
        assert _same_process(primary)
        assert not job_quiescent(job.session)
        assert not request_job_stop(ProcessRegistry(), job.session)
        assert _same_process(primary), "failed helper verification signalled its former child"
    finally:
        try:
            if primary and _same_process(primary):
                os.kill(primary["pid"], signal.SIGKILL)
                waited, _ = os.waitpid(primary["pid"], 0)
                assert waited == primary["pid"]
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.platforms('linux')
def test_unresolved_native_scope_receipt_does_not_expire_with_output_history(jobs, monkeypatch):
    import time
    from agent import runtime_session_store
    from tools.process_registry_output_log import worker_output_log
    from tools.process_registry_results import RESULT_RETENTION_SECONDS
    from tools.process_registry_job import read_job_receipt
    job = jobs('detached')
    child = job.fixture_identity('detached')
    job.evidence(lambda row: row['exit_code'] == 7)
    cutoff = time.time() - RESULT_RETENTION_SECONDS - 60
    os.utime(job.receipt, (cutoff, cutoff))
    monkeypatch.setattr(runtime_session_store, 'is_worker_process', lambda: True)
    worker_output_log('proc_new_output_history')
    assert read_job_receipt(job.session)['quiescent'] is False
    assert _same_process(child)
    job.stop()
    assert job.evidence(lambda row: row['quiescent'])['exit_code'] == 7
    assert not _same_process(child)


@pytest.mark.platforms('linux')
def test_zero_grace_preserves_no_escalation_and_reports_unknown(jobs, monkeypatch):
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_job import job_quiescent, request_job_stop
    job = jobs('stop', grace=0)
    primary = job.fixture_identity('primary')
    stubborn = job.fixture_identity('stubborn')
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_daemon_term_grace_seconds', lambda: 0)
    try:
        assert not request_job_stop(registry, job.session)
        late = job.fixture_identity('late')
        assert not job_quiescent(job.session) and _same_process(primary)
    finally:
        for name in ('primary', 'stubborn', 'late'):
            identity = _read_json(job.root / (name + '.json'))
            if identity and _same_process(identity):
                os.kill(identity['pid'], signal.SIGKILL)
        job.evidence(lambda row: row['quiescent'])
    assert not _same_process(stubborn) and not _same_process(late)


@pytest.mark.platforms('linux')
def test_stop_waits_for_ready_receipt_within_existing_native_deadline(jobs, monkeypatch):
    from tools import process_registry_job as native
    from tools.process_registry import ProcessRegistry
    job = jobs('crash')
    job.fixture_identity('primary')
    read = native.read_job_receipt
    reads = []

    def delayed_read(session):
        reads.append(session)
        # The captured launch is real; early receipt reads can precede ready.
        # No positive receipt or identity is fabricated at this boundary.
        return None if len(reads) <= 2 else read(session)

    monkeypatch.setattr(native, 'read_job_receipt', delayed_read)
    assert native.request_job_stop(ProcessRegistry(), job.session)
    assert len(reads) > 2
    job.proc.wait(timeout=5)
    assert native.job_quiescent(job.session)
    assert not _same_process(job.fixture_identity('primary'))
