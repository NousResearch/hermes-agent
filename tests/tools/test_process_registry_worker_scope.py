"""The kanban worker's hard exit must not leak the terminal children it started.

``cli_single_query`` finishes its SIGTERM/SIGINT path with ``os._exit(0)``, which skips
``atexit`` and ``AIAgent.close()`` -> ``process_registry.kill_process``, so a background
terminal child (spawned with ``start_new_session=True``, i.e. its own process group) reparents
to init and keeps the worker's whole environment. ``terminate_worker_scope`` is the exit-path
kill. These tests exercise it against REAL spawned children and pin the two properties that
make it safe to call from a signal handler: coverage is registered leader-only (never a group
the worker does not own), and the kill is scoped to THIS worker's identities, so a sibling
worker's child is left running. The other half is the out-of-process parent-death supervisor: one
test drives a REAL worker subprocess and pins that a SIGKILLed worker — no handler, no ``atexit``,
no ``AIAgent.close()`` — still gets its child's group reaped.
"""

import os
import select
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from tools.process_registry import ProcessRegistry, _supervise_child_group

# Process groups and ``killpg`` are POSIX-only; on Windows the cleanup paths' ``taskkill /T``
# remains the backstop.
pytestmark = pytest.mark.platforms("posix")

WORKER_TASK = "t_worker_scope_test"
SIBLING_TASK = "t_sibling_worker_test"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _spawn(registry, task_id, session_key="", owner_task_id=""):
    return registry.spawn_local("sleep 60", task_id=task_id, session_key=session_key,
                                owner_task_id=owner_task_id)


def _group_alive(pgid):
    """Independent liveness probe: the raw syscall, not the registry's own helper."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_alive(pid):
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except OSError:
        return False


@pytest.fixture()
def registry():
    return ProcessRegistry()


@pytest.fixture()
def worker_env(monkeypatch):
    """Make this test process look like the worker that owns WORKER_TASK."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", WORKER_TASK)
    monkeypatch.delenv("HERMES_TASK_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)


def _cleanup(registry):
    for session in list(registry._running.values()):
        if not session.exited:
            registry.kill_process(session.id, source="test_cleanup")


def test_local_child_registers_its_own_leader_pgid(registry):
    """A spawned child leads its own group, and only a group LEADER is handed to the
    supervisor — a non-leader pgid would make the supervisor ``killpg`` unrelated
    processes after the worker dies."""
    session = _spawn(registry, WORKER_TASK)
    try:
        assert session.pid == os.getpgid(session.pid), "spawn must start a new session"
        assert session.supervised_pgid == session.pid
    finally:
        _cleanup(registry)


def test_terminate_worker_scope_kills_its_own_children_only(registry, worker_env):
    """The exit kill reaps this worker's background group and leaves a sibling worker's
    child alone (the hard exit must never fall back to a global kill_all)."""
    mine = _spawn(registry, WORKER_TASK)
    others = _spawn(registry, SIBLING_TASK)
    try:
        receipt = registry.terminate_worker_scope(budget=2.0)

        assert receipt["status"] == "ok"
        assert receipt["matched"] == 1
        assert receipt["killed"] == 1
        assert receipt["survivors"] == []
        assert not _group_alive(mine.pid), "the worker's own child group must be reaped"
        assert _pid_alive(others.pid), "a sibling worker's child must still be running"
        # The sibling was never a target, so it must carry neither the escalation nor the receipt
        # the exit kill stamps on what it actually signalled.
        assert others.id not in receipt.get("escalated", [])
        assert others.completion_reason != "killed"
        assert others.termination_source != "worker_exit"
        # The durable receipt says why it ended, not a bare exit.
        assert mine.completion_reason == "killed"
        assert mine.termination_source == "worker_exit"
    finally:
        _cleanup(registry)


def test_terminate_worker_scope_matches_inherited_session_key(registry, monkeypatch):
    """A subagent-created session inherits HERMES_SESSION_ID (as ``parent_session_id``) and its own
    task id, so neither the task ids nor ``session_key`` can be what scopes it — the session id has
    to bring it into scope, or it leaks on every hard exit. A session from another session context
    must stay out of scope even though it is in the same registry."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_TASK_ID", raising=False)
    # Spawned while some other session context is live: unrelated to this worker's identity.
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-other")
    unrelated = _spawn(registry, "t_other_child", session_key="sess-other")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-abc")
    inherited = _spawn(registry, "t_subagent_child", session_key="sess-abc")
    try:
        receipt = registry.terminate_worker_scope(budget=2.0)

        assert receipt["matched"] == 1
        assert not _group_alive(inherited.pid)
        assert _pid_alive(unrelated.pid)
    finally:
        _cleanup(registry)


def test_terminate_worker_scope_without_identity_signals_nothing(registry, monkeypatch):
    """No worker identity in env means "cannot scope", never "kill everything"."""
    for key in ("HERMES_KANBAN_TASK", "HERMES_TASK_ID", "HERMES_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)
    session = _spawn(registry, WORKER_TASK)
    try:
        receipt = registry.terminate_worker_scope(budget=2.0)

        assert receipt["status"] == "skipped"
        assert receipt["matched"] == 0
        assert _pid_alive(session.pid)
    finally:
        _cleanup(registry)


def test_supervise_child_group_refuses_unusable_pgids():
    """Never hand the supervisor a pgid that is not a real group we own."""
    assert _supervise_child_group(0) is False
    assert _supervise_child_group(1) is False
    assert _supervise_child_group(-1) is False


def test_scope_is_dropped_when_the_session_finishes(registry):
    """A finished session's pgid must not stay registered: the kernel recycles it, and the
    supervisor would then killpg a stranger after the worker dies."""
    session = _spawn(registry, WORKER_TASK)
    assert session.supervised_pgid == session.pid  # it was covered before the kill
    registry.kill_process(session.id, source="test_cleanup")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and session.supervised_pgid:
        time.sleep(0.05)
    assert session.supervised_pgid == 0
    assert not _group_alive(session.pid)


# A real worker process: spawn a background child through the registry, report the child's process
# group, then sit there waiting to be killed the way a worker actually dies.
_PARENT_DEATH_WORKER = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
from tools.process_registry import ProcessRegistry

session = ProcessRegistry().spawn_local("sleep 300", task_id="t_parent_death")
print(f"PGID={os.getpgid(session.pid)}", flush=True)
time.sleep(120)
"""


@pytest.mark.live_system_guard_bypass  # the child is spawned by a worker subprocess, so it is reparented out of the test subtree by design
def test_parent_death_reaps_a_child_group_when_the_worker_is_sigkilled():
    """SIGKILL and the OOM killer run no teardown at all — no handler, no ``atexit``, no
    ``AIAgent.close()`` — so the only thing that can reap a terminal child's process group is the
    out-of-process supervisor bound to the worker's own lifetime.

    This drives a REAL worker: it spawns a child through the registry, the test SIGKILLs it, and the
    child's process group must be gone within the supervisor's own grace window.
    """
    pgid = 0
    worker = subprocess.Popen(
        [sys.executable, "-c", _PARENT_DEATH_WORKER, str(REPO_ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(REPO_ROOT), start_new_session=True,
    )
    try:
        # The worker's stdout can carry interpreter warnings before the line we want; scan for the
        # PGID marker until the deadline instead of trusting the first line.
        line = ""
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not line.startswith("PGID="):
            ready, _, _ = select.select([worker.stdout], [], [], 0.5)
            if not ready:
                continue
            candidate = worker.stdout.readline()
            if not candidate:
                break
            if candidate.startswith("PGID="):
                line = candidate
        assert line.startswith("PGID="), f"worker never reported a pgid: {line!r}"
        pgid = int(line.strip().split("=", 1)[1])
        assert pgid > 1
        assert _group_alive(pgid), "precondition: the child's group is alive before the worker dies"

        worker.kill()  # SIGKILL: nothing in the worker runs after this
        worker.wait(timeout=10)

        # The supervisor SIGTERMs, waits its grace window, then SIGKILLs; allow that plus margin.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and _group_alive(pgid):
            time.sleep(0.1)
        assert not _group_alive(pgid), (
            f"process group {pgid} outlived the SIGKILLed worker: the parent-death supervisor did "
            f"not reap it")
    finally:
        if worker.poll() is None:
            worker.kill()
        if pgid > 1:
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pgid, signal.SIGKILL)