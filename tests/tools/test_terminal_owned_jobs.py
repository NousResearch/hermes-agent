"""POSIX: terminal commands and background jobs die with the Hermes process that started them.

The owners are real host processes using the real ``ProcessRegistry.spawn_local`` and
``LocalEnvironment.execute``. One owner is SIGKILLed, so no Python cleanup of ours runs; the other is the
live-owner control. persist_on_release jobs (#41225) and the jobs of a checkpoint-recovering host must
outlive the owner.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")

_OWNER = r'''
import json, os, sys, threading, time
from pathlib import Path
root, role = Path(sys.argv[1]), sys.argv[2]
from tools.process_registry import process_registry
from tools.environments.local import LocalEnvironment
if role.startswith("recovering"):
    process_registry.recover_from_checkpoint()
ids = {}
for name, kwargs in (("job", {}), ("pty_job", {"use_pty": True}), ("persisted_job", {"persist_on_release": True})):
    marker = root / f"{role}.{name}"
    # A job that ignores SIGHUP (nohup'd servers): a PTY hangup alone does not stop it.
    s = process_registry.spawn_local(f"trap '' HUP; sleep 120 & echo $! > {marker}; wait", cwd=str(root), **kwargs)
    wait_until = time.time() + 30
    while not marker.exists() and time.time() < wait_until:
        time.sleep(0.05)
    ids[name] = [s.pid, int(marker.read_text().strip())]
env = LocalEnvironment(cwd=str(root), timeout=600)
fg = root / f"{role}.fg"
threading.Thread(target=env.execute, args=(f"sleep 120 & echo $! > {fg}; wait",), daemon=True).start()
wait_until = time.time() + 30
while not fg.exists() and time.time() < wait_until:
    time.sleep(0.05)
ids["foreground"] = [int(fg.read_text().strip())]
out = {}
for name, pids in ids.items():
    out[name] = [{"pid": pid, "created": __import__("psutil").Process(pid).create_time()} for pid in pids]
(root / (role + ".tmp")).write_text(json.dumps(out))
os.replace(root / (role + ".tmp"), root / (role + ".json"))
while True:
    time.sleep(1)
'''


def _alive(identity):
    """Zombie-aware: a killed non-child lingers as a zombie until its reaper collects it."""
    try:
        proc = psutil.Process(identity["pid"])
        return proc.create_time() == identity["created"] and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _read(path):
    wait_until(path.exists, timeout=60, what=f"owner receipt {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_owners(tmp_path, roles):
    script = tmp_path / "disposable owner.py"
    script.write_text(_OWNER, encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    env.pop("_HERMES_GATEWAY", None)
    log = (tmp_path / "owners.log").open("w", encoding="utf-8")
    owners = {role: subprocess.Popen([sys.executable, str(script), str(tmp_path), role], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
              for role in roles}
    return owners, log


def _cleanup(owners, log, ids):
    for identity in (i for group in ids.values() for job in group.values() for i in job):
        if _alive(identity):
            psutil.Process(identity["pid"]).kill()
    for owner in owners.values():
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=10)
    log.close()


# Jobs reparent to init once their owner dies, so cleaning them up leaves this test's subtree; real
# signal delivery to real orphans is the point of the test.
@pytest.mark.live_system_guard_bypass
def test_owner_sigkill_takes_its_jobs_and_commands_not_a_live_owners(tmp_path):
    owners, log = _run_owners(tmp_path, ("control", "victim"))
    ids = {}
    try:
        ids = {role: _read(tmp_path / f"{role}.json") for role in owners}
        owners["victim"].kill()  # SIGKILL: no atexit, no finally, nothing of ours runs
        owners["victim"].wait(timeout=10)

        owned = {name: job for name, job in ids["victim"].items() if name != "persisted_job"}
        wait_until(lambda: not any(_alive(i) for job in owned.values() for i in job), timeout=15,
                   what="the dead owner's background jobs and foreground command to exit")
        assert all(_alive(i) for i in ids["victim"]["persisted_job"]), (
            "a persist_on_release job died with its owner", ids["victim"]["persisted_job"])
        assert all(_alive(i) for job in ids["control"].values() for i in job), (
            "a live owner's jobs were killed", ids["control"])
    finally:
        _cleanup(owners, log, ids)


@pytest.mark.live_system_guard_bypass
def test_a_checkpoint_recovering_hosts_jobs_outlive_its_crash(tmp_path):
    owners, log = _run_owners(tmp_path, ("recovering",))
    ids = {}
    try:
        ids = {"recovering": _read(tmp_path / "recovering.json")}
        owners["recovering"].kill()
        owners["recovering"].wait(timeout=10)
        # The foreground command is not a job: it dies even here.
        wait_until(lambda: not any(_alive(i) for i in ids["recovering"]["foreground"]), timeout=15,
                   what="the dead host's foreground command to exit")
        jobs = [i for name, job in ids["recovering"].items() if name != "foreground" for i in job]
        assert all(_alive(i) for i in jobs), (
            "a recovering host's background job died with it; the next host could not re-adopt it", jobs)
    finally:
        _cleanup(owners, log, ids)


def _record_calls(monkeypatch):
    from tools import process_registry_ownership as registry_mod
    from tools.environments import local as local_mod

    calls = []
    for mod in (registry_mod, local_mod):
        monkeypatch.setattr(mod, "own_process_groups", lambda pgids: calls.append(("own", list(pgids))))
        monkeypatch.setattr(mod, "release_process_groups", lambda pgids: calls.append(("release", list(pgids))))
    monkeypatch.setattr("hermes_cli.process_identity.own_process_groups",
                        lambda pgids: calls.append(("own", list(pgids))))
    monkeypatch.setattr("hermes_cli.process_identity.register_child", lambda pid, purpose: None)
    return calls


def test_a_finished_job_and_a_returned_command_release_their_group(monkeypatch, tmp_path):
    from tools.environments.local import LocalEnvironment
    from tools.process_registry import ProcessRegistry

    calls = _record_calls(monkeypatch)
    registry = ProcessRegistry()
    session = registry.spawn_local("exit 0", cwd=str(tmp_path))
    wait_until(lambda: session.exited and ("release", [session.pid]) in calls, timeout=15,
               what="the finished job's release")
    assert calls[0] == ("own", [session.pid])

    calls.clear()
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    calls.clear()
    assert env.execute("echo hi")["returncode"] == 0
    assert [verb for verb, _ in calls] == ["own", "release"] and calls[0][1] == calls[1][1], calls


def test_a_command_yielded_to_background_keeps_its_group_until_the_job_ends(monkeypatch, tmp_path):
    import threading

    from tools.environments.local import LocalEnvironment
    from tools.interrupt import request_yield
    from tools.process_registry import ProcessRegistry

    calls = _record_calls(monkeypatch)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=60)
    registry = ProcessRegistry()
    gate, started = tmp_path / "gate", tmp_path / "started"
    adopted = []

    def handler(proc, output_so_far):
        adopted.append(registry.adopt_local(proc, command="wait for gate", cwd=str(tmp_path)))
        return {"output": output_so_far, "returncode": None}

    result = {}
    worker = threading.Thread(target=lambda: result.update(env.execute(
        f"touch {started}; while [ ! -e {gate} ]; do sleep 0.05; done", yield_handler=handler)))
    calls.clear()
    worker.start()
    wait_until(started.exists, timeout=15, what="the foreground command to start")
    request_yield(worker.ident)
    worker.join(timeout=15)
    assert result.get("returncode") is None and adopted, result
    session = adopted[0]
    assert calls == [("own", [session.pid])], "the yielding foreground wait released the job's group"
    assert session._owned_pgid == session.pid
    gate.touch()
    wait_until(lambda: ("release", [session.pid]) in calls, timeout=15, what="the adopted job's release")
    assert calls.count(("release", [session.pid])) == 1, calls


@pytest.mark.parametrize("jobs_outlive_host", [False, True], ids=["owned-jobs", "gateway"])
def test_a_failed_handoff_leaves_the_foreground_owner_its_group(monkeypatch, tmp_path, jobs_outlive_host):
    """adopt_local moved the group to the job before its reader started. If the reader cannot start,
    the foreground wait resumes: it must still hold the group, covered, so its completion releases
    it (a gateway had already released it; elsewhere the dead session kept it)."""
    import types

    from tools.process_registry import ProcessRegistry

    calls = _record_calls(monkeypatch)
    registry = ProcessRegistry()
    registry._jobs_outlive_host = jobs_outlive_host

    def no_reader(*_a, **_kw):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("threading.Thread.start", no_reader)
    proc = types.SimpleNamespace(pid=4242, _hermes_owned_pgid=4242, poll=lambda: None)
    with pytest.raises(RuntimeError):
        registry.adopt_local(proc, command="sleep 60", cwd=str(tmp_path))

    assert proc._hermes_owned_pgid == 4242, "the foreground wait lost its group"
    covered = not calls or calls[-1] == ("own", [4242])
    assert covered, f"the group is left uncovered: {calls}"
    assert not registry._running, "a session with no reader stayed registered"


def test_unowned_jobs_are_never_handed_to_the_supervisor(monkeypatch, tmp_path):
    from tools.process_registry import ProcessRegistry

    calls = _record_calls(monkeypatch)
    registry = ProcessRegistry()
    persisted = registry.spawn_local("exit 0", cwd=str(tmp_path), persist_on_release=True)
    registry._jobs_outlive_host = True
    recovered_host_job = registry.spawn_local("exit 0", cwd=str(tmp_path))
    wait_until(lambda: persisted.exited and recovered_host_job.exited, timeout=15, what="both jobs to exit")
    assert calls == [], calls


def test_a_managed_workers_jobs_are_never_handed_to_the_supervisor(monkeypatch, tmp_path):
    """A managed worker is a per-turn interpreter: the next turn's worker adopts its jobs from the
    checkpoint and reads their output log, so the jobs must outlive the worker that started them."""
    from agent import runtime_session_store
    from tools.process_registry import ProcessRegistry

    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(runtime_session_store, "_worker_process", True)
    job = ProcessRegistry().spawn_local("exit 0", cwd=str(tmp_path))
    wait_until(lambda: job.exited, timeout=15, what="the worker's job to exit")
    assert calls == [], calls
