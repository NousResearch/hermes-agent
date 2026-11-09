"""POSIX session kernels: the kernel's whole process group dies with its owner, however the owner dies.

The owners are real host processes importing the real ``tools.code_kernel._spawn``; each starts a kernel
whose cell spawns a subprocess of its own (it stays in the kernel's group, as every cell child does).
One owner is SIGKILLed so no Python cleanup of ours can run; the other is the live-owner control.
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
import json, sys, time
from pathlib import Path
root, role = Path(sys.argv[1]), sys.argv[2]
from tools.code_kernel import SessionKernel, _spawn
k = SessionKernel((role,))
_spawn(k, task_id=role, child_python=sys.executable, child_cwd="", sandbox_tools=frozenset(),
       max_tool_calls=1)
receipt = str(root / (role + ".json"))
code = (
    "import json, os, subprocess, sys, time, psutil\n"
    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "ids = {n: {'pid': p.pid, 'created': p.create_time()} for n, p in"
    " (('kernel', psutil.Process()), ('cell_child', psutil.Process(c.pid)))}\n"
    "open(%r + '.tmp', 'w').write(json.dumps(ids)); os.replace(%r + '.tmp', %r)\n"
    "time.sleep(120)\n") % (receipt, receipt, receipt)
k.proc.stdin.write((json.dumps({"id": "x", "code": code}) + "\n").encode())
k.proc.stdin.flush()
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
    wait_until(path.exists, timeout=30, what=f"kernel receipt {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


# The kernel and its cell child reparent to init once their owner dies, so cleaning them up leaves
# this test's subtree; real signal delivery to a real orphan is the point of the test.
@pytest.mark.live_system_guard_bypass
def test_owner_sigkill_reaps_the_kernel_group_not_a_live_owners(tmp_path):
    script = tmp_path / "disposable owner.py"
    script.write_text(_OWNER, encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    ids = {}
    with (tmp_path / "owners.log").open("w", encoding="utf-8") as log:
        owners = {role: subprocess.Popen([sys.executable, str(script), str(tmp_path), role], env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                  for role in ("control", "victim")}
        try:
            ids = {role: _read(tmp_path / f"{role}.json") for role in owners}
            for role in owners:
                assert os.getpgid(ids[role]["cell_child"]["pid"]) == ids[role]["kernel"]["pid"], (
                    "the cell's subprocess left the kernel's group", role)

            owners["victim"].kill()  # SIGKILL: no atexit, no finally, nothing of ours runs
            owners["victim"].wait(timeout=10)

            wait_until(lambda: not any(_alive(i) for i in ids["victim"].values()), timeout=15,
                       what="the dead owner's kernel and its cell child to exit")
            assert all(_alive(i) for i in ids["control"].values()), (
                "a live owner's kernel group was killed", ids["control"])
        finally:
            for identity in (i for group in ids.values() for i in group.values()):
                if _alive(identity):
                    psutil.Process(identity["pid"]).kill()
            for owner in owners.values():
                if owner.poll() is None:
                    owner.kill()
                owner.wait(timeout=10)


def _kernel_with_cell_child(monkeypatch, tmp_path, child_argv):
    """A real kernel (spawn registration captured) running a cell that spawned *child_argv*."""
    from hermes_cli import process_identity
    from tools.code_kernel import SessionKernel, _spawn

    calls = []
    for verb, name in (("register", "own_process_groups"), ("unregister", "release_process_groups")):
        monkeypatch.setattr(process_identity, name, lambda pgids, verb=verb: calls.append((verb, list(pgids))))
    monkeypatch.setattr(process_identity, "register_child", lambda pid, purpose: None)
    kernel = SessionKernel(("graceful",))
    _spawn(kernel, task_id="graceful", child_python=sys.executable, child_cwd="",
           sandbox_tools=frozenset(), max_tool_calls=1)
    receipt = tmp_path / "cell_child.json"
    code = ("import json, os, subprocess, time, psutil\n"
            f"c = subprocess.Popen({child_argv!r})\n"
            "ids = {'pid': c.pid, 'created': psutil.Process(c.pid).create_time()}\n"
            f"open({str(receipt)!r} + '.tmp', 'w').write(json.dumps(ids))\n"
            f"os.replace({str(receipt)!r} + '.tmp', {str(receipt)!r})\n"
            "time.sleep(60)\n")
    kernel.proc.stdin.write((json.dumps({"id": "x", "code": code}) + "\n").encode())
    kernel.proc.stdin.flush()
    return kernel, _read(receipt), calls


def _kill(identity):
    if _alive(identity):
        psutil.Process(identity["pid"]).kill()
    wait_until(lambda: not _alive(identity), what="cell child cleanup")


def test_teardown_releases_the_kernels_group(monkeypatch, tmp_path):
    kernel, child, calls = _kernel_with_cell_child(
        monkeypatch, tmp_path, [sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert calls == [("register", [kernel.proc.pid])]
        kernel.teardown()
        assert not _alive(child), "teardown left the cell's subprocess running"
        assert calls[-1] == ("unregister", [kernel.proc.pid]), (
            "a kernel group this process tore down itself stayed registered with the owner-death supervisor")
    finally:
        kernel.teardown()
        _kill(child)


# The surviving cell child is orphaned once its kernel exits; killing it leaves the test subtree.
@pytest.mark.live_system_guard_bypass
def test_teardown_keeps_covering_a_cell_child_that_survived_it(monkeypatch, tmp_path):
    # Ignores the teardown's SIGTERM. Escalation is off so a kernel slow to exit under load cannot
    # turn the TERM into a tree SIGKILL that takes the child too (the precondition, not the point).
    from tools import code_execution_tool
    real_kill = code_execution_tool._kill_process_group
    monkeypatch.setattr(code_execution_tool, "_kill_process_group",
                        lambda proc, escalate=False: real_kill(proc, escalate=False))
    # The ready file is written only once the trap is in place: the receipt alone races a loaded
    # host, where the teardown's SIGTERM can land before sh has installed it.
    ready = tmp_path / "trapped"
    kernel, child, calls = _kernel_with_cell_child(
        monkeypatch, tmp_path, ["sh", "-c", f"trap '' TERM; : > '{ready}'; exec sleep 60"])
    try:
        wait_until(ready.exists, what="the cell child to ignore SIGTERM")
        kernel.teardown()
        assert _alive(child), "precondition: the TERM-ignoring cell child survived teardown"
        assert calls == [("register", [kernel.proc.pid])], (
            "the group was released while a cell child was still in it")
    finally:
        _kill(child)
