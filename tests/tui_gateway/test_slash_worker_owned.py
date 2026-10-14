"""The slash worker and everything it starts in its group die with the gateway that owns it.

``_SlashWorker`` is started through ``process_identity.spawn_owned``: the worker leads its own process
group and that group goes to the gateway's owner-death supervisor. Before, the worker only had a ppid
watchdog of its own, so a gateway SIGKILL / OOM left whatever the worker had started (a running quick
command's shell, its children) alive forever, and a normal close signalled the worker alone.

The live tests drive the real ``_SlashWorker`` and the real worker process under disposable homes; the
owner is SIGKILLed, so no Python cleanup of ours can run.
"""

import contextlib
import importlib.abc
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from tests.waiting import wait_until
from hermes_cli import process_identity
from tui_gateway import server, slash_worker

_REPO = Path(__file__).resolve().parents[2]

# Owner stand-in: the real gateway-side class, optionally running a quick command whose shell stays in
# the worker's group. Blocks on stdin so a killed pytest leaves no owner behind.
_OWNER = r'''
import json, os, sys, threading
from pathlib import Path
from tui_gateway import server
out = Path(sys.argv[1])
worker = server._SlashWorker("owned-worker-test", "")
if sys.argv[2] == "quick":
    threading.Thread(target=lambda: worker.run("/linger"), daemon=True).start()
(out.with_suffix(".tmp")).write_text(json.dumps({"owner": os.getpid(), "worker": worker.proc.pid}))
(out.with_suffix(".tmp")).replace(out)
sys.stdin.read()
'''


def _group(pgid: int) -> list[psutil.Process]:
    """Live (non-zombie) members of ``pgid``."""
    members = []
    for proc in psutil.process_iter():
        try:
            if os.getpgid(proc.pid) == pgid and proc.status() != psutil.STATUS_ZOMBIE:  # windows-footgun: ok — posix-marked tests
                members.append(proc)
        except (OSError, psutil.Error):
            continue
    return members


def _cmdline(proc: psutil.Process) -> list[str]:
    try:
        return proc.cmdline()
    except psutil.Error:
        return []


def _home(tmp_path: Path, name: str) -> dict:
    home = tmp_path / name
    (home / ".hermes").mkdir(parents=True)
    (home / ".hermes" / "config.yaml").write_text(
        "quick_commands:\n  linger:\n    type: exec\n    command: sleep 120\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"), PYTHONPATH=str(_REPO))
    return env


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the worker group reparents to init once its owner dies
def test_owner_sigkill_takes_the_worker_group_and_spares_a_live_owners_worker(tmp_path):
    script = tmp_path / "owner.py"
    script.write_text(_OWNER, encoding="utf-8")
    owners, ids = {}, {}
    try:
        for role, mode in (("doomed", "quick"), ("control", "idle")):
            owners[role] = subprocess.Popen(
                [sys.executable, str(script), str(tmp_path / f"{role}.json"), mode],
                env=_home(tmp_path, role), cwd=str(tmp_path), stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for role in owners:
            receipt = wait_until(lambda role=role: (tmp_path / f"{role}.json").exists()
                                 and json.loads((tmp_path / f"{role}.json").read_text(encoding="utf-8")),
                                 timeout=60, what=f"{role} owner receipt")
            ids[role] = receipt["worker"]
        doomed, control = ids["doomed"], ids["control"]
        # The quick command's shell runs inside the worker's group; that is what used to survive.
        def _quick_running():
            members = _group(doomed)
            return any(_cmdline(p)[-2:] == ["sleep", "120"] for p in members) and members
        members = wait_until(_quick_running, timeout=90,
                             what="the quick command running in the doomed worker's group")
        assert wait_until(lambda: _group(control), timeout=30, what="the control worker running")

        owners["doomed"].kill()  # SIGKILL the owner: no atexit, no finally, nothing of ours runs
        owners["doomed"].wait(timeout=10)

        wait_until(lambda: not _group(doomed), timeout=15,
                   what=f"worker group {doomed} gone after its owner was SIGKILLed: {members}")
        assert _group(control), "a live owner's slash worker was killed"
    finally:
        for pgid in ids.values():
            for proc in _group(pgid):
                try:
                    proc.kill()
                except psutil.Error:
                    pass
        for owner in owners.values():
            if owner.poll() is None:
                owner.kill()
            owner.communicate(timeout=10)


def _group_leader(*shell: str) -> subprocess.Popen:
    """A stand-in worker leading its own group, with a TERM-ignoring child that stays in it."""
    return subprocess.Popen(["sh", "-c", *shell], stdin=subprocess.DEVNULL,
                            start_new_session=True)  # detached-spawn: ok — test stand-in, killed in finally


@pytest.mark.platforms("posix")
def test_close_stops_the_whole_group_then_releases_it(monkeypatch):
    released = []
    monkeypatch.setattr(process_identity, "release_process_groups", lambda pgids: released.append(list(pgids)))
    leader = _group_leader("trap '' TERM; sleep 120 & wait")
    try:
        wait_until(lambda: len(_group(leader.pid)) >= 2, what="the stand-in's child in its group")

        server._stop_worker_group(leader)

        assert not _group(leader.pid), "close() left a member of the worker's group running"
        assert released == [[leader.pid]]
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(leader.pid, signal.SIGKILL)  # windows-footgun: ok — posix-marked test
        leader.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_a_worker_that_already_died_is_never_group_signalled(monkeypatch):
    """A dead leader no longer proves the pgid is ours: leave the group to the supervisor."""
    released, signalled = [], []
    monkeypatch.setattr(process_identity, "release_process_groups", lambda pgids: released.append(list(pgids)))
    leader = _group_leader("sleep 120 & exit 0")
    try:
        leader.wait(timeout=10)
        survivors = wait_until(lambda: _group(leader.pid), what="the leaderless member")
        real_killpg = os.killpg
        monkeypatch.setattr(server.os, "killpg",
                            lambda pgid, sig: (signalled.append(sig), real_killpg(pgid, sig))[1])

        server._stop_worker_group(leader)

        assert signalled and set(signalled) == {0}, "a dead worker's group number was signalled"
        assert released == [], "a group with a live member was released from the supervisor"
        assert all(p.is_running() for p in survivors)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(leader.pid, signal.SIGKILL)  # windows-footgun: ok — posix-marked test


def test_main_imports_cli_before_runtime_prep_and_accepts_a_legacy_parent_pid(monkeypatch):
    """``cli`` loads ~/.hermes/.env, which MCP discovery in the runtime prep needs. A gateway started
    before an update still passes ``--parent-pid``; the updated worker must still start."""
    events = []

    class FakeHermesCLI:
        def __init__(self, **kwargs):
            events.append(("cli", kwargs["resume"]))

    class FakeCliFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            return importlib.util.spec_from_loader(name, self) if name == "cli" else None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            events.append(("cli-import",))
            module.HermesCLI = FakeHermesCLI

    # setitem first so teardown also removes the stub module the import below inserts.
    monkeypatch.setitem(sys.modules, "cli", None)
    monkeypatch.delitem(sys.modules, "cli")
    monkeypatch.setattr(sys, "meta_path", [FakeCliFinder(), *sys.meta_path])
    monkeypatch.setattr(slash_worker, "_prepare_slash_worker_runtime", lambda: events.append(("runtime",)))
    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "session-1", "--parent-pid", "424242"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    slash_worker.main()

    assert events == [("cli-import",), ("runtime",), ("cli", "session-1")]


def test_slash_worker_close_reaps_zombie_and_closes_fds(monkeypatch):
    """A hung worker is SIGKILLed (its whole group on POSIX), the zombie reaped, all pipes closed — once."""
    calls = {k: 0 for k in ("terminate", "kill", "wait", "stdin", "stdout", "stderr")}
    group_signals = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls[self.name] += 1

    class FakeProc:
        pid = 424242
        stdin, stdout, stderr = (FakeStream(n) for n in ("stdin", "stdout", "stderr"))

        def poll(self):
            return None  # always alive -> forces terminate then kill

        def terminate(self):
            calls["terminate"] += 1

        def kill(self):
            calls["kill"] += 1

        def wait(self, timeout=None):
            calls["wait"] += 1
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: group_signals.append((pgid, sig)), raising=False)
    monkeypatch.setattr(process_identity, "release_process_groups", lambda pgids: pytest.fail("a live group was released"))
    worker = object.__new__(server._SlashWorker)
    worker.proc = FakeProc()

    worker.close()
    worker.close()  # idempotent

    if os.name == "posix":
        # TERM then KILL to the worker's group; the trailing 0s are the existence probes that keep a
        # group with survivors registered with the owner-death supervisor.
        assert group_signals[:2] == [(424242, signal.SIGTERM), (424242, signal.SIGKILL)]
        assert group_signals[2:] and set(group_signals[2:]) == {(424242, 0)}
    else:
        assert calls["terminate"] == calls["kill"] == 1
    assert calls["wait"] >= 2  # reaped after both terminate and kill
    assert calls["stdin"] == calls["stdout"] == calls["stderr"] == 1
