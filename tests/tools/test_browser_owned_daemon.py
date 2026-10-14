"""agent-browser daemons and their Chromium die with the Hermes process that drove them.

The daemon detaches into its own session and Chromium moves into a process group of its own, so a
SIGKILLed Hermes used to leave both running. Each local command now hands both groups to the
per-process owner-death supervisor. These tests use a stand-in daemon with the same topology.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import process_identity
from tools import browser_tool_lifecycle as lifecycle

# The stand-in Chromium reparents to init once its daemon is killed; cleaning it up leaves this
# test's subtree, which is exactly the orphan shape under test.
pytestmark = [pytest.mark.platforms("posix"), pytest.mark.live_system_guard_bypass]

# Named so the identity check sees "agent-browser"; the socket dir is bound via argv. Its child
# leaves for a group of its own, as Chromium does under the real daemon.
_DAEMON = r'''
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import os, time; os.setpgid(0, 0); time.sleep(300)"],
                         stdin=subprocess.DEVNULL)
print(child.pid, flush=True)
time.sleep(300)
'''


def _spawn_daemon(tmp_path, socket_dir: Path, session: str):
    script = tmp_path / "agent-browser"
    script.write_text(_DAEMON, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(socket_dir)], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, text=True, start_new_session=True)
    chromium = int(proc.stdout.readline())
    deadline = time.monotonic() + 10
    while os.getpgid(chromium) != chromium and time.monotonic() < deadline:  # windows-footgun: ok — posix-marked
        time.sleep(0.02)
    (socket_dir / f"{session}.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc, chromium


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(lifecycle, "_owned_session_groups", {})
    name = "h_owned"
    socket_dir = tmp_path / f"agent-browser-{name}"
    socket_dir.mkdir()
    procs = []
    yield name, socket_dir, procs
    for pid in procs:
        try:
            os.killpg(pid, 9)  # windows-footgun: ok — posix-marked
        except ProcessLookupError:
            pass


@pytest.fixture
def calls(monkeypatch):
    seen = []
    for verb, name in (("register", "own_process_groups"), ("unregister", "release_process_groups")):
        monkeypatch.setattr(lifecycle, name, lambda pgids, verb=verb: seen.append((verb, sorted(pgids))))
    monkeypatch.setattr(lifecycle, "register_child", lambda pid, purpose: None)
    return seen


def test_daemon_and_chromium_groups_are_registered_once(tmp_path, session, calls):
    name, socket_dir, procs = session
    daemon, chromium = _spawn_daemon(tmp_path, socket_dir, name)
    procs += [daemon.pid, chromium]

    lifecycle._own_local_daemon(name)
    lifecycle._own_local_daemon(name)  # every command re-checks; nothing new to register

    assert calls == [("register", sorted([daemon.pid, chromium]))]
    lifecycle._release_owned_daemon(name)
    assert calls[-1] == ("unregister", sorted([daemon.pid, chromium]))
    daemon.kill()
    daemon.wait(timeout=10)


def test_a_pid_file_naming_an_unbound_process_is_never_registered(tmp_path, session, calls):
    name, socket_dir, procs = session
    # A detached process of ours that is not bound to this session's socket dir.
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                stdin=subprocess.DEVNULL, start_new_session=True)
    procs.append(stranger.pid)
    (socket_dir / f"{name}.pid").write_text(str(stranger.pid), encoding="utf-8")

    lifecycle._own_local_daemon(name)

    assert calls == []
    stranger.kill()
    stranger.wait(timeout=10)


def test_a_sigkilled_owner_takes_daemon_and_chromium_with_it(tmp_path):
    """Real supervisor, real SIGKILL of the owner: both groups must go with it."""
    name = "h_owned"
    socket_dir = tmp_path / f"agent-browser-{name}"
    socket_dir.mkdir()
    (tmp_path / "agent-browser").write_text(_DAEMON, encoding="utf-8")
    owner_src = (
        "import subprocess, sys, time\n"
        "from tools import browser_tool_lifecycle as lc\n"
        f"lc._bt._socket_safe_tmpdir = lambda: {str(tmp_path)!r}\n"
        f"d = subprocess.Popen([sys.executable, {str(tmp_path / 'agent-browser')!r}, {str(socket_dir)!r}],"
        " stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, start_new_session=True)\n"
        "c = int(d.stdout.readline())\n"
        "import os\n"
        "while os.getpgid(c) != c: time.sleep(0.02)\n"  # windows-footgun: ok — posix-marked
        f"open({str(socket_dir / (name + '.pid'))!r}, 'w', encoding='utf-8').write(str(d.pid))\n"  # windows-footgun: ok — a write
        f"lc._own_local_daemon({name!r})\n"
        "print(d.pid, c, flush=True)\n"
        "time.sleep(300)\n"
    )
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    owner = subprocess.Popen([sys.executable, "-c", owner_src], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, text=True)
    groups = []
    try:
        groups = [int(x) for x in owner.stdout.readline().split()]
        assert len(groups) == 2
        owner.kill()
        owner.wait(timeout=10)
        deadline = time.monotonic() + 15
        alive = list(groups)
        while alive and time.monotonic() < deadline:
            for g in list(alive):
                try:
                    os.killpg(g, 0)  # windows-footgun: ok — posix-marked
                except ProcessLookupError:
                    alive.remove(g)
            time.sleep(0.05)
        assert not alive, f"groups outlived kill -9 of their owner: {alive}"
    finally:
        for g in groups:
            try:
                os.killpg(g, 9)  # windows-footgun: ok — posix-marked
            except ProcessLookupError:
                pass
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=10)
        process_identity._supervised_pgids.clear()
