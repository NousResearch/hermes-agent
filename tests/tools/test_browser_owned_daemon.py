"""agent-browser daemons and their Chromium die with the Hermes process that drove them.

The daemon detaches into its own session and Chromium moves into a process group of its own, so a
SIGKILLed Hermes used to leave both running. Each local command now hands both groups to the
per-process owner-death supervisor. These tests use a stand-in daemon with the same topology.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import process_identity
from tests.waiting import wait_until
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


def _spawn_daemon(tmp_path, socket_dir: Path, session: str, env=None):
    env = {**os.environ, **process_identity.spawn_env("test")} if env is None else env
    script = tmp_path / "agent-browser"
    script.write_text(_DAEMON, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(socket_dir)], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, text=True, start_new_session=True, env=env)
    chromium = int(proc.stdout.readline())
    wait_until(lambda: os.getpgid(chromium) == chromium, timeout=10,  # windows-footgun: ok — posix-marked
               what="the stand-in Chromium to leave for its own group")
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
    monkeypatch.setattr(lifecycle, "release_process_group_once_empty",
                        lambda pgid, **kw: seen.append(("unregister", [pgid])))
    for verb, name in (("register", "own_process_groups"),):
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
    assert sorted(calls[1:]) == [("unregister", [pgid]) for pgid in sorted([daemon.pid, chromium])]
    daemon.kill()
    daemon.wait(timeout=10)


def test_an_owned_daemon_skips_the_identity_checks_and_an_exited_one_is_forgotten(tmp_path, session, calls,
                                                                                    monkeypatch):
    name, socket_dir, procs = session
    daemon, chromium = _spawn_daemon(tmp_path, socket_dir, name)
    procs += [daemon.pid, chromium]
    lifecycle._own_local_daemon(name)

    def _never(*_a):
        raise AssertionError("an already-owned daemon was re-verified on a later command")
    monkeypatch.setattr(lifecycle, "_verify_reapable_browser_daemon", _never)
    monkeypatch.setattr(lifecycle, "spawned_by_this_process", _never)
    lifecycle._own_local_daemon(name)  # steady state: pid-file read and children rescan only
    assert calls == [("register", sorted([daemon.pid, chromium]))]

    (socket_dir / f"{name}.pid").unlink()  # the daemon idled out and removed its pid file
    lifecycle._own_local_daemon(name)
    assert name not in lifecycle._owned_session_groups
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
        "import os, subprocess, sys, time\n"
        "from tools import browser_tool_lifecycle as lc\n"
        "from hermes_cli.process_identity import spawn_env\n"
        f"lc._bt._socket_safe_tmpdir = lambda: {str(tmp_path)!r}\n"
        f"d = subprocess.Popen([sys.executable, {str(tmp_path / 'agent-browser')!r}, {str(socket_dir)!r}],"
        " stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, start_new_session=True,"
        " env={**os.environ, **spawn_env('agent-browser-daemon')})\n"
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
        def _alive_groups():
            alive = []
            for g in groups:
                try:
                    os.killpg(g, 0)  # windows-footgun: ok — posix-marked
                    alive.append(g)
                except (ProcessLookupError, PermissionError):
                    pass  # macOS answers EPERM for a group left with only unreaped zombies
            return alive

        wait_until(lambda: not _alive_groups(), timeout=15,
                   what="the daemon and Chromium groups to die with their kill -9'd owner")
    finally:
        for g in groups:
            try:
                os.killpg(g, 9)  # windows-footgun: ok — posix-marked
            except (ProcessLookupError, PermissionError):
                pass
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=10)
        process_identity._supervised_pgids.clear()


def test_a_daemon_another_live_hermes_started_is_never_registered(tmp_path, session, calls):
    """Two Hermes processes can share a session (the real-profile attach session always is shared):
    a verified daemon in that session belongs to whoever started it, never to every Hermes that
    reads its pid file. Killing this process must not take another live Hermes's browser."""
    name, socket_dir, procs = session
    other_hermes = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        foreign_tag = process_identity.spawn_env("agent-browser-daemon")
        foreign_tag[process_identity.SPAWN_ENV_VAR] = foreign_tag[process_identity.SPAWN_ENV_VAR].replace(
            f":{os.getpid()}:", f":{other_hermes.pid}:")
        daemon, chromium = _spawn_daemon(tmp_path, socket_dir, name, env={**os.environ, **foreign_tag})
        procs += [daemon.pid, chromium]

        lifecycle._own_local_daemon(name)

        assert calls == []
        daemon.kill()
        daemon.wait(timeout=10)
    finally:
        other_hermes.stdin.close()
        other_hermes.wait(timeout=10)


def test_a_long_first_agent_browser_command_owns_its_daemon_before_it_returns(tmp_path, monkeypatch):
    """The first command forks the daemon and Chromium and may navigate for a long time. They must
    reach the supervisor during that command, or a Hermes killed mid-command leaves them running."""
    import threading
    from tools import browser_tool_session as bt_session

    owned_during_call = threading.Event()
    running = threading.Event()
    monkeypatch.setattr(lifecycle, "_own_local_daemon",
                        lambda name: running.is_set() and name == "s" and owned_during_call.set())

    class _Proc:
        returncode = 0

        def wait(self, timeout=None):
            running.set()
            owned_during_call.wait(10)
            running.clear()
            return 0

    sock = tmp_path / "sock"
    sock.mkdir()
    (sock / "_stdout_open").write_text('{"success": true, "data": {}}', encoding="utf-8")
    (sock / "_stderr_open").write_text("", encoding="utf-8")
    monkeypatch.setattr(bt_session, "_prepare_session_socket_dir", lambda name: str(sock))
    monkeypatch.setattr(bt_session, "_ensure_screen_for_headed_chromium", lambda: None)
    monkeypatch.setattr(bt_session, "_agent_browser_command_env", lambda d: {})
    monkeypatch.setattr(bt_session, "_apply_chromium_sandbox_args", lambda env: None)
    monkeypatch.setattr(bt_session, "_sandbox_wrap", lambda parts, env, d: (parts, env))
    monkeypatch.setattr(bt_session, "_popen_agent_browser", lambda *a: _Proc())

    info = {"session_name": "s", "cdp_url": None, "features": {"local": True}}
    result = bt_session._spawn_and_collect("t", info, ["agent-browser", "open"], "open", "chrome", 30)
    assert result.get("success"), result
    assert owned_during_call.is_set(), "the daemon was only owned after the command returned"
