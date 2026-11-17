"""Browser Use harness daemons die with the Hermes process that drove them.

The harness detaches its daemon on purpose (``start_new_session``) so a session persists across
``browser_exec`` calls; that also made it outlive a SIGKILLed / OOM-killed Hermes forever (#121095).
browser_exec now hands the daemon's process group to the per-process owner-death supervisor. These
tests drive the real registration seam against a real detached stand-in daemon.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import process_identity
from tests.waiting import wait_until
from tools import browser_use_cli


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)  # windows-footgun: ok — posix-marked test
    except (ProcessLookupError, PermissionError):  # macOS: EPERM for a group left with only zombies
        return False
    return True

pytestmark = pytest.mark.platforms("posix")

# A stand-in harness daemon: the real module name on argv (the identity check reads argv), detached
# into its own session exactly like browser_harness._ipc.spawn_kwargs() does.
_DAEMON_PKG = "browser_harness_standin"


@pytest.fixture
def fake_daemon(tmp_path):
    pkg = tmp_path / "pkgroot" / "browser_harness"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "daemon.py").write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "browser_harness.daemon"], cwd=str(tmp_path / "pkgroot"),
                            stdin=subprocess.DEVNULL, start_new_session=True,
                            env={**os.environ, **process_identity.spawn_env("browser-harness-daemon")})
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    yield proc, runtime
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


@pytest.fixture
def registered(monkeypatch):
    calls = []
    monkeypatch.setattr(browser_use_cli, "release_process_group_once_empty",
                        lambda pgid, **kw: calls.append(("unregister", [pgid])))
    for verb, name in (("register", "own_process_groups"),):
        monkeypatch.setattr(browser_use_cli, name, lambda pgids, verb=verb: calls.append((verb, list(pgids))))
    monkeypatch.setattr(browser_use_cli, "register_child", lambda pid, purpose: None)
    monkeypatch.setattr(browser_use_cli, "_owned_daemon_groups", {})
    return calls


def _publish(runtime: Path, stem: str, pid: int, json_record: bool = True) -> None:
    body = json.dumps({"pid": pid, "started": 1}) if json_record else str(pid)
    (runtime / f"{stem}.pid").write_text(body, encoding="utf-8")


@pytest.mark.parametrize("json_record", [True, False])
def test_the_driven_daemon_is_handed_to_the_supervisor_once(fake_daemon, registered, json_record):
    proc, runtime = fake_daemon
    _publish(runtime, "bu", proc.pid, json_record)  # BH_RUNTIME_DIR isolates one daemon: stem "bu"
    env = {"BH_RUNTIME_DIR": str(runtime), "BU_NAME": "work"}

    browser_use_cli._own_harness_daemon(env)
    browser_use_cli._own_harness_daemon(env)  # every browser_exec call re-checks; register once

    assert registered == [("register", [proc.pid])]


def test_an_owned_harness_daemon_is_not_re_verified_and_a_replacement_releases_the_old_group(
        fake_daemon, registered, monkeypatch):
    proc, runtime = fake_daemon
    _publish(runtime, "bu", proc.pid)
    env = {"BH_RUNTIME_DIR": str(runtime), "BU_NAME": "work"}
    browser_use_cli._own_harness_daemon(env)

    def _never(*_a):
        raise AssertionError("an already-owned daemon was re-verified")
    monkeypatch.setattr(browser_use_cli, "spawned_by_this_process", _never)
    browser_use_cli._own_harness_daemon(env)  # steady state: one pid-file read
    assert registered == [("register", [proc.pid])]

    earlier = subprocess.Popen([sys.executable, "-c", ""], stdin=subprocess.DEVNULL, start_new_session=True)
    earlier.wait(timeout=10)  # an earlier daemon of this identity, gone: its group is empty
    browser_use_cli._owned_daemon_groups[browser_use_cli.daemon_identity(env)] = earlier.pid
    monkeypatch.setattr(browser_use_cli, "spawned_by_this_process", lambda pid: True)
    browser_use_cli._own_harness_daemon(env)
    assert registered[1:] == [("register", [proc.pid]), ("unregister", [earlier.pid])]


def test_a_pid_file_naming_something_else_is_never_registered(tmp_path, registered):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    # A live process of ours that is NOT a harness daemon (and not a session leader): a stale or
    # planted pid file must never hand its group to the supervisor.
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL)
    try:
        _publish(runtime, "bu-default", stranger.pid)
        browser_use_cli._own_harness_daemon({"XDG_CONFIG_HOME": str(tmp_path / "x"), "BH_HOME": str(runtime.parent)})
        browser_use_cli._own_harness_daemon({"BH_RUNTIME_DIR": str(runtime), "BH_RUNTIME_DIR_SHARED": "1"})
    finally:
        stranger.kill()
        stranger.wait(timeout=10)
    assert registered == []


def test_stopping_the_daemons_releases_their_groups(monkeypatch, registered):
    daemon = browser_use_cli.daemon_identity({"BU_NAME": "work"})
    browser_use_cli._owned_daemon_groups.update({daemon: 4242})
    monkeypatch.setattr(browser_use_cli, "_driven_daemons", {daemon: "home"})
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-harness"])
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", dict)
    monkeypatch.setattr(browser_use_cli, "_run_cli_killing_process_group", lambda *a: None)

    browser_use_cli.stop_harness_daemons()

    assert registered == [("unregister", [4242])]
    assert browser_use_cli._owned_daemon_groups == {}


def _shared_runtime(tmp_path) -> tuple[Path, dict]:
    """A runtime dir shared between names (``bu-<BU_NAME>.pid``) and the env naming it, so a daemon
    identity built from that env carries the dir."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return runtime, {"BH_RUNTIME_DIR": str(runtime), "BH_RUNTIME_DIR_SHARED": "1"}


@pytest.mark.live_system_guard_bypass
def test_a_sigkilled_owner_takes_its_harness_daemon_with_it(tmp_path):
    """End to end through the real supervisor: the owner registers its detached daemon, then dies
    by SIGKILL (no Python cleanup runs), and the daemon must go with it."""
    pkg = tmp_path / "pkgroot" / "browser_harness"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "daemon.py").write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    owner_src = (
        "import json, os, subprocess, sys, time\n"
        "from tools import browser_use_cli\n"
        f"d = subprocess.Popen([sys.executable, '-m', 'browser_harness.daemon'], cwd={str(tmp_path / 'pkgroot')!r},"
        " stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, start_new_session=True,"
        " env=browser_use_cli._base_subprocess_env())\n"
        f"open({str(runtime / 'bu.pid')!r}, 'w', encoding='utf-8').write(json.dumps({{'pid': d.pid}}))\n"  # windows-footgun: ok — a write
        f"browser_use_cli._own_harness_daemon({{'BH_RUNTIME_DIR': {str(runtime)!r}}})\n"
        "print(d.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    owner = subprocess.Popen([sys.executable, "-c", owner_src], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, text=True)
    daemon_pid = None
    try:
        daemon_pid = int(owner.stdout.readline())
        owner.kill()
        owner.wait(timeout=10)
        wait_until(lambda: not _group_alive(daemon_pid), timeout=15,
                   what="the harness daemon to die with the kill -9'd Hermes process that drove it")
    finally:
        if daemon_pid is not None:
            try:
                os.killpg(daemon_pid, 9)  # windows-footgun: ok — posix-marked test
            except (ProcessLookupError, PermissionError):
                pass
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=10)
        process_identity._supervised_pgids.clear()


def test_a_daemon_another_live_hermes_started_on_the_same_name_is_never_registered(tmp_path, registered):
    """Two Hermes processes on the default BU_NAME read the same bu-default.pid. The daemon belongs
    to the one whose CLI started it: the other must not hand it to its supervisor, or that Hermes
    dying would kill a browser session the first still uses."""
    pkg = tmp_path / "pkgroot" / "browser_harness"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "daemon.py").write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    other_hermes = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    tag = process_identity.spawn_env("browser-harness-daemon")
    tag[process_identity.SPAWN_ENV_VAR] = tag[process_identity.SPAWN_ENV_VAR].replace(
        f":{os.getpid()}:", f":{other_hermes.pid}:")
    daemon = subprocess.Popen([sys.executable, "-m", "browser_harness.daemon"], cwd=str(tmp_path / "pkgroot"),
                              stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, **tag})
    try:
        _publish(runtime, "bu-default", daemon.pid)
        # The in-call check stops too, instead of re-checking a daemon that is never ours every tick.
        assert browser_use_cli._own_harness_daemon_until_owned({"BH_HOME": str(runtime.parent)}) is False
        assert registered == []
    finally:
        daemon.kill()
        daemon.wait(timeout=10)
        other_hermes.stdin.close()
        other_hermes.wait(timeout=10)


def test_a_long_first_browser_exec_owns_its_daemon_before_the_call_returns(monkeypatch):
    """The first call starts the daemon and may run for minutes (navigation, user code). Its daemon
    must reach the supervisor during the call, or a Hermes killed mid-call leaves it running."""
    import threading

    owned_during_call = threading.Event()
    running = threading.Event()
    monkeypatch.setattr(browser_use_cli, "_own_harness_daemon",
                        lambda env: running.is_set() and owned_during_call.set())

    def run_cli(*_args):
        running.set()
        owned_during_call.wait(10)  # the CLI is still "running" until the daemon was owned
        running.clear()
        return subprocess.CompletedProcess([], 0, "ok", "")

    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", dict)
    monkeypatch.setattr(browser_use_cli, "_route_exec_current",
                        lambda env, *_a: ({**env, "BU_CDP_URL": "http://127.0.0.1:9222"}, None, False, 0))
    monkeypatch.setattr(browser_use_cli, "_run_cli_killing_process_group", run_cli)
    monkeypatch.setattr(browser_use_cli, "_driven_daemons", {})

    browser_use_cli.browser_exec("print(page_info())", task_id="long-first-call")
    assert owned_during_call.is_set(), "the daemon was only owned after the CLI call returned"


def test_a_managed_worker_never_owns_its_harness_daemon(fake_daemon, registered, monkeypatch):
    """A managed per-turn worker keeps its browser between turns (agent/managed_worker.py); owning
    the daemon would kill it with every turn's worker, as _owns_job already avoids for jobs."""
    from agent import runtime_session_store
    proc, runtime = fake_daemon
    _publish(runtime, "bu", proc.pid)
    monkeypatch.setattr(runtime_session_store, "_worker_process", True)

    env = {"BH_RUNTIME_DIR": str(runtime), "BU_NAME": "work"}
    assert browser_use_cli._own_harness_daemon_until_owned(env) is False  # and the in-call check stops

    assert registered == []
