"""Bot Desktop package install: sudo hand-off and single-flight invariants."""

from __future__ import annotations

import os
import threading

import pytest

from tools.bot_desktop import install, runtime

_REAL_INSTALL_COMMAND = runtime.install_command  # captured before the fixture pins a sudo line


@pytest.fixture(autouse=True)
def _isolated_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(install, "_sudo_nopasswd", lambda: False)
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(runtime, "install_command", lambda: "sudo apt-get install -y tigervnc-standalone-server")
    yield
    install._running.clear()


def test_empty_password_cancels_without_spawning(monkeypatch):
    monkeypatch.setattr(install.subprocess, "Popen", lambda *a, **k: pytest.fail("package manager spawned"))
    lines: list[str] = []
    code = install.install_packages(ask_password=lambda: "", on_line=lines.append)
    assert code == -1
    assert any("cancelled" in line for line in lines)


def test_second_install_for_same_profile_is_refused(monkeypatch):
    gate = threading.Event()
    entered = threading.Event()

    def slow_run(cmd, *, ask_password, on_line, timeout_seconds):
        entered.set()
        gate.wait(5)
        return 0

    monkeypatch.setattr(install, "_run", slow_run)
    worker = threading.Thread(target=install.install_packages, kwargs={"ask_password": lambda: "pw", "on_line": lambda _l: None})
    worker.start()
    assert entered.wait(5)
    with pytest.raises(install.InstallBusy):
        install.install_packages(ask_password=lambda: "pw", on_line=lambda _l: None)
    gate.set()
    worker.join(5)
    install.assert_not_running()  # slot released once the run finishes


def test_claim_is_atomic_and_refuses_a_second_claim():
    """The gateway claims BEFORE spawning its worker; a second Install click must fail at claim time, not
    pass a read-only check and race the worker for the slot."""
    key = install.claim()
    with pytest.raises(install.InstallBusy):
        install.claim()
    with pytest.raises(install.InstallBusy):
        install.install_packages(ask_password=lambda: "pw", on_line=lambda _l: None)
    install.release(key)
    install.claim()  # free again
    install.release(key)


@pytest.mark.platforms("linux")
def test_timeout_kills_the_package_managers_whole_process_group(monkeypatch):
    """sudo forks the package manager into the same (new) session; killing sudo alone leaves apt/dnf
    holding the dpkg lock as root. The timeout must take the group."""
    import subprocess
    import time

    monkeypatch.setattr(install, "_sudo_nopasswd", lambda: True)
    # stand-in for `sudo apt-get ...`: a parent that spawns a child and waits, both in the new session
    fake = ["sudo"]
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        if argv[:1] != fake:
            return real_popen(argv, **kw)
        return real_popen(["bash", "-c", "sleep 30 >/dev/null 2>&1 & echo child $!; wait"], **kw)

    monkeypatch.setattr(install.subprocess, "Popen", popen)
    lines: list[str] = []
    code = install._run("sudo apt-get install -y x", ask_password=lambda: "", on_line=lines.append, timeout_seconds=0.5)
    assert code != 0
    child = next(int(line.split()[1]) for line in lines if line.startswith("child "))
    from pathlib import Path

    def gone() -> bool:  # /proc-based: a reparented orphan sits outside our subtree, where os.kill(pid, 0) is guarded
        try:
            return "Z" in (Path(f"/proc/{child}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split() or ["Z"])[0]
        except OSError:
            return True

    for _ in range(50):  # the child must die with the group, not linger reparented to init
        if gone():
            break
        time.sleep(0.1)
    else:
        subprocess.run(["kill", "-9", str(child)], check=False)
        pytest.fail("grandchild survived the install timeout")


def test_passwordless_sudo_runs_the_install_without_asking_for_a_password(monkeypatch):
    """A host with NOPASSWD sudo must not pop the masked password card — the card blocks the install
    for up to five minutes on an answer nobody needs to give — and the package command still runs
    and streams its output."""
    monkeypatch.setattr(install, "_sudo_nopasswd", lambda: True)
    spawned: list[list[str]] = []

    class _Proc:
        """Stands in for the sudo child: the drain reads the pipe by fd, so stdout is a real pipe with
        the output already written and the write end closed (EOF)."""
        pid = 4242
        returncode = 0
        stdin = __import__("io").StringIO()

        def __init__(self, argv, **kw):
            spawned.append(argv)
            r, w = os.pipe()
            os.write(w, b"Reading package lists...\nDone\n")
            os.close(w)
            self.stdout = os.fdopen(r, "rb")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(install.subprocess, "Popen", _Proc)
    monkeypatch.setattr(install.os, "killpg", lambda *a: None)
    lines: list[str] = []
    code = install.install_packages(ask_password=lambda: pytest.fail("sudo password asked despite NOPASSWD"),
                                    on_line=lines.append)
    assert code == 0
    assert spawned and spawned[0][:1] == ["sudo"] and "-S" not in spawned[0]
    assert "Done" in lines


@pytest.mark.platforms("linux")
def test_timeout_returns_and_frees_the_slot_even_when_a_descendant_survives(monkeypatch):
    """From an unprivileged Hermes, killpg reaches the sudo leader but not a root-owned apt child; that
    child keeps the pipe's write end open, so draining stdout never sees EOF and the profile slot stays
    taken forever. The timeout must end the drain and release the slot regardless of what survived.
    Stand-in for the unkillable root child: a grandchild in its own session holding our stdout."""
    import subprocess
    import time

    monkeypatch.setattr(install, "_sudo_nopasswd", lambda: True)
    monkeypatch.setattr(install, "_TERM_GRACE_SECONDS", 0.2, raising=False)
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        if argv[:1] != ["sudo"]:
            return real_popen(argv, **kw)
        return real_popen(["sh", "-c", "setsid sleep 30 & echo child $!; wait"], **kw)

    monkeypatch.setattr(install.subprocess, "Popen", popen)
    lines: list[str] = []
    started = time.monotonic()
    worker = threading.Thread(target=lambda: lines.append(
        f"code {install.install_packages(ask_password=lambda: '', on_line=lines.append, timeout_seconds=0.5)}"))
    worker.start()
    worker.join(3.0)
    survivor = next((int(line.split()[1]) for line in lines if line.startswith("child ")), None)
    if survivor:
        subprocess.run(["kill", "-9", str(survivor)], check=False)
    assert not worker.is_alive(), f"install_packages hung {time.monotonic() - started:.1f}s on a surviving descendant"
    assert any(line.startswith("code ") and line != "code 0" for line in lines), lines
    assert any("timed out" in line for line in lines), lines
    install.release(install.claim())  # slot is free again


def test_root_installs_without_sudo_and_without_asking(monkeypatch):
    """The official Docker image runs Hermes as uid 0 with no sudo binary: the package manager must be
    run directly, and the password card must never be raised for a user who already is root."""
    import subprocess

    monkeypatch.setattr(install.os, "geteuid", lambda: 0)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
    monkeypatch.setattr(runtime, "package_manager", lambda: "apt")
    monkeypatch.setattr(runtime, "install_command", _REAL_INSTALL_COMMAND)
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        spawned.append(list(argv))
        return real_popen(["true"], **kw)

    monkeypatch.setattr(install.subprocess, "Popen", popen)
    code = install.install_packages(ask_password=lambda: pytest.fail("root was asked for a sudo password"),
                                    on_line=lambda _l: None)
    assert code == 0
    assert spawned and spawned[0][0] == "apt-get", spawned


def test_no_sudo_binary_returns_the_host_command_instead_of_a_password_card(monkeypatch):
    """Unprivileged with no sudo on the host (minimal containers): a password card would be a dead end,
    the user needs the exact command to run on the host instead."""
    monkeypatch.setattr(install.os, "geteuid", lambda: 1000)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
    monkeypatch.setattr(install.subprocess, "Popen", lambda *a, **k: pytest.fail("package manager spawned"))
    lines: list[str] = []
    code = install.install_packages(ask_password=lambda: pytest.fail("password card raised without sudo"),
                                    on_line=lines.append)
    assert code == install.NO_SUDO
    assert any("apt-get install" in line for line in lines), lines


def test_timeout_finishes_the_group_when_only_the_leader_dies_on_term(monkeypatch):
    """TERM ends the leader shell at once, but a descendant in the same group that ignores TERM used to be
    left alive holding the dpkg lock: the kill helper returned as soon as the leader was reaped. Cleanup is
    done only when the whole process group is gone."""
    import subprocess
    import time

    monkeypatch.setattr(install, "_sudo_nopasswd", lambda: True)
    monkeypatch.setattr(install, "_TERM_GRACE_SECONDS", 0.3, raising=False)
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        if argv[:1] != ["sudo"]:
            return real_popen(argv, **kw)
        return real_popen(["sh", "-c", "sh -c 'trap \"\" TERM; echo child $$; sleep 30' & wait"], **kw)

    monkeypatch.setattr(install.subprocess, "Popen", popen)
    lines: list[str] = []
    install.install_packages(ask_password=lambda: "", on_line=lines.append, timeout_seconds=0.5)
    child = next(int(line.split()[1]) for line in lines if line.startswith("child "))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _alive(child):
        time.sleep(0.05)
    try:
        assert not _alive(child), "the TERM-ignoring descendant survived the timeout cleanup"
    finally:
        subprocess.run(["kill", "-9", str(child)], check=False)


def _alive(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)


def test_install_slot_is_held_across_processes(tmp_path):
    """The gateway (Install card) and the CLI (`screen install`) are separate processes on one host: the
    single-flight must be a file lock in the profile's state dir, not only the in-process set."""
    import subprocess
    import sys

    key = install.claim()
    try:
        probe = subprocess.run([sys.executable, "-c",
            "import fcntl,sys\nfh=open(sys.argv[1],'a+',encoding='utf-8')\n"
            "try:\n    fcntl.flock(fh.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB); print('free')\n"
            "except OSError:\n    print('held')", str(runtime.state_dir() / "install.lock")],
            capture_output=True, text=True)
        assert probe.stdout.strip() == "held", probe
    finally:
        install.release(key)
    install.release(install.claim())  # freed: claimable again
