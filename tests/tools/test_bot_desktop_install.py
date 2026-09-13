"""Bot Desktop package install: sudo hand-off and single-flight invariants."""

from __future__ import annotations

import io
import os
import threading

import pytest

from tools.bot_desktop import install, runtime


@pytest.fixture(autouse=True)
def _isolated_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    system_executables = {
        "sudo": "/usr/bin/sudo",
        "apt-get": "/usr/bin/apt-get",
        "dnf": "/usr/bin/dnf",
        "pacman": "/usr/bin/pacman",
    }
    monkeypatch.setattr(runtime, "_system_executable", system_executables.get)
    monkeypatch.setattr(install, "_sudo_nopasswd", lambda _sudo: False)
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(runtime, "install_command",
                        lambda: "/usr/bin/sudo /usr/bin/apt-get install -y tigervnc-standalone-server")
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


def test_install_uses_absolute_executables_and_a_scrubbed_environment(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.pid = os.getpid()
            self.stdin = io.StringIO()
            self.stdout: list[str] = []

        @staticmethod
        def wait():
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setenv("PATH", "/home/user/.local/bin:/usr/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setattr(install.subprocess, "Popen", fake_popen)

    command = "/usr/bin/sudo /usr/bin/apt-get install -y package"
    assert install._run(command, ask_password=lambda: "password", on_line=lambda _line: None,
                        timeout_seconds=1) == 0
    assert captured["argv"][:5] == ["/usr/bin/sudo", "-S", "-p", "", "/usr/bin/apt-get"]
    assert captured["env"] == {
        "PATH": runtime._SYSTEM_PATH,
        "DEBIAN_FRONTEND": "noninteractive",
        "LC_ALL": "C.UTF-8",
    }


@pytest.mark.linux_only
def test_timeout_kills_the_package_managers_whole_process_group(monkeypatch):
    """sudo forks the package manager into the same (new) session; killing sudo alone leaves apt/dnf
    holding the dpkg lock as root. The timeout must take the group."""
    import subprocess
    import time

    monkeypatch.setattr(install, "_sudo_nopasswd", lambda _sudo: True)
    # stand-in for `sudo apt-get ...`: a parent that spawns a child and waits, both in the new session
    fake = ["/usr/bin/sudo"]
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        if argv[:1] != fake:
            return real_popen(argv, **kw)
        return real_popen(["bash", "-c", "sleep 30 >/dev/null 2>&1 & echo child $!; wait"], **kw)

    monkeypatch.setattr(install.subprocess, "Popen", popen)
    lines: list[str] = []
    code = install._run("/usr/bin/sudo /usr/bin/apt-get install -y x",
                        ask_password=lambda: "", on_line=lines.append, timeout_seconds=0.5)
    assert code != 0
    child = next(int(line.split()[1]) for line in lines if line.startswith("child "))
    from pathlib import Path

    def gone() -> bool:  # /proc-based: a reparented orphan sits outside our subtree, where os.kill(pid, 0) is guarded
        try:
            return "Z" in (Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split() or ["Z"])[0]
        except OSError:
            return True

    for _ in range(50):  # the child must die with the group, not linger reparented to init
        if gone():
            break
        time.sleep(0.1)
    else:
        subprocess.run(["kill", "-9", str(child)], check=False)
        pytest.fail("grandchild survived the install timeout")
