"""Bot Desktop package install: sudo hand-off and single-flight invariants."""

from __future__ import annotations

import threading

import pytest

from tools.bot_desktop import install, runtime


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
