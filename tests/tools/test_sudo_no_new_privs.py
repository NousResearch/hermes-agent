"""Desktop Linux: sudo under Electron's inherited NoNewPrivs (#108595).

Packaged Electron with a setuid chrome-sandbox latches PR_SET_NO_NEW_PRIVS on
the main process; the spawned backend inherits it and kernel-refuses sudo's
setuid bit (NOPASSWD and SUDO_PASSWORD both fail with "no new privileges").
Escape hatch: systemd-run --user --pipe so the command runs in a fresh user
unit outside that tree.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import tools.terminal_tool_sudo as terminal_tool


def test_wraps_sudo_in_systemd_run_pipe_when_no_new_privs(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)

    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true")

    assert wrapped != "sudo -n true"
    assert "systemd-run" in wrapped
    assert "--user" in wrapped
    assert "--pipe" in wrapped
    assert "--wait" in wrapped
    assert "sudo -n true" in wrapped


def test_does_not_wrap_when_no_new_privs_is_clear(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: False)

    assert terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true") == "sudo -n true"


def test_does_not_wrap_commands_without_sudo(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)

    assert terminal_tool._wrap_local_command_for_no_new_privs("id -un") == "id -un"


@pytest.mark.skipif(
    not shutil.which("setpriv") or not shutil.which("systemd-run"),
    reason="setpriv + systemd-run required for the kernel-latch harness",
)
def test_wrapped_sudo_does_not_hit_kernel_no_new_privs_latch(monkeypatch):
    """setpriv reproduces Electron's latch; the wrap must change the error."""
    raw = subprocess.run(
        ["setpriv", "--no-new-privs", "sudo", "-n", "true"],
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert raw.returncode != 0
    assert "no new privileges" in (raw.stderr or "").lower()

    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true")
    escaped = subprocess.run(
        ["setpriv", "--no-new-privs", "bash", "-lc", wrapped],
        capture_output=True,
        text=True,
        timeout=12,
    )
    combined = f"{escaped.stdout}\n{escaped.stderr}".lower()
    assert "no new privileges" not in combined
