"""Desktop Linux: sudo under Electron's inherited NoNewPrivs (#108595).

Packaged Electron with a setuid chrome-sandbox latches PR_SET_NO_NEW_PRIVS on
the main process; the spawned backend inherits it and kernel-refuses sudo's
setuid bit (NOPASSWD and SUDO_PASSWORD both fail with "no new privileges").
Escape hatch: systemd-run --user --pipe so the command runs in a fresh user
unit outside that tree.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import tools.terminal_tool_sudo as terminal_tool


def test_wraps_sudo_in_systemd_run_pipe_when_no_new_privs(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)

    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true", cwd="/tmp")

    assert wrapped.startswith("/usr/bin/systemd-run")
    assert " --user " in f" {wrapped} " or "--user" in wrapped
    assert "--pipe" in wrapped
    assert "--wait" in wrapped
    assert "--unit=hermes-nnp-sudo-" in wrapped
    assert "--working-directory=/tmp" in wrapped
    assert "sudo -n true" in wrapped
    assert shutil.which("systemd-run") is not None or wrapped.startswith("/usr/bin/systemd-run")


def test_wrap_units_are_unique(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    a = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true")
    b = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true")
    assert terminal_tool._nnp_sudo_unit_from_command(a) != terminal_tool._nnp_sudo_unit_from_command(b)


def test_does_not_wrap_when_no_new_privs_is_clear(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: False)

    assert terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true") == "sudo -n true"


def test_does_not_wrap_commands_without_sudo(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)

    assert terminal_tool._wrap_local_command_for_no_new_privs("id -un") == "id -un"


def test_does_not_use_untrusted_systemd_run_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: None)
    fake = tmp_path / "systemd-run"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))

    assert terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true") == "sudo -n true"


@pytest.mark.skipif(
    not shutil.which("setpriv") or not os.path.isfile("/usr/bin/systemd-run"),
    reason="setpriv + /usr/bin/systemd-run required for the kernel-latch harness",
)
def test_wrapped_sudo_does_not_hit_kernel_no_new_privs_latch(monkeypatch):
    """setpriv reproduces Electron's latch; wrap must actually reach sudo."""
    raw = subprocess.run(
        ["setpriv", "--no-new-privs", "sudo", "-n", "true"],
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert raw.returncode != 0
    assert "no new privileges" in (raw.stderr or "").lower()

    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true", cwd="/tmp")
    escaped = subprocess.run(
        ["setpriv", "--no-new-privs", "bash", "-lc", wrapped],
        capture_output=True,
        text=True,
        timeout=12,
    )
    combined = f"{escaped.stdout}\n{escaped.stderr}".lower()
    assert "no new privileges" not in combined
    assert "failed to connect" not in combined
    sudo_ran = escaped.returncode == 0 or "password is required" in combined
    assert sudo_ran, combined
    unit = terminal_tool._nnp_sudo_unit_from_command(wrapped)
    assert unit
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "stop", unit],
        capture_output=True,
        timeout=5,
    )
