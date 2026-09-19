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
import sys

import pytest

import tools.terminal_tool_sudo as terminal_tool


@pytest.mark.linux_only
def test_wraps_sudo_in_systemd_run_pipe_when_no_new_privs(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")

    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true", cwd="/tmp")

    assert wrapped.startswith("/usr/bin/systemd-run")
    assert " --user " in f" {wrapped} " or "--user" in wrapped
    assert "--pipe" in wrapped
    assert "--wait" in wrapped
    assert "--unit=hermes-nnp-sudo-" in wrapped
    assert "--working-directory=/tmp" in wrapped
    assert "sudo -n true" in wrapped
    assert shutil.which("systemd-run") is not None or wrapped.startswith("/usr/bin/systemd-run")


@pytest.mark.linux_only
def test_wrap_units_are_unique(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")
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


def test_trusted_helper_stat_rejects_group_or_world_writable():
    class _St:
        st_mode = 0o100755 | 0o022
        st_uid = 0

    assert terminal_tool._is_trusted_helper_stat(_St()) is False


def test_trusted_helper_stat_accepts_root_owned_0755():
    class _St:
        st_mode = 0o100755
        st_uid = 0

    assert terminal_tool._is_trusted_helper_stat(_St()) is True


@pytest.mark.linux_only
def test_wrap_passes_environment_file(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")
    env_file = tmp_path / "nnp.env"
    env_file.write_text("HERMES_NNP_PROBE=from-profile\n")
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs(
        "sudo -n true", cwd="/tmp", env_file=str(env_file)
    )
    assert f"EnvironmentFile={env_file}" in wrapped


@pytest.mark.linux_only
def test_wrap_unsets_manager_only_names(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs(
        "sudo -n true", unset_names=["HERMES_NNP_SENTINEL"]
    )
    assert "--expand-environment=no" in wrapped
    assert "HERMES_NNP_SENTINEL" in wrapped
    assert "UnsetEnvironment=" in wrapped


def test_manager_keys_to_unset_keeps_run_env_names(monkeypatch):
    class _Completed:
        returncode = 0
        stdout = "HOME=/home/u\nHERMES_NNP_SENTINEL=from-manager\nPATH=/usr/bin\n"

    monkeypatch.setattr(terminal_tool.subprocess, "run", lambda *a, **k: _Completed())
    names = terminal_tool._nnp_manager_keys_to_unset({"HOME": "/tmp", "PATH": "/bin"})
    assert "HERMES_NNP_SENTINEL" in names
    assert "HOME" not in names
    assert "PATH" not in names


def test_write_nnp_env_file_is_owner_only(tmp_path):
    path = terminal_tool._write_nnp_env_file({"HERMES_NNP_PROBE": "xyz", "EMPTY": ""}, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert 'HERMES_NNP_PROBE="xyz"' in text
    if sys.platform != "win32":
        assert (os.stat(path).st_mode & 0o077) == 0


def test_write_nnp_env_file_keeps_empty_values(tmp_path):
    """Empty keys must be written so a user-manager value cannot fill the omission."""
    path = terminal_tool._write_nnp_env_file({"MANAGER_ONLY": ""}, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert 'MANAGER_ONLY=""' in text


def test_write_nnp_env_file_quotes_whitespace_and_escapes(tmp_path):
    path = terminal_tool._write_nnp_env_file(
        {
            "SPACED": " leading and trailing ",
            "QUOTED": 'say "hi"',
            "SLASHED": r"C:\temp\nnp",
        },
        str(tmp_path),
    )
    text = open(path, encoding="utf-8").read()
    assert 'SPACED=" leading and trailing "' in text
    assert r'QUOTED="say \"hi\""' in text
    assert r'SLASHED="C:\\temp\\nnp"' in text


def test_release_nnp_sudo_env_file_is_exactly_once(tmp_path):
    path = tmp_path / "hermes-nnp-env-once.env"
    path.write_text("A=1\n", encoding="utf-8")

    class _Proc:
        pass

    proc = _Proc()
    proc._nnp_sudo_env_file = str(path)
    terminal_tool._release_nnp_sudo_env_file(proc)
    assert not path.exists()
    assert getattr(proc, "_nnp_sudo_env_file", None) is None
    terminal_tool._release_nnp_sudo_env_file(proc)  # idempotent


def test_wait_unlinks_nnp_env_file_after_natural_exit(monkeypatch, tmp_path):
    from tools.environments.local import LocalEnvironment

    path = tmp_path / "hermes-nnp-env-wait.env"
    path.write_text("A=1\n", encoding="utf-8")

    class _Proc:
        def poll(self):
            return 0

    proc = _Proc()
    proc._nnp_sudo_env_file = str(path)
    monkeypatch.setattr(LocalEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        "tools.environments.base.BaseEnvironment._wait_for_process",
        lambda self, p, *a, **k: {"returncode": 0},
    )
    env = LocalEnvironment(cwd=str(tmp_path))
    result = env._wait_for_process(proc, timeout=1)
    assert result["returncode"] == 0
    assert not path.exists()
    assert getattr(proc, "_nnp_sudo_env_file", None) is None


def test_unit_for_kill_comes_from_proc_not_environment():
    class _Proc:
        pass

    proc_a = _Proc()
    proc_b = _Proc()
    proc_a._nnp_sudo_unit = "hermes-nnp-sudo-1-aaaaaaaa.service"
    proc_b._nnp_sudo_unit = "hermes-nnp-sudo-2-bbbbbbbb.service"
    assert terminal_tool._nnp_sudo_unit_from_proc(proc_a) == "hermes-nnp-sudo-1-aaaaaaaa.service"
    assert terminal_tool._nnp_sudo_unit_from_proc(proc_b) == "hermes-nnp-sudo-2-bbbbbbbb.service"
