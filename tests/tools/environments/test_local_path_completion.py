"""Terminal-subshell PATH completion in ``tools/environments/local.py``.

A backend started by a non-interactive SSH session, systemd or a GUI launcher
inherits a PATH without ``~/.local/bin`` (only the login shell adds it), so CLIs
installed there were ``command not found`` from the terminal tool (#111778).
"""

import os
import subprocess
import sys

import pytest

from tools.environments import local as local_mod
from tools.environments.local import _append_missing_sane_path_entries, _make_run_env

pytestmark = pytest.mark.platforms("posix")  # POSIX PATH completion only


def test_existing_user_local_bin_appended_after_inherited_entries(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    entries = _make_run_env({})["PATH"].split(os.pathsep)

    assert entries[:2] == ["/usr/bin", "/bin"]
    assert entries.count(str(local_bin)) == 1
    # Already on PATH: position kept, no duplicate appended.
    already = _append_missing_sane_path_entries(f"{local_bin}:/usr/bin").split(":")
    assert already[0] == str(local_bin) and already.count(str(local_bin)) == 1


def test_missing_user_local_bin_not_appended(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])

    assert ".local" not in _append_missing_sane_path_entries("/usr/bin:/bin")


def test_committed_python_precedes_store_tool_python_in_terminal_path(monkeypatch, tmp_path):
    """A PM tool Python on the inherited PATH must not shadow dependency imports."""
    tool_bin = tmp_path / "tools" / "python" / "bin"
    venv_bin = tmp_path / "installs" / "selected" / "environments" / "current" / "venv" / "bin"
    tool_bin.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    (venv_bin / "python3").write_text("#!/bin/sh\necho dependency-venv\n")
    (tool_bin / "python3").write_text("#!/bin/sh\necho bare-tool-python\n")
    for path in (venv_bin / "python3", tool_bin / "python3"):
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"/custom/bin:{tool_bin}:/usr/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(tool_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _repo: venv_bin.parent)
    monkeypatch.setattr("pm.env_for", lambda *_args, **_kwargs: {"PATH": str(tool_bin)})

    entries = _make_run_env({})["PATH"].split(os.pathsep)
    assert entries[0] == "/custom/bin"
    assert entries.index(str(venv_bin)) < entries.index(str(tool_bin))
    assert entries.count(str(venv_bin)) == 1
    output = subprocess.check_output(["python3"], env={"PATH": os.pathsep.join(entries)}, text=True)
    assert output.strip() == "dependency-venv"


def test_without_committed_venv_does_not_reorder_python(monkeypatch, tmp_path):
    tool_bin = tmp_path / "tools" / "python" / "bin"
    tool_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", f"/custom/bin:{tool_bin}:/usr/bin")
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(tool_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _repo: None)
    entries = _make_run_env({})["PATH"].split(os.pathsep)
    assert entries[:2] == ["/custom/bin", str(tool_bin)]


def test_committed_venv_already_after_tool_python_moves_once(monkeypatch, tmp_path):
    tool_bin = tmp_path / "tools" / "python" / "bin"
    venv = tmp_path / "installs" / "selected" / "venv"
    tool_bin.mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setenv("PATH", f"/custom/bin:{tool_bin}:{venv / 'bin'}:/usr/bin")
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(tool_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _repo: venv)
    monkeypatch.setattr("pm.env_for", lambda *_args, **_kwargs: {"PATH": str(tool_bin)})
    entries = _make_run_env({})["PATH"].split(os.pathsep)
    assert entries[:3] == ["/custom/bin", str(venv / "bin"), str(tool_bin)]
    assert entries.count(str(venv / "bin")) == 1


def test_unrelated_managed_runtime_order_is_unchanged(monkeypatch, tmp_path):
    node_bin = tmp_path / "tools" / "node" / "bin"
    tool_bin = tmp_path / "tools" / "python" / "bin"
    venv = tmp_path / "installs" / "selected" / "venv"
    for path in (node_bin, tool_bin, venv / "bin"):
        path.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/custom/bin:/usr/bin")
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(node_bin), str(tool_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _repo: venv)
    monkeypatch.setattr("pm.env_for", lambda *_args, **_kwargs: {"PATH": str(tool_bin)})
    entries = _make_run_env({})["PATH"].split(os.pathsep)
    assert entries.index(str(node_bin)) < entries.index(str(venv / "bin")) < entries.index(str(tool_bin))
