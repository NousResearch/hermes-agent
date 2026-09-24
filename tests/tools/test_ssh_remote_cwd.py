"""SSH file paths must not land on the Hermes host's subprocess home.

Docker sets that home to ``/opt/data/home``. Expanding ``~`` or a relative
path against it, then sending the result to an SSH backend, writes on a
directory the remote machine does not have.
"""

from unittest.mock import patch

import tools.file_tools_paths as paths
import tools.terminal_tool as terminal_tool


def test_ssh_relative_path_stays_off_container_home(monkeypatch):
    monkeypatch.setattr(paths, "_terminal_env_type_for_task", lambda task_id="default": "ssh")
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"sess": "/opt/data/home"})
    with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/home"):
        resolved = paths._resolve_path_for_task("cwd_probe.txt", task_id="sess")
    assert str(resolved) == "~/cwd_probe.txt"
    assert "/opt/data/home" not in str(resolved)


def test_ssh_tilde_is_not_rewritten_to_the_host_home(monkeypatch):
    monkeypatch.setattr(paths, "_terminal_env_type_for_task", lambda task_id="default": "ssh")
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/home"):
        resolved = paths._resolve_path_for_task("~/cwd_probe.txt", task_id="sess")
    assert str(resolved) == "~/cwd_probe.txt"


def test_ssh_keeps_a_real_remote_directory(monkeypatch):
    monkeypatch.setattr(paths, "_terminal_env_type_for_task", lambda task_id="default": "ssh")
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"sess": "/home/ubuntu/COMPRESS"})
    with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/home"):
        resolved = paths._resolve_path_for_task("cwd_probe.txt", task_id="sess")
    assert str(resolved) == "/home/ubuntu/COMPRESS/cwd_probe.txt"


def test_ssh_command_cwd_drops_container_home(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"sess": "/opt/data/home"})
    with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/home"):
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="~", session_key="sess", env_type="ssh")
        explicit = terminal_tool._resolve_command_cwd(
            workdir="/opt/data/home", default_cwd="~", session_key="sess", env_type="ssh")
    assert cwd == "~"
    assert explicit == "~"


def test_ssh_command_cwd_keeps_remote_directory(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"sess": "/home/ubuntu"})
    with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/home"):
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="~", session_key="sess", env_type="ssh")
    assert cwd == "/home/ubuntu"


def test_local_tilde_still_uses_subprocess_home(monkeypatch, tmp_path):
    home = tmp_path / "profile_home"
    home.mkdir()
    monkeypatch.setattr(paths, "_terminal_env_type_for_task", lambda task_id="default": "local")
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    with patch("hermes_constants.get_subprocess_home", return_value=str(home)):
        resolved = paths._resolve_path_for_task("~/cwd_probe.txt", task_id="sess")
    assert str(resolved).startswith(str(home))
