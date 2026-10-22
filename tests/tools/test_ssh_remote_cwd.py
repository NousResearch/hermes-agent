"""SSH file paths and cwds live in the remote namespace, never the Hermes host's.

In Docker the host subprocess home is ``/opt/data/home``; expanding ``~`` or a
relative path against it and sending the result over SSH names a directory the
remote machine does not have (writes fail, ``cd`` exits 126).
"""

import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.file_tools_paths as paths
import tools.file_tools_write_guards as write_guards
from tools import terminal_tool
from tools import file_tools
from tools.file_tools_write_guards import _check_sensitive_path
from tools.terminal_tool_config import coerce_ssh_remote_cwd

HOST_HOME = "/opt/data/home"


def test_ssh_paths_resolve_on_the_remote_and_stay_guarded(monkeypatch):
    for module in (paths, write_guards):
        monkeypatch.setattr(module, "_terminal_env_type_for_task", lambda task_id="default": "ssh")
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(paths, "_ssh_home_failed_at", {})

    def unreachable(task_id="default"):
        raise ConnectionError("ssh unreachable")

    monkeypatch.setattr(file_tools, "_get_file_ops", unreachable)
    monkeypatch.setenv("TERMINAL_CWD", "~/proj")

    def resolve(p):
        return str(paths._resolve_path_for_task(p, task_id="sess"))

    def go_live(**env):
        key = terminal_tool._resolve_container_task_id("sess")
        terminal_tool._active_environments[key] = SimpleNamespace(**env)

    with patch("hermes_constants.get_subprocess_home", return_value=HOST_HOME):
        # Without a detected remote home, ``~`` stays for the remote shell.
        assert resolve("x.txt") == "~/proj/x.txt"
        assert resolve("~/y.txt") == "~/y.txt"
        assert resolve("~bob/z.txt") == "~bob/z.txt"
        assert _check_sensitive_path("../../../etc/cron.d/evil", "sess") is not None
        go_live(_remote_home="/home/probe", _remote_home_detected=False)  # a guess, not detected
        assert resolve("x.txt") == "~/proj/x.txt"

        # The terminal tool connects inside the failed bring-up's retry window.
        go_live(_remote_home="/config", _remote_home_detected=True)
        assert resolve("x.txt") == "/config/proj/x.txt"
        assert resolve("../../etc/cron.d/evil") == "/etc/cron.d/evil"
        assert _check_sensitive_path("../../etc/cron.d/evil", "sess") is not None
        resolved = paths._resolve_path_for_task("x.txt", task_id="sess")
        assert paths._path_resolution_warning("x.txt", resolved, task_id="sess") is None


@pytest.mark.parametrize(("cwd", "env_type", "expected"), [
    (HOST_HOME, "ssh", "~"),
    (f"{HOST_HOME}/proj", "ssh", "~/proj"),
    ("~bob/src", "ssh", "~bob/src"),
    ("/home/ubuntu", "ssh", "/home/ubuntu"),
    ("/opt/data/homework", "ssh", "/opt/data/homework"),
    (HOST_HOME, "docker", HOST_HOME),
])
def test_ssh_cwd_never_names_the_host_subprocess_home(cwd, env_type, expected):
    with patch("hermes_constants.get_subprocess_home", return_value=HOST_HOME):
        assert coerce_ssh_remote_cwd(cwd, env_type) == expected


def test_ssh_command_cwd_rewrites_the_hermes_home():
    """The official image's data dir is the parent of the subprocess home, so the
    relative rewrite leaves it alone. The SSH wrapper then cds to /opt/data."""
    with ExitStack() as stack:
        stack.enter_context(patch("hermes_constants.get_subprocess_home", return_value=HOST_HOME))
        stack.enter_context(patch("hermes_constants.get_hermes_home", return_value="/opt/data"))
        stack.enter_context(patch("hermes_constants.get_real_home", return_value="/root"))
        assert coerce_ssh_remote_cwd("/opt/data", "ssh") == "~"
        assert coerce_ssh_remote_cwd("/opt/data/", "ssh") == "~"
        assert coerce_ssh_remote_cwd("/opt/data/homework", "ssh") == "/opt/data/homework"
        assert coerce_ssh_remote_cwd("/opt/data/proj", "ssh") == "/opt/data/proj"
        assert terminal_tool._resolve_command_cwd(
            workdir="/opt/data", default_cwd="~", env_type="ssh",
        ) == "~"
        terminal_tool.record_session_cwd("ssh-home", "/opt/data")
        try:
            assert terminal_tool._resolve_command_cwd(
                workdir=None, default_cwd="~", session_key="ssh-home", env_type="ssh",
            ) == "~"
        finally:
            terminal_tool.clear_session_cwd("ssh-home")


def test_ssh_cwd_rewrites_the_hermes_home_without_a_subprocess_home():
    with patch("hermes_constants.get_subprocess_home", return_value=None), \
         patch("hermes_constants.get_hermes_home", return_value="/opt/data"), \
         patch("hermes_constants.get_real_home", return_value="/root"):
        assert coerce_ssh_remote_cwd("/opt/data", "ssh") == "~"
        assert coerce_ssh_remote_cwd("/opt/data/proj", "ssh") == "/opt/data/proj"


def test_ssh_cwd_keeps_a_hermes_home_that_is_the_real_home():
    with patch("hermes_constants.get_subprocess_home", return_value=HOST_HOME), \
         patch("hermes_constants.get_hermes_home", return_value="/opt/data"), \
         patch("hermes_constants.get_real_home", return_value="/opt/data"):
        assert coerce_ssh_remote_cwd("/opt/data", "ssh") == "/opt/data"


def test_ssh_child_workspace_skips_the_hermes_home(tmp_path, monkeypatch):
    hermes = tmp_path / "data"
    hermes.mkdir()
    monkeypatch.chdir(hermes)
    here = os.path.abspath(".")
    sub = os.path.join(here, "home")
    os.mkdir(sub)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", ".")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: here)
    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda env=None: sub)
    monkeypatch.setattr("hermes_constants.get_real_home", lambda env=None: "/root")
    from tools.delegate_tool_progress import _resolve_workspace_hint

    parent = SimpleNamespace(
        _subdirectory_hints=SimpleNamespace(working_dir=here),
        terminal_cwd=sub,
        cwd=here,
    )
    assert _resolve_workspace_hint(parent) is None

    parent.cwd = str(project)
    assert _resolve_workspace_hint(parent) == os.path.abspath(project)


def test_local_child_workspace_keeps_the_process_directory(tmp_path, monkeypatch):
    hermes = tmp_path / "data"
    hermes.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(hermes))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes)
    from tools.delegate_tool_progress import _resolve_workspace_hint

    parent = SimpleNamespace(_subdirectory_hints=None, terminal_cwd=None, cwd=None)
    assert _resolve_workspace_hint(parent) == os.path.abspath(hermes)
