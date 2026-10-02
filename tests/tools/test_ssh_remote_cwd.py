"""SSH file paths and cwds live in the remote namespace, never the Hermes host's.

In Docker the host subprocess home is ``/opt/data/home``; expanding ``~`` or a
relative path against it and sending the result over SSH names a directory the
remote machine does not have (writes fail, ``cd`` exits 126).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.file_tools_paths as paths
import tools.file_tools_write_guards as write_guards
import tools.terminal_tool as terminal_tool
import tools.file_tools as file_tools
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
