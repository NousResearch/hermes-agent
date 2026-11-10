"""SSH file paths and cwds live in the remote namespace, never the Hermes host's.

In Docker the host subprocess home is ``/opt/data/home``; expanding ``~`` or a
relative path against it and sending the result over SSH names a directory the
remote machine does not have (writes fail, ``cd`` exits 126).
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.file_tools_paths as paths
import tools.file_tools_write_guards as write_guards
from tools import terminal_tool
from tools import file_tools
from tools.file_tools_write_guards import _check_sensitive_path
from hermes_cli._startup_fast import project_root_str
from tools import terminal_tool_config
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


@pytest.fixture
def image_home(tmp_path, monkeypatch):
    """The official image: HOME and HERMES_HOME are both the data dir, its ``home`` is the subprocess HOME."""
    root = os.path.join(os.path.realpath(tmp_path), "data")
    os.makedirs(os.path.join(root, "home"))
    monkeypatch.setattr("hermes_constants.is_container", lambda: True)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: Path(root))
    monkeypatch.setattr("hermes_constants.get_real_home", lambda env=None: root)
    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda env=None: os.path.join(root, "home"))
    terminal_tool_config._launch_dirs.cache_clear()
    yield root
    terminal_tool_config._launch_dirs.cache_clear()


def test_ssh_cwd_rewrites_the_image_hermes_home(image_home):
    """The data dir is the image user's HOME and the parent of the subprocess home, so neither older rule
    caught it: the SSH wrapper cd'd to /opt/data on the target and exited 126."""
    root = image_home
    assert coerce_ssh_remote_cwd(root, "ssh") == "~"
    assert coerce_ssh_remote_cwd(root + "/", "ssh") == "~"
    assert coerce_ssh_remote_cwd(f"{root}/home/proj", "ssh") == "~/proj"
    assert coerce_ssh_remote_cwd(f"{root}/homework", "ssh") == f"{root}/homework"
    assert coerce_ssh_remote_cwd(f"{root}/proj", "ssh") == f"{root}/proj"
    assert coerce_ssh_remote_cwd(project_root_str(), "ssh") == "~"
    assert coerce_ssh_remote_cwd(root, "docker") == root
    assert coerce_ssh_remote_cwd("/srv/\x00bad", "ssh") == "/srv/\x00bad"
    assert terminal_tool._resolve_command_cwd(workdir=root, default_cwd="~", env_type="ssh") == "~"
    terminal_tool.record_session_cwd("ssh-home", root)
    try:
        assert terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="~", session_key="ssh-home", env_type="ssh",
        ) == "~"
    finally:
        terminal_tool.clear_session_cwd("ssh-home")


def test_ssh_cwd_keeps_relative_and_symlinked_paths(image_home, tmp_path, monkeypatch):
    """Only the exact spelling of a launch dir is rewritten: a symlink on this machine, or the process cwd,
    says nothing about the SSH target."""
    alias = tmp_path / "alias"
    alias.symlink_to(image_home)
    monkeypatch.chdir(image_home)
    assert coerce_ssh_remote_cwd(str(alias), "ssh") == str(alias)
    assert coerce_ssh_remote_cwd(".", "ssh") == "."
    assert coerce_ssh_remote_cwd("home", "ssh") == "home"


def test_every_ssh_env_starts_in_the_remote_cwd(image_home, tmp_path, monkeypatch):
    """execute_code, image tools, the prompt probe and the dashboard files pane build SSHEnvironment from the raw
    config cwd."""
    import tools.environments.ssh as ssh_env

    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    for cwd, expected in ((image_home, "~"), (f"{image_home}/home/p", "~/p"), ("/srv/p", "/srv/p"), ("~/q", "~/q")):
        env = ssh_env.SSHEnvironment(host="h", user="u", cwd=cwd, probe_only=True)
        assert env.cwd == expected
    # The terminal tool coerces before building: a second pass keeps a rewritten subprocess-home path.
    assert coerce_ssh_remote_cwd(coerce_ssh_remote_cwd(f"{image_home}/home/p", "ssh"), "ssh") == "~/p"


def test_files_pane_maps_a_named_profiles_dirs_in_its_own_scope(image_home, monkeypatch):
    """The dashboard dials the SSH adapter after leaving the requested profile's scope, so the cwd is coerced
    while that profile's home is bound."""
    from contextlib import contextmanager

    import hermes_constants
    from hermes_cli import ssh_workspace_fs
    from hermes_cli.web_routers import files as file_routes

    profile = os.path.join(image_home, "profiles", "work")
    os.makedirs(os.path.join(profile, "home"))
    monkeypatch.setattr("hermes_constants.get_hermes_home",
                        lambda: Path(hermes_constants.get_hermes_home_override() or image_home))

    @contextmanager
    def scope(_name):
        token = hermes_constants.set_hermes_home_override(profile)
        try:
            yield Path(profile)
        finally:
            hermes_constants.reset_hermes_home_override(token)

    captured = {}
    monkeypatch.setattr(file_routes, "_profile_scope", scope)
    monkeypatch.setattr(file_routes, "load_config", lambda: {"terminal": {"backend": "ssh", "cwd": profile}})
    monkeypatch.setattr(file_routes, "load_env", lambda: {"TERMINAL_SSH_HOST": "h", "TERMINAL_SSH_USER": "u"})
    monkeypatch.setattr(ssh_workspace_fs, "get_ssh_workspace_fs", lambda _key, terminal: captured.update(terminal))
    file_routes._fs_backend("work")
    assert captured["cwd"] == "~"


def test_ssh_cwd_rewrites_the_hermes_home_without_a_subprocess_home(image_home, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda env=None: None)
    assert coerce_ssh_remote_cwd(image_home, "ssh") == "~"
    assert coerce_ssh_remote_cwd(f"{image_home}/./home", "ssh") == "~"
    assert coerce_ssh_remote_cwd(f"{image_home}/proj", "ssh") == f"{image_home}/proj"


def test_ssh_cwd_rewrites_the_root_under_a_named_profile(image_home, monkeypatch):
    profile = os.path.join(image_home, "profiles", "work")
    os.makedirs(os.path.join(profile, "home"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: Path(profile))
    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda env=None: os.path.join(profile, "home"))
    for launch in (image_home, f"{image_home}/home", profile, f"{profile}/home"):
        assert coerce_ssh_remote_cwd(launch, "ssh") == "~"
    assert coerce_ssh_remote_cwd(f"{image_home}/proj", "ssh") == f"{image_home}/proj"


def test_ssh_cwd_keeps_the_hermes_home_on_a_host(image_home, monkeypatch):
    """Outside a container the SSH target may share the layout (an NFS home holding ~/.hermes)."""
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)
    assert coerce_ssh_remote_cwd(image_home, "ssh") == image_home


def test_ssh_child_workspace_skips_the_hermes_home(image_home, tmp_path, monkeypatch):
    under_sub = os.path.join(image_home, "home", "x")
    os.mkdir(under_sub)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(image_home)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", ".")
    from tools.delegate_tool_progress import _resolve_workspace_hint

    parent = SimpleNamespace(
        _subdirectory_hints=SimpleNamespace(working_dir=image_home), terminal_cwd=under_sub, cwd=image_home,
    )
    assert _resolve_workspace_hint(parent) is None

    parent.cwd = str(project)
    assert _resolve_workspace_hint(parent) == os.path.abspath(project)


def test_local_child_workspace_keeps_the_process_directory(image_home, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", image_home)
    from tools.delegate_tool_progress import _resolve_workspace_hint

    parent = SimpleNamespace(_subdirectory_hints=None, terminal_cwd=None, cwd=None)
    assert _resolve_workspace_hint(parent) == image_home

@pytest.mark.parametrize("remote_cwd", ["~/project", "project"])
def test_ssh_child_workspace_never_expands_remote_relative_cwd_on_host(
    image_home, monkeypatch, remote_cwd,
):
    """A remote-relative cwd must not become a host-only WORKSPACE PATH in a child's prompt."""
    from tools.delegate_tool_progress import _resolve_workspace_hint

    # An unrelated host directory happens to have the remote cwd's name.
    # On SSH, ~ and relative paths must resolve on the target, never here.
    (Path(image_home) / "project").mkdir()
    monkeypatch.setenv("HOME", image_home)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", remote_cwd)
    monkeypatch.chdir(image_home)
    parent = SimpleNamespace(_subdirectory_hints=None, terminal_cwd=None, cwd=None)
    assert _resolve_workspace_hint(parent) is None

