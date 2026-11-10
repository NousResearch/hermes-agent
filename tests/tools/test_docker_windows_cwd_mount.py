"""Docker must bind a configured Windows workspace even when /workspace is taken.

A volume that already claims ``/workspace`` used to set
``workspace_explicitly_mounted`` and skip the configured working directory.
Tools then refused that host path because nothing in the container mapped it.
The bind has to land (at ``/workspace`` when free, otherwise a second mount)
and tools have to follow that mount — any drive, any directory, not one user.
"""

import os

import tools.terminal_tool as tt
from tools.environments.docker import DockerEnvironment


WIN_WS = r"D:\Work\proj"
WIN_FILE = r"D:\Work\proj\Downloads\clip.jpg"


def _mount(volumes, host_cwd, auto_mount=True):
    env = object.__new__(DockerEnvironment)
    env._persistent = False
    volume_args, _writable = DockerEnvironment._mount_args(
        env, volumes, host_cwd, auto_mount, "t")
    return env, volume_args


def _preserve_drive_path(monkeypatch, existing: str):
    """POSIX ``abspath`` would prefix the process cwd onto ``D:\\...`` and hide it."""
    real_abspath = os.path.abspath
    real_isdir = os.path.isdir

    def abspath(path):
        if isinstance(path, str) and len(path) >= 3 and path[1] == ":" and path[2] in "\\/":
            return path
        return real_abspath(path)

    def isdir(path):
        return path == existing or real_isdir(path)

    monkeypatch.setattr(os.path, "abspath", abspath)
    monkeypatch.setattr(os.path, "isdir", isdir)


class TestClaimedWorkspaceStillBindsConfiguredCwd:
    def test_real_directory_is_bound_beside_an_existing_workspace_volume(self, tmp_path):
        host = tmp_path / "proj"
        host.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        env, volume_args = _mount([f"{other}:/workspace"], str(host), auto_mount=True)

        specs = [arg for arg in volume_args if arg != "-v"]
        assert any(spec.startswith(f"{other}:/workspace") for spec in specs)
        mount = env.host_cwd_mount
        assert env.host_cwd == os.path.abspath(str(host))
        assert mount and mount != "/workspace"
        assert any(spec == f"{os.path.abspath(str(host))}:{mount}" for spec in specs)

    def test_windows_drive_cwd_binds_when_workspace_is_claimed(self, monkeypatch):
        _preserve_drive_path(monkeypatch, WIN_WS)
        env, volume_args = _mount(
            [r"E:\elsewhere:/workspace"], WIN_WS, auto_mount=False)

        specs = [arg for arg in volume_args if arg != "-v"]
        mount = env.host_cwd_mount
        assert env.host_cwd == WIN_WS
        assert mount and mount != "/workspace"
        assert f"{WIN_WS}:{mount}" in specs
        # The volume that already owns /workspace stays; we do not steal it.
        assert any(":/workspace" in spec and WIN_WS not in spec for spec in specs)

    def test_same_windows_directory_already_at_workspace_points_tools_there(self, monkeypatch):
        _preserve_drive_path(monkeypatch, WIN_WS)
        env, volume_args = _mount([f"{WIN_WS}:/workspace"], WIN_WS, auto_mount=False)

        specs = [arg for arg in volume_args if arg != "-v"]
        assert specs.count(f"{WIN_WS}:/workspace") == 1
        assert env.host_cwd == WIN_WS
        assert env.host_cwd_mount == "/workspace"


class TestToolsFollowTheMount:
    def test_live_cwd_and_child_paths_use_the_second_mount(self):
        mount = "/host-cwd"
        env = type("E", (), {"env_type": "docker", "host_cwd": WIN_WS, "host_cwd_mount": mount})()
        assert tt._sanitize_cwd_for_live_env(env, WIN_WS) == mount
        assert tt._sanitize_cwd_for_live_env(env, WIN_FILE) == f"{mount}/Downloads/clip.jpg"

    def test_file_ops_rewrite_a_mounted_windows_path(self):
        from tools.file_operations import ShellFileOperations

        env = type("E", (), {
            "env_type": "docker", "cwd": "/host-cwd",
            "host_cwd": WIN_WS, "host_cwd_mount": "/host-cwd",
        })()
        ops = ShellFileOperations(env, cwd="/host-cwd")
        assert ops._expand_path(WIN_FILE) == "/host-cwd/Downloads/clip.jpg"
        assert ops._expand_path("/workspace/keep") == "/workspace/keep"

    def test_absolute_mounted_host_dir_follows_the_second_mount(self, monkeypatch):
        """A /mnt or /srv host dir (no /Users prefix) bound beside a claimed /workspace."""
        host, mount = "/mnt/d/projects/app", "/host-cwd"
        env = type("E", (), {"env_type": "docker", "host_cwd": host, "host_cwd_mount": mount})()
        assert tt._sanitize_cwd_for_live_env(env, host) == mount

        monkeypatch.setattr(tt, "_session_cwd", {"sess": host})
        assert tt._resolve_command_cwd(
            workdir=None, default_cwd=mount, session_key="sess", env_type="docker",
            mounted_host=host, env=env,
        ) == mount
