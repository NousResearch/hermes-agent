"""A session cwd that IS the directory mounted at /workspace must not be a container cd.

``_is_unusable_container_cwd`` only treats ``/Users``, ``/home``, drive letters,
and relative paths as host paths. An absolute checkout such as ``/mnt/d/...`` or
``/srv/...`` is ``os.path.isabs`` and skips that heuristic, so the mount-equality
check — which runs only after the heuristic says "unusable" — never fires.

Two sites then wrap that host path as ``builtin cd`` inside the container:

  * the live-env write (``register_task_env_overrides`` → ``env.cwd``, which
    file tools exec with ``cd <env.cwd> || exit 126``)
  * the per-command resolver (``_resolve_command_cwd`` → ``env.execute(cwd=...)``,
    which the session wrapper turns into ``builtin cd -- <cwd> || exit 126``)

When the registered cwd is the mounted host directory, both must land on
``/workspace`` instead. In-container paths and an explicit workdir stay as-is.
"""

import pytest

import tools.terminal_tool as tt
from tools.environments.base import BaseEnvironment
from tools.environments.base_session_env import _wrap_command_script


MNT = "/mnt/d/projects/app"
SRV = "/srv/projects/app"


class _FakeEnv:
    def __init__(self, cwd, host_cwd):
        self.env_type = "docker"
        self.cwd = cwd
        self.host_cwd = host_cwd
        self.execute_cwds = []

    def execute(self, command, **kwargs):
        self.execute_cwds.append(kwargs.get("cwd"))
        return {"output": "", "exit_code": 0, "returncode": 0}


def _cd_line(cwd: str) -> str:
    """The ``builtin cd`` line the container wrapper would run for *cwd*."""
    quoted = BaseEnvironment._quote_cwd_for_cd(cwd)
    script = _wrap_command_script(
        "pwd",
        quoted_cwd=quoted,
        quoted_snap="'/tmp/snap'",
        snap_tmp_template="'/tmp/snap.XXXXXX'",
        passthrough_names=(),
        snapshot_ready=False,
        cwd_marker="__HERMES_CWD__",
    )
    return next(line for line in script.splitlines() if line.startswith("builtin cd -- "))


def _config(host_cwd):
    return {
        "env_type": "docker",
        "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "cwd": "/workspace",
        "host_cwd": host_cwd,
        "timeout": 180,
        "lifetime_seconds": 300,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        "docker_env": {},
        "docker_extra_args": [],
        "docker_mount_cwd_to_workspace": True,
        "docker_run_as_host_user": False,
        "docker_forward_env": [],
        "modal_mode": "auto",
    }


@pytest.fixture
def docker_session(monkeypatch):
    """Docker backend with a live env whose host dir is mounted at /workspace."""
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_container_aliases", {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})

    state = {"host": MNT}

    def config():
        return _config(state["host"])

    monkeypatch.setattr(tt, "_get_env_config", config)

    def bind(host):
        state["host"] = host
        env = _FakeEnv("/workspace", host)
        # CWD-only overrides collapse to "default"; the live env is cached there.
        tt._active_environments["default"] = env
        return env

    return bind


class TestMountedHostSessionCwd:
    @pytest.mark.parametrize("host", [MNT, SRV])
    def test_live_env_write_remaps_mounted_host_dir(self, docker_session, host):
        env = docker_session(host)
        tt.register_task_env_overrides("desktop-sess", {"cwd": host, "cwd_source": "session"})
        assert env.cwd == "/workspace"
        # Host surfaces still track the raw workspace; only the container cwd changes.
        assert tt.get_session_cwd("desktop-sess") == host

    @pytest.mark.parametrize("host", [MNT, SRV])
    def test_command_cwd_is_not_the_mounted_host_dir(self, docker_session, host):
        env = docker_session(host)
        task_id = "desktop-sess"
        tt.register_task_env_overrides(task_id, {"cwd": host, "cwd_source": "session"})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
        assert env.execute_cwds, "terminal command never reached env.execute"
        command_cwd = env.execute_cwds[-1]
        cd = _cd_line(command_cwd)
        assert command_cwd == "/workspace"
        assert host not in cd
        assert cd == "builtin cd -- /workspace || exit 126"

    def test_background_command_uses_the_same_remap(self, docker_session, monkeypatch):
        host = MNT
        docker_session(host)
        captured = {}

        def fake_spawn(process_registry, *, env, env_type, command, cwd, **kwargs):
            captured["cwd"] = cwd

            class _Session:
                id = "bg-1"
                pid = 1

            return _Session()

        monkeypatch.setattr("tools.terminal_tool_background._spawn", fake_spawn)
        task_id = "desktop-sess"
        tt.register_task_env_overrides(task_id, {"cwd": host, "cwd_source": "session"})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id, background=True)
        finally:
            tt.clear_task_env_overrides(task_id)
        assert captured["cwd"] == "/workspace"
        assert host not in _cd_line(captured["cwd"])

    def test_in_container_session_cwd_is_preserved(self, docker_session):
        env = docker_session(MNT)
        task_id = "desktop-sess"
        tt.register_task_env_overrides(task_id, {"cwd": "/workspace/task42"})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
        assert env.cwd == "/workspace/task42"
        assert env.execute_cwds[-1] == "/workspace/task42"

    def test_explicit_workdir_still_wins(self, docker_session):
        env = docker_session(MNT)
        task_id = "desktop-sess"
        tt.register_task_env_overrides(task_id, {"cwd": MNT, "cwd_source": "session"})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id, workdir="/workspace/sub")
        finally:
            tt.clear_task_env_overrides(task_id)
        assert env.execute_cwds[-1] == "/workspace/sub"
