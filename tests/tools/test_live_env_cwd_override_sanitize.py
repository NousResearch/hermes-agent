"""Invariants for cwd overrides written into LIVE cached envs (#113894, #98723).

``register_task_env_overrides`` applies a ``cwd`` override to the cached live
environment directly. On container backends a raw host path (a desktop/TUI
session registering its workspace) can never be the in-sandbox workdir: every
file-tools ``_exec`` wrapper does ``builtin cd -- <env.cwd> || exit 126``, so a
host cwd poisoned all later file operations with an unrelated ``cd:`` error
while terminal commands kept working (their per-command resolver sanitizes).
"""

import pytest

import tools.terminal_tool as tt


class _FakeEnv:
    def __init__(self, env_type, cwd, host_cwd=None):
        self.env_type = env_type
        self.cwd = cwd
        self.host_cwd = host_cwd

    def execute(self, *a, **k):
        return {"output": "", "returncode": 0}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_container_aliases", {})


WIN_WS = r"C:\Users\rashi\ai_workspace"


@pytest.mark.parametrize(
    "env_type, host_cwd, override, expected_live_cwd",
    [
        # Reported shape: docker cwd passthrough, the host dir IS the one mounted
        # at /workspace -> remap to the in-container view of the same directory.
        ("docker", WIN_WS, WIN_WS, "/workspace"),
        # Host path that is not mounted anywhere: leave the live env untouched.
        ("docker", None, WIN_WS, "/root"),
        ("singularity", None, "/Users/me/workspace", "/root"),
        # In-sandbox override (ACP project switch) still applies on a container.
        ("docker", None, "/workspace/task42", "/workspace/task42"),
        # Non-container backends apply the override verbatim.
        ("local", None, "/proj/two", "/proj/two"),
    ],
)
def test_live_env_cwd_write_is_sanitized_for_container_backends(
    env_type, host_cwd, override, expected_live_cwd
):
    env = _FakeEnv(env_type, "/root", host_cwd=host_cwd)
    # CWD-only overrides collapse to "default": the env is cached under the
    # collapsed key, not the raw task id.
    tt._active_environments["default"] = env
    tt.register_task_env_overrides("sess-abc", {"cwd": override})
    assert env.cwd == expected_live_cwd
    # The session record keeps the RAW path: host-side surfaces track the
    # workspace there and its readers already guard container use.
    assert tt.get_session_cwd("sess-abc") == override


def test_real_builders_tag_env_type_and_record_host_cwd(tmp_path):
    """The sanitizer reads ``env_type``/``host_cwd`` off the live instance, so the
    producers must actually set them: ``_create_environment`` tags the backend and
    ``DockerEnvironment._mount_args`` records the host dir bound at /workspace."""
    import os

    from tools.environments.docker import DockerEnvironment
    from tools.terminal_tool_backends import _create_environment

    assert _create_environment("local", image="", cwd=str(tmp_path), timeout=5).env_type == "local"
    docker = object.__new__(DockerEnvironment)  # _mount_args needs no docker daemon
    docker._persistent = False
    DockerEnvironment._mount_args(docker, [], str(tmp_path), True, "t")
    assert docker.host_cwd == os.path.abspath(str(tmp_path))
    DockerEnvironment._mount_args(docker, [], str(tmp_path / "gone"), True, "t")
    assert docker.host_cwd is None
