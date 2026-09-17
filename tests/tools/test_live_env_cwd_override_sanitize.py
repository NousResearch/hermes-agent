"""Regression tests for cwd-override sanitization on LIVE cached envs.

``register_task_env_overrides`` applies a ``cwd`` override to the cached live
environment directly. On container backends a raw host path (a desktop/TUI
session registering its workspace) can never be the in-sandbox workdir: every
file-tools ``_exec`` wrapper does ``builtin cd -- <env.cwd> || exit 126``, so a
host cwd poisoned all later file operations with an unrelated ``cd:`` error
while terminal commands kept working (their per-command resolver sanitizes).
These tests pin the live-env write so a host cwd is either remapped to the
in-container view of the same mounted directory or not applied at all.
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


class TestLiveEnvCwdSanitized:
    def test_windows_host_cwd_not_applied_to_docker_env(self):
        # The reported bug: a desktop session registers its Windows workspace,
        # env.cwd becomes C:\... and every file op dies on `cd: C:\...`.
        env = _FakeEnv("docker", "/workspace")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides(
            "default", {"cwd": r"C:\Users\rashi\ai_workspace"}
        )
        assert env.cwd == "/workspace"

    def test_posix_host_cwd_not_applied_to_docker_env(self):
        env = _FakeEnv("docker", "/root")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides("default", {"cwd": "/Users/me/workspace"})
        assert env.cwd == "/root"

    def test_mounted_host_cwd_remaps_to_workspace(self):
        # Docker cwd passthrough: the host dir IS mounted at /workspace, so the
        # session's directory stays reachable — remap instead of discarding.
        env = _FakeEnv("docker", "/workspace", host_cwd="/Users/me/ws")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides("default", {"cwd": "/Users/me/ws"})
        assert env.cwd == "/workspace"

    def test_mounted_windows_host_cwd_remaps_to_workspace(self):
        # The exact reported shape: a Windows host cwd with docker cwd
        # passthrough mounted at /workspace.
        env = _FakeEnv("docker", "/workspace", host_cwd=r"C:\Users\rashi\ai_workspace")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides(
            "default", {"cwd": r"C:\Users\rashi\ai_workspace"}
        )
        assert env.cwd == "/workspace"

    def test_host_cwd_not_applied_on_singularity(self):
        env = _FakeEnv("singularity", "/root")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides("default", {"cwd": "/Users/me/workspace"})
        assert env.cwd == "/root"

    def test_in_container_override_applied_verbatim(self):
        env = _FakeEnv("docker", "/root")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides("default", {"cwd": "/workspace/task42"})
        assert env.cwd == "/workspace/task42"

    def test_local_env_applies_override_verbatim(self):
        # Non-container backends must keep applying overrides raw: ACP
        # session/load project-root switching relies on it.
        env = _FakeEnv("local", "/start")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides("default", {"cwd": "/proj/two"})
        assert env.cwd == "/proj/two"

    def test_session_record_keeps_raw_host_path(self):
        # The record is the tracking surface for host workspaces; only the
        # live-env write is sanitized.
        tt.register_task_env_overrides("default", {"cwd": "/Users/me/workspace"})
        assert tt.get_session_cwd("default") == "/Users/me/workspace"

    def test_env_found_under_collapsed_container_id(self):
        # CWD-only overrides collapse to "default": the env may be cached under
        # the collapsed key, not the raw task id.
        env = _FakeEnv("docker", "/workspace")
        tt._active_environments["default"] = env
        tt.register_task_env_overrides(
            "sess-abc", {"cwd": r"C:\Users\rashi\ai_workspace"}
        )
        assert env.cwd == "/workspace"


class TestSanitizerUnit:
    def test_none_env_type_returns_cwd(self):
        # Env without the backend tag (pre-existing instances, __slots__ plugin
        # providers): do not guess, apply verbatim like before the tag existed.
        env = _FakeEnv(None, "/root")
        assert tt._sanitize_cwd_for_live_env(env, "/any/path") == "/any/path"

    def test_docker_env_mounted_remap(self, monkeypatch):
        monkeypatch.setattr("os.path.isdir", lambda p: True)
        env = _FakeEnv("docker", "/workspace", host_cwd="/Users/me/ws")
        assert tt._sanitize_cwd_for_live_env(env, "/Users/me/ws") == "/workspace"

    def test_docker_env_unmounted_host_path_returns_none(self):
        env = _FakeEnv("docker", "/root", host_cwd=None)
        assert tt._sanitize_cwd_for_live_env(env, "/Users/me/ws") is None

    def test_relative_cwd_not_applied_to_docker_env(self):
        env = _FakeEnv("docker", "/root")
        assert tt._sanitize_cwd_for_live_env(env, "relative/dir") is None
