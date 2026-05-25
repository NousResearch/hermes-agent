import logging
from io import StringIO
import subprocess
import sys
import types

import pytest

from tools.environments import docker as docker_env


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        env=kwargs.get("env"),
        run_as_host_user=kwargs.get("run_as_host_user", False),
    )


def test_ensure_docker_available_logs_and_raises_when_not_found(monkeypatch, caplog):
    """When docker cannot be found, raise a clear error before container setup."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called when docker is missing"),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker executable not found in PATH or known install locations" in str(excinfo.value)
    assert any(
        "no docker executable was found in PATH or known install locations"
        in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_logs_and_raises_on_timeout(monkeypatch, caplog):
    """When docker version times out, surface a helpful error instead of hanging."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["/custom/docker", "version"], timeout=5)

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/custom/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _raise_timeout)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker daemon is not responding" in str(excinfo.value)
    assert any(
        "/custom/docker version' timed out" in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_uses_resolved_executable(monkeypatch):
    """When docker is found outside PATH, preflight should use that resolved path."""

    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/opt/homebrew/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    docker_env._ensure_docker_available()

    assert calls == [
        (["/opt/homebrew/bin/docker", "version"], {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        })
    ]


def test_auto_mount_host_cwd_adds_volume(monkeypatch, tmp_path):
    """Opt-in docker cwd mounting should bind the host cwd to /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # Find the docker run call and check its args
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str


def test_auto_mount_disabled_by_default(monkeypatch, tmp_path):
    """Host cwd should not be mounted unless the caller explicitly opts in."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/root",
        host_cwd=str(project_dir),
        auto_mount_cwd=False,
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" not in run_args_str


def test_auto_mount_skipped_when_workspace_already_mounted(monkeypatch, tmp_path):
    """Explicit user volumes for /workspace should take precedence over cwd mount."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        volumes=[f"{other_dir}:/workspace"],
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{other_dir}:/workspace" in run_args_str
    assert run_args_str.count(":/workspace") == 1


def test_auto_mount_replaces_persistent_workspace_bind(monkeypatch, tmp_path):
    """Persistent mode should still prefer the configured host cwd at /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        persistent_filesystem=True,
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        task_id="test-persistent-auto-mount",
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str
    assert "/sandboxes/docker/test-persistent-auto-mount/workspace:/workspace" not in run_args_str


def test_non_persistent_cleanup_removes_container(monkeypatch):
    """When persistent=false, cleanup() must schedule docker stop + rm."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    popen_cmds = []
    monkeypatch.setattr(
        docker_env.subprocess, "Popen",
        lambda cmd, **kw: (popen_cmds.append(cmd), type("P", (), {"poll": lambda s: 0, "wait": lambda s, **k: None, "returncode": 0, "stdout": iter([]), "stdin": None})())[1],
    )

    env = _make_dummy_env(persistent_filesystem=False, task_id="ephemeral-task")
    assert env._container_id
    container_id = env._container_id

    env.cleanup()

    # Should have stop and rm calls via Popen
    stop_cmds = [c for c in popen_cmds if container_id in str(c) and "stop" in str(c)]
    assert len(stop_cmds) >= 1, f"cleanup() should schedule docker stop for {container_id}"


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = StringIO("")
        self.stdin = None
        self.returncode = 0

    def poll(self):
        return self.returncode


def _make_execute_only_env(forward_env=None):
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env.cwd = "/root"
    env.timeout = 60
    env._forward_env = forward_env or []
    env._env = {}
    env._prepare_command = lambda command: (command, None)
    env._timeout_result = lambda timeout: {"output": f"timed out after {timeout}", "returncode": 124}
    env._container_id = "test-container"
    env._docker_exe = "/usr/bin/docker"
    # Base class attributes needed by unified execute()
    env._session_id = "test123"
    env._snapshot_path = "/tmp/hermes-snap-test123.sh"
    env._cwd_file = "/tmp/hermes-cwd-test123.txt"
    env._cwd_marker = "__HERMES_CWD_test123__"
    env._snapshot_ready = True
    env._last_sync_time = None
    env._init_env_args = []
    return env


def test_init_env_args_uses_hermes_dotenv_for_allowlisted_env(monkeypatch):
    """_build_init_env_args picks up forwarded env vars from .env file at init time."""
    # Use a var that is NOT in _HERMES_PROVIDER_ENV_BLOCKLIST (GITHUB_TOKEN
    # is in the copilot provider's api_key_env_vars and gets stripped).
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "DATABASE_URL=value_from_dotenv" in args_str


def test_init_env_args_prefers_shell_env_over_hermes_dotenv(monkeypatch):
    """Shell env vars take priority over .env file values in init env args."""
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.setenv("DATABASE_URL", "value_from_shell")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "DATABASE_URL=value_from_shell" in args_str
    assert "value_from_dotenv" not in args_str


# ── docker_env tests ──────────────────────────────────────────────


def test_docker_env_appears_in_run_command(monkeypatch):
    """Explicit docker_env values should be passed via -e at docker run time."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock", "GNUPGHOME": "/root/.gnupg"})

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]
    run_args_str = " ".join(run_args)
    assert "SSH_AUTH_SOCK=/run/user/1000/ssh-agent.sock" in run_args_str
    assert "GNUPGHOME=/root/.gnupg" in run_args_str


def test_docker_env_appears_in_init_env_args(monkeypatch):
    """Explicit docker_env values should appear in _build_init_env_args."""
    env = _make_execute_only_env()
    env._env = {"MY_VAR": "my_value"}

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "MY_VAR=my_value" in args_str


def test_forward_env_overrides_docker_env_in_init_args(monkeypatch):
    """docker_forward_env should override docker_env for the same key."""
    env = _make_execute_only_env(forward_env=["MY_KEY"])
    env._env = {"MY_KEY": "static_value"}

    monkeypatch.setenv("MY_KEY", "dynamic_value")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "MY_KEY=dynamic_value" in args_str
    assert "MY_KEY=static_value" not in args_str


def test_docker_env_and_forward_env_merge_in_init_args(monkeypatch):
    """docker_env and docker_forward_env with different keys should both appear."""
    env = _make_execute_only_env(forward_env=["TOKEN"])
    env._env = {"SSH_AUTH_SOCK": "/run/user/1000/agent.sock"}

    monkeypatch.setenv("TOKEN", "secret123")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "SSH_AUTH_SOCK=/run/user/1000/agent.sock" in args_str
    assert "TOKEN=secret123" in args_str



def test_normalize_env_dict_filters_invalid_keys():
    """_normalize_env_dict should reject invalid variable names."""
    result = docker_env._normalize_env_dict({
        "VALID_KEY": "ok",
        "123bad": "rejected",
        "": "rejected",
        "also valid": "rejected",  # spaces invalid
        "GOOD": "ok",
    })
    assert result == {"VALID_KEY": "ok", "GOOD": "ok"}


def test_normalize_env_dict_coerces_scalars():
    """_normalize_env_dict should coerce int/float/bool to str."""
    result = docker_env._normalize_env_dict({
        "PORT": 8080,
        "DEBUG": True,
        "RATIO": 0.5,
    })
    assert result == {"PORT": "8080", "DEBUG": "True", "RATIO": "0.5"}


def test_normalize_env_dict_rejects_non_dict():
    """_normalize_env_dict should return empty dict for non-dict input."""
    assert docker_env._normalize_env_dict("not a dict") == {}
    assert docker_env._normalize_env_dict(None) == {}
    assert docker_env._normalize_env_dict([]) == {}


def test_normalize_env_dict_rejects_complex_values():
    """_normalize_env_dict should reject list/dict values."""
    result = docker_env._normalize_env_dict({
        "GOOD": "string",
        "BAD_LIST": [1, 2, 3],
        "BAD_DICT": {"nested": True},
    })
    assert result == {"GOOD": "string"}


def test_security_args_include_setuid_setgid_for_gosu_drop(monkeypatch):
    """The default (run_as_host_user=False) invocation must include SETUID and
    SETGID caps so the image entrypoint can drop from root to the non-root
    `hermes` user via gosu.

    Without these caps gosu exits with
    ``error: failed switching to 'hermes': operation not permitted``
    and the container exits immediately (exit 1) before running any work.

    `no-new-privileges` is kept, so gosu still cannot escalate back to root
    after the drop — the drop is a one-way transition performed before the
    `no_new_privs` bit is enforced on the exec boundary.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env()

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    added = {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--cap-add"
    }
    assert "SETUID" in added, "SETUID cap missing — gosu drop in entrypoint will fail"
    assert "SETGID" in added, "SETGID cap missing — gosu drop in entrypoint will fail"


# ── run_as_host_user tests ────────────────────────────────────────


def test_run_as_host_user_passes_uid_gid(monkeypatch):
    """With run_as_host_user=True, --user <uid>:<gid> is added to docker run."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(docker_env.os, "getgid", lambda: 5678, raising=False)
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(run_as_host_user=True)

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    # --user must be present and must be paired with "1234:5678"
    assert "--user" in run_args, f"--user flag missing from docker run args: {run_args}"
    idx = run_args.index("--user")
    assert run_args[idx + 1] == "1234:5678", (
        f"expected --user 1234:5678, got --user {run_args[idx + 1]}"
    )


def test_run_as_host_user_drops_setuid_setgid_caps(monkeypatch):
    """When --user is passed, the container never needs gosu, so SETUID/SETGID
    caps are omitted for a tighter security posture."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(docker_env.os, "getgid", lambda: 1000, raising=False)
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(run_as_host_user=True)

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    run_args = run_calls[0][0]

    added = {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--cap-add"
    }
    assert "SETUID" not in added, (
        "SETUID cap should be dropped when running as host user — no gosu drop is needed"
    )
    assert "SETGID" not in added, (
        "SETGID cap should be dropped when running as host user — no gosu drop is needed"
    )
    # Core non-privilege-drop caps must still be there (pip/npm/apt need them).
    assert "DAC_OVERRIDE" in added
    assert "CHOWN" in added
    assert "FOWNER" in added


def test_run_as_host_user_default_off(monkeypatch):
    """Without the opt-in, no --user flag is emitted — preserving existing behavior."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env()  # run_as_host_user defaults to False

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    run_args = run_calls[0][0]
    assert "--user" not in run_args, (
        f"--user should not be in docker run args when opt-in is off: {run_args}"
    )


def test_run_as_host_user_warns_and_skips_when_no_posix_ids(monkeypatch, caplog):
    """On platforms without POSIX getuid/getgid, log a warning and leave the
    container at its image default user (no --user flag, full cap set)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    # Simulate a platform where os.getuid is absent (e.g. Windows host).
    monkeypatch.delattr(docker_env.os, "getuid", raising=False)
    monkeypatch.delattr(docker_env.os, "getgid", raising=False)
    calls = _mock_subprocess_run(monkeypatch)

    with caplog.at_level(logging.WARNING):
        _make_dummy_env(run_as_host_user=True)

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    run_args = run_calls[0][0]

    assert "--user" not in run_args
    # Fall back to the full cap set since the container still starts as root.
    added = {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--cap-add"
    }
    assert "SETUID" in added
    assert "SETGID" in added
    assert any(
        "does not expose POSIX uid/gid" in rec.getMessage()
        for rec in caplog.records
    ), "expected a warning when POSIX ids are unavailable"
