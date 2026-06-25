import logging
from io import StringIO
import subprocess

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
        forward_env=kwargs.get("forward_env"),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        env=kwargs.get("env"),
        run_as_host_user=kwargs.get("run_as_host_user", False),
        extra_args=kwargs.get("extra_args", []),
        persist_across_processes=kwargs.get("persist_across_processes", True),
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
            "stdin": subprocess.DEVNULL,
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
    """When persist_across_processes=false, cleanup() must docker stop AND
    docker rm so containers don't leak across hermes processes.

    Updated for issue #20561: the previous implementation used fire-and-forget
    ``subprocess.Popen("... &", shell=True)`` which raced with parent exit;
    the new implementation uses ``subprocess.run`` on a daemon thread with
    bounded timeouts. See test_cleanup_with_persist_disabled_stops_and_rms
    for the full behavior contract.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    # Run the worker thread synchronously so assertions can observe its work.
    import threading
    monkeypatch.setattr(threading, "Thread", _FakeThread)

    env = docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=60,
        task_id="ephemeral-task", persistent_filesystem=False,
        persist_across_processes=False,
    )
    container_id = env._container_id
    assert container_id

    # Capture cleanup-time docker calls (everything before this was init).
    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capture(cmd, **kw):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kw))
        return real_run(cmd, **kw)

    monkeypatch.setattr(docker_env.subprocess, "run", _capture)
    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and c[0][1:2] == ["stop"]]
    assert stops, f"cleanup() should docker stop {container_id}; got {cleanup_calls}"


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


def test_init_env_args_uses_hermes_dotenv_for_empty_shell_env(monkeypatch):
    """A transient empty-string in the live env must fall back to .env, not win.

    Regression: the disk fallback used to fire only on `value is None`, so a
    present-but-empty `MY_SECRET=""` skipped it and was forwarded as `-e
    MY_SECRET=`, clobbering the correct value sitting in ~/.hermes/.env.
    """
    env = _make_execute_only_env(["MY_SECRET"])

    monkeypatch.setenv("MY_SECRET", "")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"MY_SECRET": "value_from_dotenv"})

    args = env._build_init_env_args()

    # Assert on the resolved value, not the printed -e flag: the disk value
    # must win and a blank "MY_SECRET=" flag must never be emitted.
    assert "MY_SECRET=value_from_dotenv" in args
    assert "MY_SECRET=" not in args


def test_init_env_args_never_forwards_blank_secret(monkeypatch):
    """A legitimately-empty key with no disk value is not forwarded as -e KEY=."""
    env = _make_execute_only_env(["MY_SECRET"])

    monkeypatch.setenv("MY_SECRET", "")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()

    # The key must not appear at all — not even as an empty -e MY_SECRET= flag.
    assert not any(a.startswith("MY_SECRET=") for a in args)
    assert "MY_SECRET" not in " ".join(args)


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


def _node_options_from_run(calls):
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    args = run_calls[0][0]
    for i, a in enumerate(args):
        if a == "-e" and i + 1 < len(args) and args[i + 1].startswith("NODE_OPTIONS="):
            return args[i + 1].split("=", 1)[1]
    return None


def test_egress_node_options_overrides_conflicting_ca_flag(monkeypatch):
    """maxpetrusenko P1: a conflicting docker_env NODE_OPTIONS CA-mode flag
    (--use-bundled-ca) must be replaced by the egress-required --use-openssl-ca,
    not left to survive alongside it (final Node trust would depend on order)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env, "_egress_proxy_args_for_docker",
        lambda: ([], {"_HERMES_EGRESS_NODE_OPTIONS_APPEND": "--use-openssl-ca"}, []),
    )
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"NODE_OPTIONS": "--max-old-space-size=8192 --use-bundled-ca"})

    node_opts = (_node_options_from_run(calls) or "").split()
    assert "--use-openssl-ca" in node_opts, "egress CA flag must be present"
    assert "--use-bundled-ca" not in node_opts, "conflicting CA flag must be stripped"
    # Operator's unrelated tuning must be preserved.
    assert "--max-old-space-size=8192" in node_opts


def test_egress_node_options_preserves_operator_tuning(monkeypatch):
    """Non-conflicting operator NODE_OPTIONS survive the egress append-merge."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env, "_egress_proxy_args_for_docker",
        lambda: ([], {"_HERMES_EGRESS_NODE_OPTIONS_APPEND": "--use-openssl-ca"}, []),
    )
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"NODE_OPTIONS": "--max-old-space-size=4096"})

    node_opts = (_node_options_from_run(calls) or "").split()
    assert "--use-openssl-ca" in node_opts
    assert "--max-old-space-size=4096" in node_opts


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


def test_security_args_include_setuid_setgid_for_privdrop(monkeypatch):
    """The default (run_as_host_user=False) invocation must include SETUID and
    SETGID caps so the image's init can drop from root to a non-root user
    (e.g. via ``s6-setuidgid`` in the bundled Hermes image, or ``gosu``/``su``
    in user-provided images).

    Without these caps the privilege-drop helper fails with
    ``operation not permitted`` and the container exits immediately (exit 1)
    before running any work.

    ``no-new-privileges`` is kept, so the dropped process still cannot
    escalate back to root after the drop — the drop is a one-way transition
    performed before the ``no_new_privs`` bit is enforced on the exec boundary.
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
    assert "SETUID" in added, "SETUID cap missing — image privilege-drop will fail"
    assert "SETGID" in added, "SETGID cap missing — image privilege-drop will fail"


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
    """When --user is passed, the container already starts unprivileged and
    never needs a privilege drop, so SETUID/SETGID caps are omitted for a
    tighter security posture."""
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
        "SETUID cap should be dropped when running as host user — no privilege drop is needed"
    )
    assert "SETGID" not in added, (
        "SETGID cap should be dropped when running as host user — no privilege drop is needed"
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


# ── Docker labels (issue #20561) ──────────────────────────────────


def _run_args_from_calls(calls):
    """Pull the argv list passed to the first ``docker run`` invocation."""
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    return run_calls[0][0]


def _labels_in_run_args(run_args):
    """Return the set of ``key=value`` strings passed via ``--label``."""
    return {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--label"
    }


def test_run_command_tags_hermes_agent_label(monkeypatch):
    """Every container hermes-agent starts must carry the hermes-agent=1 label
    so the orphan reaper (and external operators) can identify them with a
    single ``docker ps --filter label=hermes-agent=1`` call. Regression test
    for issue #20561 — without the label there is no global sweep target."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(task_id="my-task")

    labels = _labels_in_run_args(_run_args_from_calls(calls))
    assert "hermes-agent=1" in labels, (
        f"hermes-agent=1 label missing; got labels: {sorted(labels)}"
    )


def test_run_command_tags_task_and_profile_labels(monkeypatch):
    """task_id and the active profile name are surfaced as labels so future
    cross-process reuse logic can filter to a specific (task, profile) pair
    without parsing container names. Profile resolution uses the helper that
    returns ``"default"`` for the root Hermes home."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "research-bot")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(task_id="kanban-42")

    labels = _labels_in_run_args(_run_args_from_calls(calls))
    assert "hermes-task-id=kanban-42" in labels, (
        f"hermes-task-id=kanban-42 missing; got: {sorted(labels)}"
    )
    assert "hermes-profile=research-bot" in labels, (
        f"hermes-profile=research-bot missing; got: {sorted(labels)}"
    )


def test_label_sanitizer_rejects_invalid_characters():
    """Docker label values must be alnum + ``_.-`` and ≤63 chars. Profile or
    task names containing slashes, colons, or unicode would otherwise emit
    invalid labels that round-trip badly through ``docker ps --filter``."""
    assert docker_env._sanitize_label_value("plain-name_1.0") == "plain-name_1.0"
    assert docker_env._sanitize_label_value("with/slash") == "with_slash"
    assert docker_env._sanitize_label_value("with:colon") == "with_colon"
    assert docker_env._sanitize_label_value("emoji-😀-here") == "emoji-_-here"
    # Empty / non-string inputs must collapse to a queryable token, not "".
    assert docker_env._sanitize_label_value("") == "unknown"
    assert docker_env._sanitize_label_value(None) == "unknown"  # type: ignore[arg-type]
    # >63 chars must truncate, not error.
    long_value = "x" * 100
    assert len(docker_env._sanitize_label_value(long_value)) == 63


def test_run_command_sanitizes_unsafe_task_id(monkeypatch):
    """A task_id containing characters Docker rejects in label values must be
    sanitized before reaching ``docker run --label``; otherwise the daemon
    refuses the run with an inscrutable error and the agent's first command
    blows up."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(task_id="task/with:weird*chars")

    labels = _labels_in_run_args(_run_args_from_calls(calls))
    # Each non-OK character becomes an underscore; the safe chars survive.
    assert "hermes-task-id=task_with_weird_chars" in labels, (
        f"sanitized task-id label missing; got: {sorted(labels)}"
    )


def test_labels_attribute_populated_after_init(monkeypatch):
    """``self._labels`` must be set to the same key/value pairs that went onto
    docker run, so subsequent reuse / reaper paths can match without re-running
    the sanitizer or re-importing the profile module."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)

    env = _make_dummy_env(task_id="abc")

    assert env._labels == {
        "hermes-agent": "1",
        "hermes-task-id": "abc",
        "hermes-profile": "default",
        "hermes-egress": "off",
    }


# ── Cross-process container reuse (issue #20561) ──────────────────


def _mock_subprocess_run_with_reuse(monkeypatch, ps_state: str | None,
                                     start_succeeds: bool = True):
    """Reuse-aware subprocess.run mock.

    ``ps_state`` controls what ``docker ps -a --filter ...`` returns:
      * ``None`` → no match (empty stdout). Forces a fresh ``docker run``.
      * ``"running"`` / ``"exited"`` / ... → emit ``CID\\tSTATE`` so the reuse
        path picks it up. ``"running"`` skips ``docker start``; other states
        trigger ``docker start`` (which can be forced to fail via
        ``start_succeeds=False``).

    Returns the captured call list so the test can verify which docker
    commands actually ran.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                if ps_state is None:
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                # 3-field format: ID, State, EgressLabel.  When egress_label
                # is "off" the code parses all three fields; <no value> means
                # the container has no egress label, which is acceptable.
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout=f"reused-cid\t{ps_state}\t<no value>\n",
                    stderr="",
                )
            if sub == "start":
                if not start_succeeds:
                    # Real subprocess.run with check=True raises on non-zero exit;
                    # mirror that so the production code's except clause fires.
                    raise subprocess.CalledProcessError(1, cmd, output="", stderr="no such container")
                return subprocess.CompletedProcess(cmd, 0, stdout="reused-cid\n", stderr="")
            if sub == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_reuse_attaches_to_running_container_without_docker_run(monkeypatch):
    """When a labeled container is already ``running``, the reuse probe
    must pick it up and skip ``docker run`` entirely. Regression for the
    issue #20561 root cause: every Hermes process spawning a new container
    despite docs claiming "ONE long-lived container shared across sessions"."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    calls = _mock_subprocess_run_with_reuse(monkeypatch, ps_state="running")

    env = _make_dummy_env(task_id="reuse-test")

    # The reuse path must populate _container_id from the ps probe output.
    assert env._container_id == "reused-cid", (
        f"expected reused container id, got {env._container_id!r}"
    )
    # And it must NOT have run `docker run`.
    run_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert not run_invocations, (
        f"docker run should be skipped on reuse, got: {run_invocations}"
    )
    # And it must have NOT issued a `docker start` for an already-running container.
    start_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "start"]
    assert not start_invocations, (
        f"docker start should be skipped when container already running, got: {start_invocations}"
    )


def test_egress_enabled_does_not_reuse_pre_egress_container(monkeypatch):
    """A container created before egress was enabled lacks the proxy env vars
    and CA mount.  Reusing it would silently bypass the credential firewall."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: (
            ["-v", "/tmp/ca:/etc/ssl/certs/hermes-egress-ca.crt:ro"],
            {"HTTPS_PROXY": "http://host.docker.internal:9090"},
            ["--add-host", "host.docker.internal:host-gateway"],
        ),
    )
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                # Simulate an old pre-egress container: without the egress label
                # filter it would match; with the filter Docker returns no match.
                assert any(str(part).startswith("label=hermes-egress=") for part in cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_dummy_env(task_id="reuse-egress")

    assert env._container_id == "fresh-cid"
    run_invocations = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_invocations, "egress-enabled containers require a fresh docker run"


def test_forward_env_provider_key_collision_refuses_under_egress(monkeypatch):
    """docker_forward_env is explicit, but it still must not smuggle real
    provider keys into an enforced egress sandbox."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-real")
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: (
            [],
            {
                "HTTPS_PROXY": "http://host.docker.internal:9090",
                "OPENROUTER_API_KEY": "hermes-proxy-openrouter-token",
                "HERMES_PROXY_TOKEN_OPENROUTER_API_KEY": "hermes-proxy-openrouter-token",
            },
            [],
        ),
    )
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(RuntimeError, match="docker_forward_env.*OPENROUTER_API_KEY"):
        _make_dummy_env(forward_env=["OPENROUTER_API_KEY"])


def test_extra_args_proxy_override_refuses_under_egress(monkeypatch):
    """docker_extra_args are appended after Hermes args, so egress enforcement
    must reject critical overrides before Docker sees them."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: (
            [],
            {"HTTPS_PROXY": "http://host.docker.internal:9090"},
            [],
        ),
    )
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(RuntimeError, match="docker_extra_args.*HTTPS_PROXY"):
        _make_dummy_env(extra_args=["-e", "HTTPS_PROXY="])


def test_reuse_starts_stopped_container_before_attaching(monkeypatch):
    """A labeled container in ``exited`` state must be restarted via
    ``docker start`` before the new Hermes process uses it. Without this
    step, ``docker exec`` against a stopped container errors out and the
    first agent command fails opaquely."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    calls = _mock_subprocess_run_with_reuse(monkeypatch, ps_state="exited")

    env = _make_dummy_env(task_id="reuse-stopped")

    assert env._container_id == "reused-cid"
    start_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "start"]
    assert start_invocations, "expected docker start for exited container"
    run_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert not run_invocations, "should not docker run when reusing an exited container"


def test_reuse_falls_back_to_fresh_run_when_start_fails(monkeypatch):
    """If ``docker start`` on the matched container fails (container was
    removed between probe and start, daemon paused, etc.), the code must
    silently fall through to a fresh ``docker run`` rather than leaving the
    user with a broken environment. Defensive recovery — the probe is best-
    effort, not authoritative."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    calls = _mock_subprocess_run_with_reuse(
        monkeypatch, ps_state="exited", start_succeeds=False,
    )

    env = _make_dummy_env(task_id="reuse-broken-start")

    # docker start should be attempted then fail; code falls through to run.
    assert env._container_id == "fresh-cid", (
        f"expected fresh container id after fallback, got {env._container_id!r}"
    )
    run_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_invocations, "fallback to fresh docker run must happen on start failure"


def test_failed_docker_run_cleans_up_orphaned_container(monkeypatch):
    """When ``docker run`` fails (e.g. exit 125), the partially-created
    container must be removed by name.

    Docker can create the container object before failing to start it,
    leaving a stale ``Created`` container. The exited-only orphan reaper
    (``reap_orphan_containers``, ``status=exited``) never catches a
    ``Created`` orphan, so without this cleanup it leaks permanently.
    Regression for #7439. Salvage of #7440 (@Tranquil-Flow).
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    cleanup_calls = []

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                # No reusable container -> fall through to a fresh `docker run`.
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                raise subprocess.CalledProcessError(
                    125, cmd, output="", stderr="docker: Error response from daemon"
                )
            if sub == "rm":
                cleanup_calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(subprocess.CalledProcessError):
        _make_dummy_env()

    assert len(cleanup_calls) == 1, "docker rm should be called once for the orphaned container"
    rm_cmd = cleanup_calls[0]
    assert rm_cmd[1] == "rm" and rm_cmd[2] == "-f"
    assert rm_cmd[3].startswith("hermes-"), "should remove the container by its generated name"


def test_docker_run_timeout_cleans_up_orphaned_container(monkeypatch):
    """When ``docker run`` times out (e.g. slow image pull), the
    partially-created container must be removed. Salvage of #7440
    (@Tranquil-Flow); regression for #7439.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    cleanup_calls = []

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                raise subprocess.TimeoutExpired(cmd, 120)
            if sub == "rm":
                cleanup_calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(subprocess.TimeoutExpired):
        _make_dummy_env()

    assert len(cleanup_calls) == 1, "docker rm should be called once for the orphaned container"
    rm_cmd = cleanup_calls[0]
    assert rm_cmd[1] == "rm" and rm_cmd[2] == "-f"
    assert rm_cmd[3].startswith("hermes-"), "should remove the container by its generated name"


def test_no_reuse_when_persist_across_processes_disabled(monkeypatch):
    """Opt-out path: ``persist_across_processes=False`` skips the ps probe
    entirely and always starts a fresh container, matching the pre-fix
    behavior for users who want hard per-process isolation."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    # ps_state=running would trigger reuse if the probe ran — assert it doesn't.
    calls = _mock_subprocess_run_with_reuse(monkeypatch, ps_state="running")

    env = docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=60,
        task_id="no-reuse", persist_across_processes=False,
    )

    # Must NOT have issued docker ps (the probe is gated by the flag).
    ps_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "ps"]
    assert not ps_invocations, (
        f"docker ps probe should be skipped when persist_across_processes=False, got: {ps_invocations}"
    )
    # Should have started a fresh container.
    assert env._container_id == "fresh-cid"


def test_find_reusable_container_prefers_running_over_stopped(monkeypatch):
    """When the probe returns multiple matches (shouldn't normally happen,
    but can after a crash leaves stale duplicates), a ``running`` container
    is preferred over any stopped one. The duplicate gets reaped later by
    the orphan reaper; we don't try to be heroic about it here."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            if cmd[1] == "ps":
                # Two matches: stopped first, running second.  3-field format
                # with absent egress label for the "off" path.
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="stopped-cid\texited\t<no value>\nrunning-cid\trunning\t<no value>\n",
                    stderr="",
                )
        return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_dummy_env(task_id="dup-match")
    assert env._container_id == "running-cid", (
        f"running container should win over stopped duplicate, got {env._container_id!r}"
    )


def test_find_reusable_handles_empty_label_string(monkeypatch):
    """Docker CLI v29.5.3 returns an empty string (NOT ``<no value>``)
    for absent labels.  The trailing tab produces ``cid\\trunning\\t\\n``;
    we must not strip the trailing tab or the three-field parser drops the
    container.  Regression test for the egilewski review on #48073."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            if cmd[1] == "ps":
                # Docker v29.5.3: absent label → empty string, trailing tab
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="safe-cid\trunning\t\n",
                    stderr="",
                )
        return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_dummy_env(task_id="empty-label")
    assert env._container_id == "safe-cid", (
        f"container with empty-string label should be reused, got {env._container_id!r}"
    )


def test_reuse_off_rejects_non_off_egress_container(monkeypatch):
    """When egress is off, a container that still has hermes-egress=on
    (e.g. from before ``hermes egress disable``) must be rejected and a
    fresh container created.  The post-filter protects against silently
    reusing a container with baked-in proxy env and CA mounts."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            if cmd[1] == "ps":
                # Return a container with hermes-egress=on.  With egress=off
                # the three-field format includes the label; the post-filter
                # must skip this entry.
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="stale-cid\trunning\ton\n",
                    stderr="",
                )
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_dummy_env(task_id="egress-off-reject")
    # Should fall through to fresh container because the stale one has
    # hermes-egress=on.
    assert env._container_id == "fresh-cid", (
        f"expected fresh container, got {env._container_id!r}"
    )


# ── Cleanup correctness (issue #20561) ────────────────────────────


class _FakeThread:
    """Stand-in for threading.Thread that captures target/args and calls
    target() synchronously when .start() runs, so cleanup behavior is
    observable without actually backgrounding subprocess calls."""

    def __init__(self, target=None, daemon=None, name=None):
        self._target = target
        self.daemon = daemon
        self.name = name
        self._done = False

    def start(self):
        if self._target is not None:
            self._target()
        self._done = True

    def is_alive(self):
        return not self._done

    def join(self, timeout=None):
        self._done = True


def _install_fake_thread(monkeypatch):
    import threading
    monkeypatch.setattr(threading, "Thread", _FakeThread)


def test_cleanup_with_persist_is_noop_for_container(monkeypatch):
    """``persist_across_processes=True`` (default) cleanup must NEITHER stop
    NOR remove the container — the docs promise "ONE long-lived container
    shared across sessions", and any docker stop would kill background
    processes inside the container (npm watchers, pytest watchers, etc.).

    Resource reclamation in this mode happens via the orphan reaper on next
    Hermes startup, not on graceful exit. Issue #20561 — the first iteration
    of this PR did docker stop here, which Ben caught as contradicting the
    "ONE long-lived container" semantics."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    env = _make_dummy_env(task_id="cleanup-persist", persistent_filesystem=False)
    # Default persist_across_processes=True.
    container_id = env._container_id
    assert container_id

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert not stops, (
        f"docker stop must NOT be called when persist_across_processes=True; "
        f"container has to stay running so background processes survive. "
        f"Got: {stops}"
    )
    assert not rms, (
        f"docker rm must NOT be called when persist_across_processes=True; "
        f"reuse would be impossible. Got: {rms}"
    )
    # The in-process handle must still be cleared so the next __init__
    # re-probes via labels (and reuses the still-running container).
    assert env._container_id is None, (
        "in-process container_id should be cleared even in no-op cleanup"
    )


def test_cleanup_force_remove_stops_and_rms_even_in_persist_mode(monkeypatch):
    """``cleanup(force_remove=True)`` must stop AND rm the container even
    when ``persist_across_processes=True``. This is the explicit-teardown
    path for ``/reset``, ``cleanup_vm(task_id, force_remove=True)``, and any
    future caller that wants a guaranteed fresh container.

    Without this kwarg, callers in persist mode would have no way to force a
    fresh container without also flipping the global config — too coarse for
    a per-task reset.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    env = _make_dummy_env(task_id="cleanup-force", persistent_filesystem=False)
    assert env._container_id

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    env.cleanup(force_remove=True)

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert stops, f"force_remove must docker stop; got: {cleanup_calls}"
    assert rms, f"force_remove must docker rm; got: {cleanup_calls}"


def test_cleanup_vm_default_honors_persist_mode(monkeypatch):
    """``cleanup_vm(task_id)`` without ``force_remove=True`` must be a no-op
    for a persist-mode container.

    Regression for the bug Ben caught after commit 4: ``AIAgent.close()``
    (which is called from ``tui_gateway/server.py`` on session.close, from
    ``gateway/run.py`` on per-session teardown, and from per-turn cleanup)
    calls ``cleanup_vm(task_id)``. If that defaulted to ``force_remove=True``
    we'd tear down the container on every TUI session close, defeating the
    "ONE long-lived container shared across sessions" contract.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    from tools import terminal_tool

    env = _make_dummy_env(task_id="session-close-test")
    container_id = env._container_id
    terminal_tool._active_environments["session-close-test"] = env

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    try:
        terminal_tool.cleanup_vm("session-close-test")
    finally:
        terminal_tool._active_environments.pop("session-close-test", None)

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert not stops, (
        f"cleanup_vm() default must not docker stop a persist-mode container; "
        f"got: {stops}"
    )
    assert not rms, (
        f"cleanup_vm() default must not docker rm a persist-mode container; "
        f"got: {rms}"
    )


def test_cleanup_vm_force_remove_tears_down_persist_container(monkeypatch):
    """``cleanup_vm(task_id, force_remove=True)`` tears down a persist-mode
    container — the explicit-teardown path for ``/reset``-style flows.

    Also pins the runtime-signature-inspection plumbing: the kwarg must
    actually flow through ``cleanup_vm`` into the backend's ``cleanup()``.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    from tools import terminal_tool

    env = _make_dummy_env(task_id="explicit-teardown-test")
    terminal_tool._active_environments["explicit-teardown-test"] = env

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    try:
        terminal_tool.cleanup_vm("explicit-teardown-test", force_remove=True)
    finally:
        terminal_tool._active_environments.pop("explicit-teardown-test", None)

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert stops, f"force_remove must reach docker stop; got: {cleanup_calls}"
    assert rms, f"force_remove must reach docker rm; got: {cleanup_calls}"


def test_cleanup_with_persist_disabled_stops_and_rms(monkeypatch):
    """``persist_across_processes=False`` cleanup must docker stop AND docker
    rm so containers don't leak. Crucially, this runs regardless of the
    ``persistent_filesystem`` setting — the original code only rm'd when
    ``not self._persistent``, which meant the default-on ``container_persistent:
    true`` users (the documented happy path) leaked Exited containers forever.
    Issue #20561 root-cause fix."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    # Note: persistent_filesystem=True (the prior-leak scenario) + the new
    # cross-process toggle OFF must still result in a clean rm.
    env = docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=60,
        task_id="cleanup-no-persist", persistent_filesystem=True,
        persist_across_processes=False,
    )

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert stops, "expected docker stop"
    assert rms, (
        "docker rm MUST run when persist_across_processes=False, even with "
        "persistent_filesystem=True — that gating was the leak source in #20561."
    )


def test_cleanup_uses_subprocess_run_not_detached_shell(monkeypatch):
    """The pre-fix code used ``subprocess.Popen("... &", shell=True)`` which
    raced with parent-process exit and silently dropped cleanup work. The
    new code must use ``subprocess.run`` with bounded ``timeout=`` so the
    work actually completes within the process lifetime.

    Asserts cleanup never reaches into shell-mode Popen. Uses
    ``force_remove=True`` so cleanup actually issues docker calls — the
    default persist-mode path is now a no-op (commit 4) and would trivially
    pass this assertion without exercising the docker code at all.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    def _forbidden_popen(*args, **kwargs):
        raise AssertionError(
            f"cleanup must not use subprocess.Popen anymore (issue #20561); "
            f"got args={args} kwargs={kwargs}"
        )

    monkeypatch.setattr(docker_env.subprocess, "Popen", _forbidden_popen)

    env = _make_dummy_env(task_id="no-popen-cleanup")
    env.cleanup(force_remove=True)  # must not raise


def test_wait_for_cleanup_returns_true_when_no_thread_started():
    """``wait_for_cleanup`` must be a no-op when ``cleanup`` was never called
    (or the env has no live cleanup thread) — atexit calls it unconditionally
    across all active envs, so a False return would falsely flag healthy
    shutdowns."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    # No _cleanup_thread set — simulates an env that was never cleanup()'d.
    assert env.wait_for_cleanup(timeout=1.0) is True


def test_wait_for_cleanup_after_cleanup_returns_true(monkeypatch):
    """End-to-end: cleanup() starts a thread, wait_for_cleanup() joins it
    and reports completion. Atexit relies on this contract to ensure docker
    stop/rm actually finishes before the Python interpreter exits.

    Uses ``force_remove=True`` so cleanup actually starts a worker thread —
    the default persist-mode cleanup is a no-op (commit 4) and never spawns
    a thread, so the trivial "no thread" branch of wait_for_cleanup is
    already covered by the previous test.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    env = _make_dummy_env(task_id="wait-test")
    env.cleanup(force_remove=True)
    assert env.wait_for_cleanup(timeout=5.0) is True


def test_cleanup_on_env_with_no_container_id_does_not_raise(monkeypatch):
    """A DockerEnvironment whose ``__init__`` failed before the container_id
    was set (image-pull error, docker daemon down) should still be safe to
    cleanup() — the post-creation failure path in callers always tries.
    Without this guard the daemon-down case used to NameError on the cleanup
    branch."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = None
    env._persistent = False
    env._workspace_dir = None
    env._home_dir = None
    # No exception expected.
    env.cleanup()


# ── Orphan reaper (issue #20561) ──────────────────────────────────


def _now_iso(offset_seconds: int = 0) -> str:
    """Return an RFC3339 timestamp ``offset_seconds`` in the past."""
    import datetime
    t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=offset_seconds)
    # Format like Docker emits — with nanoseconds-style trailing digits.
    return t.isoformat().replace("+00:00", ".123456789Z")


def _reaper_run_mock(monkeypatch, ps_ids: list[str], inspect_responses: dict[str, str],
                      rm_succeeds: bool = True):
    """Build a subprocess.run mock for reaper tests.

    * ``ps_ids`` — what ``docker ps -a --filter ... --format '{{.ID}}'`` returns
    * ``inspect_responses[cid]`` — what ``docker inspect ... FinishedAt`` returns
      for each cid; ``""`` means "field unset".
    * ``rm_succeeds`` — whether ``docker rm -f`` returns 0.

    Captures every call so tests can assert which containers were rm'd.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if not isinstance(cmd, list) or len(cmd) < 2:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        sub = cmd[1]
        if sub == "ps":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="\n".join(ps_ids) + ("\n" if ps_ids else ""), stderr="",
            )
        if sub == "inspect":
            # cmd is [docker, inspect, --format, '{{.State.FinishedAt}}', cid]
            cid = cmd[-1]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=inspect_responses.get(cid, "") + "\n", stderr="",
            )
        if sub == "rm":
            return subprocess.CompletedProcess(
                cmd, 0 if rm_succeeds else 1,
                stdout="", stderr="" if rm_succeeds else "no such container",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_reap_orphan_returns_zero_when_no_matches(monkeypatch):
    """No labeled containers → no rm calls, returns 0. Establishes the
    happy-path baseline for the orphan reaper (issue #20561)."""
    calls = _reaper_run_mock(monkeypatch, ps_ids=[], inspect_responses={})

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    assert removed == 0
    rms = [c for c in calls if isinstance(c[0], list) and c[0][1:2] == ["rm"]]
    assert not rms, "no rm calls expected when ps returns empty"


def test_reap_orphan_removes_stale_exited_container(monkeypatch):
    """An Exited container older than max_age_seconds must be removed.
    This is the core repair path for issue #20561 — without the reaper,
    SIGKILL'd Hermes processes leak containers permanently."""
    old = _now_iso(offset_seconds=900)  # 15 minutes ago
    calls = _reaper_run_mock(
        monkeypatch, ps_ids=["old-cid"], inspect_responses={"old-cid": old},
    )

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    assert removed == 1
    rms = [c for c in calls if isinstance(c[0], list) and c[0][1:2] == ["rm"]]
    assert len(rms) == 1
    assert "old-cid" in rms[0][0], f"expected rm of old-cid, got {rms[0][0]}"


def test_reap_orphan_spares_recently_exited_container(monkeypatch):
    """A container exited within max_age_seconds must NOT be reaped — that
    container belongs to a Hermes process that just finished and may be
    about to be replaced. Conservative window prevents racing sibling
    processes."""
    recent = _now_iso(offset_seconds=60)  # 1 minute ago
    calls = _reaper_run_mock(
        monkeypatch, ps_ids=["recent-cid"], inspect_responses={"recent-cid": recent},
    )

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    assert removed == 0
    rms = [c for c in calls if isinstance(c[0], list) and c[0][1:2] == ["rm"]]
    assert not rms, f"recent container must not be reaped, got rm calls: {rms}"


def test_reap_orphan_scopes_to_profile_filter_via_label(monkeypatch):
    """The reaper must pass ``--filter label=hermes-profile=<profile>`` to
    docker ps so it never sweeps another profile's containers. A research
    profile must not tear down the default profile's stragglers."""
    calls = _reaper_run_mock(monkeypatch, ps_ids=[], inspect_responses={})

    docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="research-bot", docker_exe="/usr/bin/docker",
    )

    ps_calls = [c for c in calls if isinstance(c[0], list) and c[0][1:2] == ["ps"]]
    assert ps_calls, "expected at least one docker ps call"
    flat = " ".join(ps_calls[0][0])
    assert "label=hermes-profile=research-bot" in flat, (
        f"profile filter not applied to docker ps; got args: {ps_calls[0][0]}"
    )
    assert "label=hermes-agent=1" in flat, (
        f"hermes-agent label filter must also be applied; got: {ps_calls[0][0]}"
    )
    assert "status=exited" in flat, (
        "must filter to exited containers only — running containers may "
        "belong to a sibling Hermes process and must NEVER be reaped"
    )


def test_reap_orphan_skips_container_with_unparseable_finished_at(monkeypatch):
    """If docker inspect returns the zero-value ``0001-01-01T00:00:00Z`` (no
    FinishedAt yet) or an unparseable timestamp, the reaper must leave the
    container alone. Defensive — never reap a container whose age we can't
    determine."""
    calls = _reaper_run_mock(
        monkeypatch,
        ps_ids=["never-finished", "garbage-ts"],
        inspect_responses={
            "never-finished": "0001-01-01T00:00:00Z",
            "garbage-ts": "not-a-timestamp",
        },
    )

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    assert removed == 0
    rms = [c for c in calls if isinstance(c[0], list) and c[0][1:2] == ["rm"]]
    assert not rms, (
        f"reaper must NOT remove containers with unparseable FinishedAt; got: {rms}"
    )


def test_reap_orphan_handles_docker_ps_failure_gracefully(monkeypatch):
    """If docker ps itself fails (daemon down, permission denied), the
    reaper returns 0 without crashing. The reaper is best-effort plumbing,
    not a critical path — it must never block container creation."""
    def _failing_ps(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "ps":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Cannot connect to daemon")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _failing_ps)

    # Must not raise
    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )
    assert removed == 0


def test_reap_orphan_continues_after_individual_rm_failure(monkeypatch):
    """If ``docker rm -f`` fails on one container (already removed by a
    concurrent process, container locked, etc.), the reaper must log and
    continue to the next candidate rather than aborting the whole sweep."""
    old = _now_iso(offset_seconds=900)
    rm_calls = []

    def _run(cmd, **kwargs):
        if not isinstance(cmd, list) or len(cmd) < 2:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        sub = cmd[1]
        if sub == "ps":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="cid-a\ncid-b\ncid-c\n", stderr="",
            )
        if sub == "inspect":
            return subprocess.CompletedProcess(cmd, 0, stdout=old + "\n", stderr="")
        if sub == "rm":
            rm_calls.append(cmd[-1])
            # cid-b fails; cid-a and cid-c succeed.
            if cmd[-1] == "cid-b":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such container")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    # All three were attempted, two succeeded.
    assert removed == 2
    assert set(rm_calls) == {"cid-a", "cid-b", "cid-c"}, (
        f"reaper must attempt all candidates even when one fails; got: {rm_calls}"
    )


def test_container_finished_at_parses_nanosecond_timestamp(monkeypatch):
    """Docker emits FinishedAt with nanosecond precision (RFC3339 with up to
    9 fractional digits), but Python's fromisoformat caps at microseconds.
    The helper must trim the extra digits without raising — otherwise every
    candidate gets skipped and the reaper does nothing."""

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout="2026-05-28T13:45:00.123456789Z\n",
            stderr="",
        )

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    result = docker_env._container_finished_at("/usr/bin/docker", "test-cid")
    assert result is not None, "must parse RFC3339 with nanoseconds"
    import datetime
    assert result.tzinfo == datetime.timezone.utc
    assert result.year == 2026 and result.month == 5 and result.day == 28


def test_container_finished_at_returns_none_on_zero_value():
    """Docker's zero-value ``0001-01-01T00:00:00Z`` (never finished) must
    map to None so the reaper treats the container as unreapable."""
    # Direct test of the parsing helper — no subprocess needed since the
    # check happens after the inspect call returns.
    import subprocess as _subprocess

    class _MockRun:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    import unittest.mock
    with unittest.mock.patch.object(
        docker_env.subprocess, "run", return_value=_MockRun("0001-01-01T00:00:00Z\n"),
    ):
        result = docker_env._container_finished_at("/usr/bin/docker", "never-finished")
    assert result is None


def test_credential_mount_skipped_when_source_is_directory(monkeypatch, tmp_path, caplog):
    """Credential mount should be skipped when source path is a directory.

    In Docker-in-Docker scenarios, Docker may auto-create the source path as
    a directory when it doesn't exist on the host.  Mounting a directory over
    a file destination causes exit 125.
    """
    # Create a directory that looks like a corrupted credential file path
    corrupted_dir = tmp_path / "google_token.json"
    corrupted_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    # Mock get_credential_file_mounts to return the corrupted entry
    fake_mounts = [
        {"host_path": str(corrupted_dir), "container_path": "/root/.hermes/google_token.json"},
    ]
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: fake_mounts,
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda: [],
    )
    monkeypatch.setattr(
        "tools.credential_files.get_cache_directory_mounts",
        lambda: [],
    )

    with caplog.at_level(logging.WARNING):
        _make_dummy_env()

    # The corrupted mount should be skipped
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert "google_token.json" not in run_args_str

    # Should log a warning about the directory source
    assert any(
        "source is a directory" in rec.getMessage()
        for rec in caplog.records
    )


def test_credential_mount_skipped_when_source_missing(monkeypatch, tmp_path, caplog):
    """Credential mount should be skipped when source file no longer exists."""
    missing_path = tmp_path / "deleted_token.json"
    # Don't create the file — it's "missing"

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    fake_mounts = [
        {"host_path": str(missing_path), "container_path": "/root/.hermes/deleted_token.json"},
    ]
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: fake_mounts,
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda: [],
    )
    monkeypatch.setattr(
        "tools.credential_files.get_cache_directory_mounts",
        lambda: [],
    )

    with caplog.at_level(logging.WARNING):
        _make_dummy_env()

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert "deleted_token.json" not in run_args_str

    assert any(
        "source not found" in rec.getMessage()
        for rec in caplog.records
    )


def test_credential_mount_works_when_source_is_valid_file(monkeypatch, tmp_path):
    """Credential mount should proceed normally when source is a valid file."""
    valid_file = tmp_path / "token.json"
    valid_file.write_text('{"token": "REDACTED"}')

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    fake_mounts = [
        {"host_path": str(valid_file), "container_path": "/root/.hermes/token.json"},
    ]
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: fake_mounts,
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda: [],
    )
    monkeypatch.setattr(
        "tools.credential_files.get_cache_directory_mounts",
        lambda: [],
    )

    _make_dummy_env()

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert "token.json" in run_args_str


# ── s6-overlay /init image handling (issue #34628) ────────────────


def _mock_subprocess_run_with_entrypoint(monkeypatch, entrypoint_json):
    """Like _mock_subprocess_run, but `docker image inspect` returns the given
    entrypoint JSON so _image_uses_init_entrypoint can be exercised end-to-end.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "image" and len(cmd) >= 3 and cmd[2] == "inspect":
                return subprocess.CompletedProcess(cmd, 0, stdout=entrypoint_json + "\n", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_image_uses_init_entrypoint_detects_s6_init(monkeypatch):
    """An image whose entrypoint is /init is detected as an s6-overlay image."""
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout='["/init"]', stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._image_uses_init_entrypoint("/usr/bin/docker", "hermes-agent:latest") is True


def test_image_uses_init_entrypoint_false_for_plain_image(monkeypatch):
    """A normal image (no /init entrypoint) is not treated as s6-overlay."""
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout='["/bin/sh","-c"]', stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._image_uses_init_entrypoint("/usr/bin/docker", "python:3.11") is False


def test_image_uses_init_entrypoint_false_for_null_entrypoint(monkeypatch):
    """Images with no declared entrypoint (null) keep hardened defaults."""
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="null", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._image_uses_init_entrypoint("/usr/bin/docker", "alpine") is False


def test_image_uses_init_entrypoint_false_on_inspect_failure(monkeypatch):
    """An inspect failure (e.g. image not pulled) is best-effort -> defaults kept."""
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No such image")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._image_uses_init_entrypoint("/usr/bin/docker", "missing:tag") is False


def test_image_uses_init_entrypoint_false_on_exception(monkeypatch):
    """A subprocess error never raises out of detection — defaults kept."""
    def _run(cmd, **kwargs):
        raise OSError("docker daemon down")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._image_uses_init_entrypoint("/usr/bin/docker", "x") is False


def test_s6_image_skips_docker_init_and_mounts_run_exec(monkeypatch):
    """For an s6-overlay /init image, docker run must omit --init and mount
    /run with exec (issue #34628)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run_with_entrypoint(monkeypatch, '["/init"]')

    _make_dummy_env(image="hermes-agent:latest")

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    assert "--init" not in run_args, "s6 /init image must not get Docker --init"

    tmpfs_vals = [run_args[i + 1] for i, a in enumerate(run_args[:-1]) if a == "--tmpfs"]
    run_mounts = [v for v in tmpfs_vals if v.startswith("/run:")]
    assert run_mounts, f"no /run tmpfs mount found in {tmpfs_vals}"
    assert "exec" in run_mounts[0] and "noexec" not in run_mounts[0], (
        f"/run must be mounted exec for s6 images, got: {run_mounts[0]}"
    )


def test_plain_image_keeps_docker_init_and_run_noexec(monkeypatch):
    """A non-s6 image keeps the hardened defaults: Docker --init and noexec /run."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run_with_entrypoint(monkeypatch, '["/bin/sh","-c"]')

    _make_dummy_env(image="python:3.11")

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    assert "--init" in run_args, "non-s6 image must keep Docker --init"

    tmpfs_vals = [run_args[i + 1] for i, a in enumerate(run_args[:-1]) if a == "--tmpfs"]
    run_mounts = [v for v in tmpfs_vals if v.startswith("/run:")]
    assert run_mounts, f"no /run tmpfs mount found in {tmpfs_vals}"
    assert "noexec" in run_mounts[0], (
        f"/run must stay noexec for non-s6 images, got: {run_mounts[0]}"
    )


# ---------------------------------------------------------------------------
# Out-of-band container removal recovery (issue #36266, PR #36631)
# ---------------------------------------------------------------------------


def test_is_container_gone_matches_removal_errors(monkeypatch):
    """``_is_container_gone`` recognizes the docker errors that mean the
    container no longer exists, and does NOT match ordinary command failures.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env()

    # Positive: the daemon's "container gone" phrasings.
    assert env._is_container_gone(
        "Error response from daemon: No such container: hermes-abc123"
    )
    assert env._is_container_gone("Error: No such container: deadbeef")
    assert env._is_container_gone(
        "Error response from daemon: Container abc is not running"
    )

    # Control / negative: a real command failure must NOT be misclassified as
    # the container being gone — otherwise every non-zero exit would trigger a
    # spurious container recreation.
    assert not env._is_container_gone("bash: nonsuch: command not found")
    assert not env._is_container_gone("Traceback (most recent call last): ...")
    assert not env._is_container_gone("")
    assert not env._is_container_gone("permission denied")


def test_execute_recovers_from_out_of_band_removal(monkeypatch):
    """When a persistent container is removed out-of-band, ``execute`` detects
    the "No such container" error, recreates the container, and retries once —
    returning success transparently.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        persistent_filesystem=True,
        persist_across_processes=True,
    )

    # First execute() sees a dead container; second (post-recovery) succeeds.
    outputs = iter([
        {"output": "Error response from daemon: No such container: hermes-x", "returncode": 1},
        {"output": "ok", "returncode": 0},
    ])

    def _fake_super_execute(self, command, cwd="", **kwargs):
        return next(outputs)

    recreate_calls = []

    def _fake_recreate(self):
        recreate_calls.append(True)
        self._container_id = "recovered-container-id"
        return True

    monkeypatch.setattr(docker_env.BaseEnvironment, "execute", _fake_super_execute)
    monkeypatch.setattr(
        docker_env.DockerEnvironment, "_recreate_container", _fake_recreate
    )

    result = env.execute("echo hi")

    assert recreate_calls == [True], "recovery should have been attempted exactly once"
    assert result.get("returncode") == 0, f"expected success after recovery, got {result!r}"
    assert result.get("output") == "ok"


def test_execute_does_not_recover_when_not_persistent(monkeypatch):
    """A non-persistent session must NOT trigger container recreation on a
    "No such container" error — recovery is only meaningful for the persistent,
    cross-process container that can be removed out-of-band.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        persistent_filesystem=True,
        persist_across_processes=False,
    )

    def _fake_super_execute(self, command, cwd="", **kwargs):
        return {"output": "No such container: x", "returncode": 1}

    def _fail_recreate(self):
        pytest.fail("recreation must not run when persist_across_processes is False")

    monkeypatch.setattr(docker_env.BaseEnvironment, "execute", _fake_super_execute)
    monkeypatch.setattr(
        docker_env.DockerEnvironment, "_recreate_container", _fail_recreate
    )

    result = env.execute("echo hi")
    assert result.get("returncode") == 1, "the original error must pass through unchanged"


def test_execute_does_not_recover_on_ordinary_failure(monkeypatch):
    """A genuine non-zero exit that is NOT a container-gone error must pass
    through without triggering recovery (guards against over-eager recreation).
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        persistent_filesystem=True,
        persist_across_processes=True,
    )

    def _fake_super_execute(self, command, cwd="", **kwargs):
        return {"output": "bash: badcmd: command not found", "returncode": 127}

    def _fail_recreate(self):
        pytest.fail("recreation must not run for an ordinary command failure")

    monkeypatch.setattr(docker_env.BaseEnvironment, "execute", _fake_super_execute)
    monkeypatch.setattr(
        docker_env.DockerEnvironment, "_recreate_container", _fail_recreate
    )

    result = env.execute("badcmd")
    assert result.get("returncode") == 127
    assert "command not found" in result.get("output", "")
