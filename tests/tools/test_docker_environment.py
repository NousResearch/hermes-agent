import logging
import os
from io import StringIO
import subprocess

import pytest

from tools.environments import docker as docker_env


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.

    Pre-seeds the cgroup-limit probe cache to ``True`` so the throwaway probe
    container (a ``docker run ... sleep 0``) does not run and pollute the
    captured call list — these tests inspect the real sandbox-start ``run``.
    Tests that exercise the probe itself live in test_docker_cgroup_limits.py.
    """
    docker_env._cgroup_limits_ok = True
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
        shared_container_key=kwargs.get("shared_container_key", ""),
        shm_size=kwargs.get("shm_size", docker_env._DEFAULT_SHM_SIZE),
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

    assert "-e" in args and "DATABASE_URL" in args
    # Value must NOT be in argv (world-readable /proc/*/cmdline, #96268) —
    # it travels via the docker client subprocess env instead.
    assert not any("value_from_dotenv" in a for a in args)
    assert env._init_env_values["DATABASE_URL"] == "value_from_dotenv"


def test_init_env_args_prefers_shell_env_over_hermes_dotenv(monkeypatch):
    """Shell env vars take priority over .env file values in init env args."""
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.setenv("DATABASE_URL", "value_from_shell")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()

    assert "DATABASE_URL" in args
    assert env._init_env_values["DATABASE_URL"] == "value_from_shell"
    assert not any("value_from_dotenv" in a for a in args)


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
    # must win and a blank value must never be forwarded.
    assert "MY_SECRET" in args
    assert env._init_env_values["MY_SECRET"] == "value_from_dotenv"


def test_init_env_args_uses_active_profile_for_forwarded_env(monkeypatch):
    """Docker forwarding must resolve the routed profile's secret scope."""
    from agent import secret_scope as ss

    env = _make_execute_only_env(forward_env=["SERVICE_TOKEN"])
    monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
    try:
        args = env._build_init_env_args()
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert "SERVICE_TOKEN" in args
    assert env._init_env_values["SERVICE_TOKEN"] == "token-for-routed-profile"
    assert not any("token-for-default" in a for a in args)


def test_init_env_args_omits_missing_scoped_forwarded_env(monkeypatch):
    """A missing routed secret must not reintroduce the process env value."""
    from agent import secret_scope as ss

    env = _make_execute_only_env(forward_env=["SERVICE_TOKEN"])
    monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        args = env._build_init_env_args()
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert not any("token-for-default" in a for a in args)
    assert "SERVICE_TOKEN" not in args
    assert "SERVICE_TOKEN" not in env._init_env_values


def test_runtime_exec_tracks_scope_and_clears_missing_value(monkeypatch):
    """Shared Docker containers must refresh and clear profile-scoped values."""
    from agent import secret_scope as ss

    env = _make_execute_only_env(forward_env=["SERVICE_TOKEN"])
    monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})
    calls = []
    monkeypatch.setattr(
        docker_env,
        "_popen_bash",
        lambda cmd, stdin_data=None, **kw: calls.append((cmd, stdin_data, kw)) or object(),
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-profile-a"})
    try:
        env._run_bash("printf '%s' \"$SERVICE_TOKEN\"")
    finally:
        ss.reset_secret_scope(token)

    token = ss.set_secret_scope({})
    try:
        env._run_bash("printf '%s' \"${SERVICE_TOKEN-unset}\"")
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    first_cmd = calls[0][0]
    # Name-only flag in argv; the value rides in the client subprocess env
    # (issue #96268: keep secrets out of world-readable /proc/*/cmdline).
    assert "SERVICE_TOKEN" in first_cmd
    assert not any("token-for-profile-a" in str(a) for a in first_cmd)
    first_env = calls[0][2].get("env") or {}
    assert first_env.get("SERVICE_TOKEN") == "token-for-profile-a"
    second_cmd = calls[1][0]
    second_env = (calls[1][2].get("env") or {})
    assert second_env.get("SERVICE_TOKEN") != "token-for-profile-a"
    assert not any("token-for-profile-a" in str(a) for a in second_cmd)
    assert "unset SERVICE_TOKEN" in second_cmd[-1]


def test_wrapped_exec_scopes_explicit_forward_env_across_profiles(monkeypatch, tmp_path):
    """The shared snapshot must not resurrect an explicit forward-only value."""
    from agent import secret_scope as ss

    env = _make_execute_only_env(forward_env=["EXPLICIT_TOKEN"])
    env.cwd = str(tmp_path)
    env._snapshot_path = str(tmp_path / "snapshot.sh")
    env._cwd_file = str(tmp_path / "cwd.txt")
    env._snapshot_passthrough_names = set()
    (tmp_path / "snapshot.sh").write_text(
        "export EXPLICIT_TOKEN=stale-from-previous-profile\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXPLICIT_TOKEN", "token-for-default")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    def _run_fake_docker_exec(cmd, stdin_data=None, **kwargs):
        """Execute the generated docker exec command in a real local bash."""
        container_index = cmd.index(env._container_id)
        # Name-only -e flags (#96268): values come from the client env kwarg.
        client_env = kwargs.get("env") or os.environ
        child_env = os.environ.copy()
        index = 2
        while index < container_index:
            assert cmd[index] == "-e"
            key = cmd[index + 1]
            assert "=" not in key, f"secret value leaked into argv: {key}"
            if key in client_env:
                child_env[key] = client_env[key]
            index += 2
        assert cmd[container_index + 1 : container_index + 3] == ["bash", "-c"]
        return subprocess.Popen(
            ["bash", "-c", cmd[container_index + 3]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )

    monkeypatch.setattr(docker_env, "_popen_bash", _run_fake_docker_exec)
    ss.set_multiplex_active(True)

    try:
        for scope, expected in (
            ({"EXPLICIT_TOKEN": "token-for-profile-a"}, "token-for-profile-a"),
            ({"EXPLICIT_TOKEN": "token-for-profile-b"}, "token-for-profile-b"),
            ({}, "unset"),
        ):
            scope_token = ss.set_secret_scope(scope)
            try:
                result = env.execute("printf '%s' \"${EXPLICIT_TOKEN-unset}\"")
            finally:
                ss.reset_secret_scope(scope_token)

            assert result["returncode"] == 0
            assert result["output"] == expected
            assert "EXPLICIT_TOKEN=" not in (
                tmp_path / "snapshot.sh"
            ).read_text(encoding="utf-8")
    finally:
        ss.set_multiplex_active(False)


# ── docker_env tests ──────────────────────────────────────────────


def test_docker_env_appears_in_run_command(monkeypatch):
    """Explicit docker_env values pass via name-only -e + client env (#96268)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock", "GNUPGHOME": "/root/.gnupg"})

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args, run_kwargs = run_calls[0]
    run_args_str = " ".join(run_args)
    # Names in argv, values ONLY in the client subprocess env (#96268).
    assert "SSH_AUTH_SOCK" in run_args and "GNUPGHOME" in run_args
    assert "/run/user/1000/ssh-agent.sock" not in run_args_str
    assert "/root/.gnupg" not in run_args_str
    client_env = run_kwargs.get("env") or {}
    assert client_env.get("SSH_AUTH_SOCK") == "/run/user/1000/ssh-agent.sock"
    assert client_env.get("GNUPGHOME") == "/root/.gnupg"


def _node_options_from_run(calls):
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    args, kwargs = run_calls[0]
    if "NODE_OPTIONS" in args and "-e" in args:
        return (kwargs.get("env") or {}).get("NODE_OPTIONS")
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


def test_forward_env_overrides_docker_env_in_init_args(monkeypatch):
    """docker_forward_env should override docker_env for the same key."""
    env = _make_execute_only_env(forward_env=["MY_KEY"])
    env._env = {"MY_KEY": "static_value"}

    monkeypatch.setenv("MY_KEY", "dynamic_value")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()

    assert "MY_KEY" in args
    assert env._init_env_values["MY_KEY"] == "dynamic_value"
    assert not any("static_value" in a for a in args)


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


def _bind_mount_specs(run_args):
    """Return every spec string passed via ``-v``."""
    return [
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "-v"
    ]


def test_persistent_bind_mounts_survive_a_session_key_task_id(monkeypatch, tmp_path):
    """A gateway session key reaches the persistent sandbox path as-is, and it
    carries colons (``session:agent:main:telegram:dm:<chat_id>``). Docker reads
    every colon in a ``-v`` spec as a field separator, so the raw key made
    ``docker run`` fail with "invalid spec ... too many colons" (exit 125) and
    no tool call could run for any Telegram DM session."""
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        task_id="session:agent:main:telegram:dm:8439114563",
        persistent_filesystem=True,
    )

    specs = _bind_mount_specs(_run_args_from_calls(calls))
    mounts = [s for s in specs if s.endswith((":/root", ":/workspace"))]
    assert len(mounts) == 2, f"expected /root and /workspace binds; got {specs}"
    for spec in mounts:
        source, _, target = spec.rpartition(":")
        assert ":" not in source, (
            f"bind source still contains a colon, docker run would fail with "
            f"'too many colons': {spec}"
        )
        # Docker splits on ':' — a sane spec has exactly source:target.
        assert spec.count(":") == 1, f"spec is not a two-field bind: {spec}"
        assert target in {"/root", "/workspace"}


def test_distinct_session_keys_get_distinct_sandbox_dirs(monkeypatch, tmp_path):
    """Sanitizing colons to underscores is not injective on its own: two
    different chats must not be collapsed onto one persistent sandbox, or one
    DM's ``/root`` (shell history, credentials, installed packages) shows up in
    another's container."""
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    sources = []
    for task_id in (
        "session:agent:main:telegram:dm:111",
        "session:agent:main:telegram:dm:222",
        # Collides with the first key under a plain ':' -> '_' rewrite.
        "session_agent_main_telegram_dm_111",
    ):
        calls = _mock_subprocess_run(monkeypatch)
        _make_dummy_env(task_id=task_id, persistent_filesystem=True)
        specs = _bind_mount_specs(_run_args_from_calls(calls))
        sources.append(
            next(s.rpartition(":")[0] for s in specs if s.endswith(":/root"))
        )

    assert len(set(sources)) == 3, f"sandbox sources collided: {sources}"


def test_sandbox_dir_name_keeps_existing_names_verbatim():
    """The shared container and RL/benchmark rollouts must keep resolving to the
    directory they already use — renaming those strands a user's installed
    packages and /root state in an orphaned sandbox."""
    for value in ("default", "bench-env", "astropy__astropy-12907", "v1.2.3_x"):
        assert docker_env._sandbox_dir_name(value) == value


def test_sandbox_dir_name_drops_separators_docker_and_the_fs_reserve():
    """':' is what docker's -v parser splits on; '/' and '\\' would place the
    sandbox outside its root entirely."""
    for value in (
        "session:agent:main:telegram:dm:8439114563",
        "task/with:weird*chars",
        "..\\..\\escape",
        "../../etc",
    ):
        name = docker_env._sandbox_dir_name(value)
        assert not (set(name) & set(':/\\')), name


def test_sandbox_dir_name_is_stable_across_calls():
    """Cross-process container reuse resolves the sandbox by name, so the
    mapping must be a pure function of the id — no randomness, no pid."""
    value = "session:agent:main:discord:guild:1:2"
    assert docker_env._sandbox_dir_name(value) == docker_env._sandbox_dir_name(value)


def test_sandbox_dir_name_bounds_pathological_ids():
    """Long keys (a Matrix room plus thread id) must stay inside the
    per-component filesystem limit."""
    name = docker_env._sandbox_dir_name("session:" + "x:" * 500)
    assert 0 < len(name) <= 128


def test_sandbox_dir_name_never_resolves_to_the_sandbox_root():
    """'.'/'..' would mount the docker sandbox root, and an empty component
    would bind every task's state into one container."""
    for value in ("", ".", "..", "  ", None):
        name = docker_env._sandbox_dir_name(value)
        assert name not in {"", ".", ".."}, repr(value)
        assert not (set(name) & set(':/\\')), name


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


def test_shared_container_key_replaces_profile_identity(monkeypatch):
    """Trusted profiles using the same explicit key share the reuse label."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "research")
    _mock_subprocess_run(monkeypatch)

    a = _make_dummy_env(task_id="abc", shared_container_key="team/workspace")
    b = _make_dummy_env(task_id="abc", shared_container_key="team/workspace")

    # Deterministic across processes/profiles, not the profile label, and
    # digest-suffixed (label sanitization alone is lossy).
    assert a._labels["hermes-profile"] == b._labels["hermes-profile"]
    assert a._labels["hermes-profile"] != "research"
    assert a._labels["hermes-profile"].startswith("team_workspace-")


def test_distinct_shared_keys_never_collide(monkeypatch):
    """Label sanitization is lossy — different raw keys MUST NOT resolve to
    one container identity, or two 'isolated' teams silently attach to the
    same running container (filesystem, processes, env)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "research")
    _mock_subprocess_run(monkeypatch)

    # Sanitize-collision pair: both stems clean to "team_workspace".
    a = _make_dummy_env(task_id="abc", shared_container_key="team/workspace")
    b = _make_dummy_env(task_id="abc", shared_container_key="team_workspace")
    assert a._labels["hermes-profile"] != b._labels["hermes-profile"]

    # Truncation pair: identical first 63 chars, differ after.
    long_a = "x" * 70 + "A"
    long_b = "x" * 70 + "B"
    c = _make_dummy_env(task_id="abc", shared_container_key=long_a)
    d = _make_dummy_env(task_id="abc", shared_container_key=long_b)
    assert c._labels["hermes-profile"] != d._labels["hermes-profile"]
    # Both stay within Docker's 63-char label-value bound.
    assert len(c._labels["hermes-profile"]) <= 63
    assert len(d._labels["hermes-profile"]) <= 63


def test_empty_shared_container_key_preserves_profile_isolation(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "research")
    _mock_subprocess_run(monkeypatch)

    env = _make_dummy_env(task_id="abc", shared_container_key="")

    assert env._labels["hermes-profile"] == "research"


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
    from tools.terminal_tool_lifecycle import cleanup_vm

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
        cleanup_vm("session-close-test")
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


# ---------------------------------------------------------------------------
# Out-of-band container removal recovery (issue #36266, PR #36631)
# ---------------------------------------------------------------------------


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


# ── /dev/shm size tests (ported from nanocoai/nanoclaw#2748) ─────────────────


def _shm_run_args(calls):
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    return run_calls[0][0]


def test_shm_size_default_applied(monkeypatch):
    """Docker's 64 MB /dev/shm default breaks Chromium and PyTorch DataLoader
    workers; the sandbox must raise it by default."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env()

    run_args = _shm_run_args(calls)
    assert "--shm-size" in run_args
    assert run_args[run_args.index("--shm-size") + 1] == docker_env._DEFAULT_SHM_SIZE


def test_shm_size_custom_value(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(shm_size="256m")

    run_args = _shm_run_args(calls)
    assert run_args[run_args.index("--shm-size") + 1] == "256m"


@pytest.mark.parametrize("opt_out", ["", "0", "  ", None])
def test_shm_size_opt_out_omits_flag(monkeypatch, opt_out):
    """Empty / '0' / None fall back to Docker's built-in default (no flag)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(shm_size=opt_out)

    run_args = _shm_run_args(calls)
    assert "--shm-size" not in run_args
    assert not any(isinstance(a, str) and a.startswith("--shm-size=") for a in run_args)


@pytest.mark.parametrize("extra", [["--shm-size", "4g"], ["--shm-size=4g"]])
def test_shm_size_skipped_when_user_sets_it_via_extra_args(monkeypatch, extra):
    """A user-supplied --shm-size in docker_extra_args must win unambiguously:
    our default is skipped rather than relying on flag-ordering behavior."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(extra_args=list(extra))

    run_args = _shm_run_args(calls)
    joined = " ".join(run_args)
    assert joined.count("--shm-size") == 1, joined
    assert "4g" in joined


def test_extra_args_set_shm_size_helper():
    assert docker_env._extra_args_set_shm_size(["--shm-size", "2g"]) is True
    assert docker_env._extra_args_set_shm_size(["--shm-size=2g"]) is True
    assert docker_env._extra_args_set_shm_size(["--memory", "512m"]) is False
    assert docker_env._extra_args_set_shm_size([]) is False
    assert docker_env._extra_args_set_shm_size(None) is False
    # non-string entries must not crash (config.yaml can be malformed)
    assert docker_env._extra_args_set_shm_size([42, None, "--shm-size=1g"]) is True


# ── issue #96268: secrets must never appear in docker argv ────────────


def test_forwarded_secret_values_never_in_argv(monkeypatch):
    """Regression #96268: values must not ride in `-e KEY=VALUE` argv.

    /proc/<pid>/cmdline is world-readable on Linux, so any forwarded secret
    placed in the docker client's argv is visible to every local user via
    plain `ps`. Names go in argv (`-e KEY`); values go in the client
    subprocess env (owner-only /proc/<pid>/environ).
    """
    secret = "s3cr3t-gitlab-token-value"
    env = _make_execute_only_env(["GITLAB_TOKEN"])
    monkeypatch.setenv("GITLAB_TOKEN", secret)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    # init path
    init_args = env._build_init_env_args()
    assert "GITLAB_TOKEN" in init_args
    assert all(secret not in a for a in init_args)
    assert env._init_env_values["GITLAB_TOKEN"] == secret

    # runtime path
    run_args, _unsets, values = env._build_runtime_env_args_with_unsets()
    assert "GITLAB_TOKEN" in run_args
    assert all(secret not in a for a in run_args)
    assert values["GITLAB_TOKEN"] == secret

    # _run_bash must put the value into the spawned client's env kwarg
    calls = []
    monkeypatch.setattr(
        docker_env,
        "_popen_bash",
        lambda cmd, stdin_data=None, **kw: calls.append((cmd, kw)) or object(),
    )
    env._init_env_args = init_args
    env._init_env_values = dict(env._init_env_values)
    env._run_bash("true", login=True)
    cmd, kw = calls[0]
    assert all(secret not in str(a) for a in cmd)
    assert (kw.get("env") or {}).get("GITLAB_TOKEN") == secret


def test_docker_run_secret_values_never_in_argv(monkeypatch):
    """Regression #96268 for the `docker run -d` container-start path."""
    secret = "run-time-secret-value"
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"MY_TOKEN": secret})

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    args, kwargs = run_calls[0]
    assert "MY_TOKEN" in args
    assert all(secret not in str(a) for a in args)
    assert (kwargs.get("env") or {}).get("MY_TOKEN") == secret
