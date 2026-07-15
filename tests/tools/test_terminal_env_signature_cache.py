"""Regression tests for terminal runtime cache invalidation."""

import json
from unittest.mock import MagicMock, patch


def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "docker_mount_cwd_to_workspace": False,
        "docker_volumes": [],
        "docker_forward_env": [],
        "docker_env": {},
        "docker_run_as_host_user": False,
        "docker_network": True,
        "docker_extra_args": [],
        "docker_persist_across_processes": True,
        "docker_orphan_reaper": True,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": 22,
        "ssh_key": "",
        "ssh_persistent": True,
        "local_persistent": False,
    }
    config.update(overrides)
    return config


def _reset_terminal_cache(tt):
    tt._active_environments.clear()
    tt._last_activity.clear()
    tt._creation_locks.clear()
    tt._task_env_overrides.clear()


def _reset_file_ops_cache(ft):
    ft._file_ops_cache.clear()
    ft._last_known_cwd.clear()


def test_terminal_env_cache_invalidates_when_backend_signature_changes(monkeypatch):
    """A session cached under SSH must not be reused after runtime switches local."""
    from tools import terminal_tool as tt

    _reset_terminal_cache(tt)
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-default")

    task_id = "session-1"
    ssh_config = _make_env_config(
        env_type="ssh",
        cwd="~",
        ssh_host="old-ssh.example.test",
        ssh_user="agent-user",
    )
    old_env = MagicMock(name="old_ssh_env")
    setattr(
        old_env,
        tt._ENV_SIGNATURE_ATTR,
        tt._terminal_env_signature(ssh_config, task_id=task_id),
    )
    tt._active_environments[task_id] = old_env
    tt._last_activity[task_id] = 1

    local_config = _make_env_config(
        env_type="local",
        cwd="/workspace/local-project",
    )
    new_env = MagicMock(name="new_local_env")
    new_env.execute.return_value = {"output": "local-host\n", "returncode": 0}

    with patch("tools.terminal_tool._get_env_config", return_value=local_config), \
         patch("tools.terminal_tool._create_environment", return_value=new_env) as create_env, \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        result = json.loads(tt.terminal_tool("hostname", task_id=task_id))

    assert result["output"] == "local-host"
    old_env.cleanup.assert_called_once_with()
    create_env.assert_called_once()
    assert create_env.call_args.kwargs["env_type"] == "local"
    assert create_env.call_args.kwargs["cwd"] == "/workspace/local-project"
    assert tt._active_environments["default"] is new_env
    assert getattr(new_env, tt._ENV_SIGNATURE_ATTR)["env_type"] == "local"
    assert getattr(new_env, tt._ENV_SIGNATURE_ATTR)["ssh_host"] == ""

    _reset_terminal_cache(tt)


def test_terminal_env_cache_reuses_when_signature_matches(monkeypatch):
    """Matching runtime signature still preserves the intended per-session cache."""
    from tools import terminal_tool as tt

    _reset_terminal_cache(tt)
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-default")

    task_id = "session-2"
    local_config = _make_env_config(env_type="local", cwd="/tmp/project")
    cached_env = MagicMock(name="cached_local_env")
    cached_env.execute.return_value = {"output": "ok\n", "returncode": 0}
    setattr(
        cached_env,
        tt._ENV_SIGNATURE_ATTR,
        tt._terminal_env_signature(local_config, task_id="default"),
    )
    tt._active_environments["default"] = cached_env
    tt._last_activity["default"] = 1

    with patch("tools.terminal_tool._get_env_config", return_value=local_config), \
         patch("tools.terminal_tool._create_environment") as create_env, \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        result = json.loads(tt.terminal_tool("echo ok", task_id=task_id))

    assert result["output"] == "ok"
    create_env.assert_not_called()
    cached_env.cleanup.assert_not_called()
    assert tt._active_environments["default"] is cached_env

    _reset_terminal_cache(tt)


def test_terminal_env_cache_reuses_raw_session_key_when_signature_matches(monkeypatch):
    """A raw-key cached session remains reusable after cwd-only task collapse."""
    from tools import terminal_tool as tt

    _reset_terminal_cache(tt)
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-default")

    task_id = "session-raw"
    local_config = _make_env_config(env_type="local", cwd="/tmp/raw-project")
    cached_env = MagicMock(name="cached_raw_local")
    cached_env.execute.return_value = {"output": "raw-ok\n", "returncode": 0}
    setattr(
        cached_env,
        tt._ENV_SIGNATURE_ATTR,
        tt._terminal_env_signature(local_config, task_id="default"),
    )
    tt._active_environments[task_id] = cached_env
    tt._last_activity[task_id] = 1

    with patch("tools.terminal_tool._get_env_config", return_value=local_config), \
         patch("tools.terminal_tool._create_environment") as create_env, \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        result = json.loads(tt.terminal_tool("echo raw-ok", task_id=task_id))

    assert result["output"] == "raw-ok"
    create_env.assert_not_called()
    cached_env.cleanup.assert_not_called()
    assert tt._active_environments[task_id] is cached_env

    _reset_terminal_cache(tt)


def test_file_ops_cache_invalidates_when_backend_signature_changes(monkeypatch):
    """File tools must not keep ShellFileOperations for a stale backend."""
    from tools import file_tools as ft
    from tools import terminal_tool as tt

    _reset_terminal_cache(tt)
    _reset_file_ops_cache(ft)
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-default")

    ssh_config = _make_env_config(
        env_type="ssh",
        cwd="~",
        ssh_host="old-ssh.example.test",
        ssh_user="agent-user",
    )
    old_backend = MagicMock(name="old_remote")
    setattr(old_backend, "cwd", "~")
    setattr(
        old_backend,
        tt._ENV_SIGNATURE_ATTR,
        tt._terminal_env_signature(ssh_config, task_id="default"),
    )
    tt._active_environments["default"] = old_backend
    tt._last_activity["default"] = 1
    old_file_ops = ft.ShellFileOperations(old_backend)
    ft._file_ops_cache["default"] = old_file_ops

    local_config = _make_env_config(env_type="local", cwd="/workspace/local-project")
    new_backend = MagicMock(name="new_local")
    setattr(new_backend, "cwd", "/workspace/local-project")

    with patch("tools.terminal_tool._get_env_config", return_value=local_config), \
         patch("tools.terminal_tool._create_environment", return_value=new_backend) as create_env, \
         patch("tools.terminal_tool._start_cleanup_thread"):
        file_ops = ft._get_file_ops("session-file")

    old_backend.cleanup.assert_called_once_with()
    create_env.assert_called_once()
    assert create_env.call_args.kwargs["env_type"] == "local"
    assert getattr(file_ops, "env") is new_backend
    assert getattr(ft._file_ops_cache["default"], "env") is new_backend
    assert tt._active_environments["default"] is new_backend

    _reset_file_ops_cache(ft)
    _reset_terminal_cache(tt)


def test_docker_creation_inputs_affect_runtime_signature(monkeypatch):
    """Docker creation knobs that change runtime compatibility are signed."""
    from tools import terminal_tool as tt

    _reset_terminal_cache(tt)
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-default")

    base_config = _make_env_config(
        env_type="docker",
        cwd="/workspace",
        docker_image="python:3.12",
    )
    base = tt.resolve_terminal_runtime_identity(base_config)["signature"]

    for key, value in {
        "docker_extra_args": ["--add-host=host.docker.internal:host-gateway"],
        "docker_network": False,
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
    }.items():
        changed_config = _make_env_config(
            env_type="docker",
            cwd="/workspace",
            docker_image="python:3.12",
            **{key: value},
        )
        changed = tt.resolve_terminal_runtime_identity(changed_config)["signature"]
        assert changed != base
        assert changed[key] != base[key]

    runtime = tt.resolve_terminal_runtime_identity(
        _make_env_config(
            env_type="docker",
            cwd="/workspace",
            docker_image="python:3.12",
            docker_extra_args=["--init"],
            docker_network=False,
            docker_persist_across_processes=False,
            docker_orphan_reaper=False,
        )
    )
    container_config = runtime["container_config"]
    assert container_config["docker_extra_args"] == ["--init"]
    assert container_config["docker_network"] is False
    assert container_config["docker_persist_across_processes"] is False
    assert container_config["docker_orphan_reaper"] is False
