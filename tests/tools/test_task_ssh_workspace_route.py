import json

from tools import code_execution_tool as cet
from tools import file_tools as ft
from tools import terminal_tool as tt


class _DummyEnv:
    cwd = "/workspace/project"

    def execute(self, *_args, **_kwargs):
        return {"output": "ok", "exit_code": 0}


def test_task_override_selects_ssh_backend_and_disables_sync(monkeypatch):
    captured = {}

    def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
        captured.update(
            env_type=env_type,
            cwd=cwd,
            ssh_config=kwargs.get("ssh_config"),
        )
        return _DummyEnv()

    monkeypatch.setattr(
        tt,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": "/srv/hermes",
            "timeout": 180,
            "lifetime_seconds": 300,
            "local_persistent": False,
        },
    )
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_container_aliases", {})

    task_id = "acp-session"
    tt.register_task_env_overrides(
        task_id,
        {
            "env_type": "ssh",
            "cwd": "/workspace/project",
            "ssh_host": "workspace.example",
            "ssh_user": "developer",
            "ssh_port": 2222,
            "ssh_key": "~/.ssh/workspace",
            "ssh_sync": False,
        },
    )

    result = tt.terminal_tool(command="pwd", task_id=task_id)

    assert captured == {
        "env_type": "ssh",
        "cwd": "/workspace/project",
        "ssh_config": {
            "host": "workspace.example",
            "user": "developer",
            "port": 2222,
            "key": "~/.ssh/workspace",
            "persistent": False,
            "sync": False,
        },
    }
    assert json.loads(result)["exit_code"] == 0


def test_unrouted_task_keeps_profile_local_backend(monkeypatch):
    captured = {}
    _install_local_config(monkeypatch)

    def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
        captured.update(env_type=env_type, cwd=cwd, ssh_config=kwargs.get("ssh_config"))
        return _DummyEnv()

    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)

    result = tt.terminal_tool(command="pwd", task_id="ordinary-session")

    assert captured["env_type"] == "local"
    assert captured["ssh_config"] is None
    assert json.loads(result)["exit_code"] == 0


def test_delegated_task_inherits_parent_ssh_route(monkeypatch):
    captured = {}
    _install_local_config(monkeypatch)

    def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
        captured.update(env_type=env_type, cwd=cwd, ssh_config=kwargs.get("ssh_config"))
        return _DummyEnv()

    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)

    parent_id = "acp-parent-session"
    child_id = "delegated-child"
    _register_workspace_route(parent_id)
    tt.register_container_alias(child_id, parent_id)

    result = tt.terminal_tool(command="pwd", task_id=child_id)

    assert tt._resolve_container_task_id(child_id) == parent_id
    assert captured["env_type"] == "ssh"
    assert captured["cwd"] == "/workspace/project"
    assert captured["ssh_config"]["sync"] is False
    assert json.loads(result)["exit_code"] == 0


def _install_local_config(monkeypatch):
    monkeypatch.setattr(
        tt,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": "/srv/hermes",
            "timeout": 180,
            "lifetime_seconds": 300,
            "local_persistent": False,
        },
    )
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_container_aliases", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})


def _register_workspace_route(task_id):
    tt.register_task_env_overrides(
        task_id,
        {
            "env_type": "ssh",
            "cwd": "/workspace/project",
            "ssh_host": "workspace.example",
            "ssh_user": "developer",
            "ssh_port": 2222,
            "ssh_key": "~/.ssh/workspace",
            "ssh_sync": False,
        },
    )


def test_file_tools_use_task_scoped_ssh_route(monkeypatch):
    captured = {}
    _install_local_config(monkeypatch)

    def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
        captured.update(env_type=env_type, cwd=cwd, ssh_config=kwargs.get("ssh_config"))
        return _DummyEnv()

    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
    task_id = "acp-file-session"
    _register_workspace_route(task_id)

    ft._get_file_ops(task_id)

    assert captured["env_type"] == "ssh"
    assert captured["cwd"] == "/workspace/project"
    assert captured["ssh_config"]["sync"] is False


def test_execute_code_uses_task_scoped_ssh_route(monkeypatch):
    captured = {}
    _install_local_config(monkeypatch)

    def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
        captured.update(env_type=env_type, cwd=cwd, ssh_config=kwargs.get("ssh_config"))
        return _DummyEnv()

    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
    task_id = "acp-code-session"
    _register_workspace_route(task_id)

    _env, env_type = cet._get_or_create_env(task_id)

    assert env_type == "ssh"
    assert captured["cwd"] == "/workspace/project"
    assert captured["ssh_config"]["sync"] is False
