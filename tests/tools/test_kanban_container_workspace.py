"""Task workspace authority survives profile config and maps to Apple guest paths."""
import pytest


@pytest.mark.parametrize('backend', ['docker', 'apple_container'])
def test_task_workspace_wins_over_profile_cwd(monkeypatch, tmp_path, backend):
    from hermes_cli.config import apply_terminal_config_to_env
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    from tools.terminal_tool import _get_env_config, _resolve_task_host_cwd
    task = tmp_path / 'task'
    other = tmp_path / 'profile-root'
    task.mkdir()
    other.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_test')
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACE', str(task))
    cfg = {'terminal': {'backend': backend, 'cwd': str(other), backend+'_mount_cwd_to_workspace': True}}
    import yaml
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg))
    env = {'HERMES_KANBAN_TASK': 't_test', 'HERMES_KANBAN_WORKSPACE': str(task)}
    apply_terminal_config_to_env(env=env, config=cfg)
    assert env['TERMINAL_CWD'] == str(task.resolve())
    token = install_profile_terminal_scope(tmp_path)
    try:
        config = _get_env_config()
        assert config['cwd'] == '/workspace'
        assert _resolve_task_host_cwd(config, 'task') == str(task.resolve())
        if backend == 'apple_container':
            from tools.terminal_tool import _docker_has_host_access
            from tools.approval import check_execute_code_guard
            assert _docker_has_host_access(config)
            monkeypatch.setenv('HERMES_CRON_SESSION', '1')
            assert check_execute_code_guard('print(1)', backend, has_host_access=True)['approved'] is False
    finally:
        reset_terminal_scope(token)


def test_invalid_assigned_workspace_refuses_profile_fallback(monkeypatch, tmp_path):
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope, terminal_env, TerminalPolicyUnavailable
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_test')
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACE', str(tmp_path/'missing'))
    token = install_profile_terminal_scope(tmp_path)
    try:
        with pytest.raises(TerminalPolicyUnavailable):
            terminal_env('TERMINAL_CWD')
    finally:
        reset_terminal_scope(token)


def test_dispatch_workspace_survives_worker_profile_loading(monkeypatch, tmp_path):
    import yaml
    from hermes_cli import kanban_db as kb
    from tests.hermes_cli.test_kanban_worker_terminal_cwd import _capture_spawn_env
    from hermes_cli.config import apply_terminal_config_to_env
    from tools.terminal_scope import build_profile_terminal_scope
    root = tmp_path / 'home'
    profile = root / 'profiles' / 'w'
    profile.mkdir(parents=True)
    workspace = tmp_path / 'selected'
    workspace.mkdir()
    cfg = {'terminal': {'backend': 'apple_container', 'cwd': str(root),
                       'apple_container_mount_cwd_to_workspace': True}, 'toolsets': ['kanban']}
    (profile/'config.yaml').write_text(yaml.safe_dump(cfg))
    (root/'config.yaml').write_text('toolsets: [kanban]\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    env = captured['env']
    apply_terminal_config_to_env(env=env, config=cfg)
    assert env['TERMINAL_CWD'] == str(workspace.resolve())
    for key in ('HERMES_KANBAN_TASK', 'HERMES_KANBAN_WORKSPACE'):
        monkeypatch.setenv(key, env[key])
    scope = build_profile_terminal_scope(profile)
    assert scope['TERMINAL_CWD'] == str(workspace.resolve())
    assert scope['TERMINAL_APPLE_CONTAINER_MOUNT_CWD_TO_WORKSPACE'].lower() == 'true'


def test_different_task_workspaces_do_not_reuse_apple_container(monkeypatch, tmp_path):
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    from tools.terminal_tool import _resolve_container_task_id
    one = tmp_path / 'one'
    two = tmp_path / 'two'
    one.mkdir()
    two.mkdir()
    token = set_terminal_scope({'TERMINAL_ENV': 'apple_container',
                               'TERMINAL_APPLE_CONTAINER_MOUNT_CWD_TO_WORKSPACE': 'true'})
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'task')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile-one'))
    try:
        monkeypatch.setenv('HERMES_KANBAN_WORKSPACE', str(one))
        first = _resolve_container_task_id('worker')
        assert _resolve_container_task_id('delegate') == first
        monkeypatch.setenv('HERMES_KANBAN_WORKSPACE', str(two))
        assert _resolve_container_task_id('worker') != first
        monkeypatch.setenv('HERMES_KANBAN_WORKSPACE', str(one))
        monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile-two'))
        assert _resolve_container_task_id('worker') != first
    finally:
        reset_terminal_scope(token)


@pytest.mark.parametrize("backend", ["docker", "apple_container"])
def test_prompt_probe_uses_assigned_workspace_before_scope_loading(monkeypatch, tmp_path, backend):
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    from tools.terminal_tool import _resolve_config_cwd
    task = tmp_path / "task"
    profile = tmp_path / "profile"
    task.mkdir()
    profile.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    token = set_terminal_scope({"TERMINAL_ENV": backend, "TERMINAL_CWD": str(profile)})
    try:
        assert _resolve_config_cwd(backend, True) == ("/workspace", str(task.resolve()))
    finally:
        reset_terminal_scope(token)
