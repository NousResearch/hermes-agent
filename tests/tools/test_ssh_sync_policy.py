from unittest.mock import MagicMock

import pytest

from tools.environments import ssh as ssh_env
from tools.terminal_scope import install_and_reset_profile_terminal_scope, reset_terminal_scope, set_terminal_scope


@pytest.mark.parametrize('setting,enabled', [
    ('unset', True),
    ('config', False),
    ('TERMINAL_FILE_SYNC_ENABLED', False),
    ('TERMINAL_SSH_FILE_SYNC_ENABLED', False),
])
def test_ssh_sync_policy_gates_entire_lifecycle(tmp_path, monkeypatch, setting, enabled):
    for key in ('TERMINAL_FILE_SYNC_ENABLED', 'TERMINAL_SSH_FILE_SYNC_ENABLED'):
        monkeypatch.delenv(key, raising=False)
    if setting == 'config':
        (tmp_path / 'config.yaml').write_text('terminal:\n  file_sync_enabled: false\n')
    elif setting != 'unset':
        (tmp_path / '.env').write_text(f'{setting}=false\n')
    monkeypatch.setattr(ssh_env.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(ssh_env, '_ensure_ssh_available', lambda: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, '_establish_connection', lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, '_detect_remote_home', lambda self: '/home/alice')
    dirs = MagicMock()
    init = MagicMock()
    factory = MagicMock()
    monkeypatch.setattr(ssh_env.SSHEnvironment, '_ensure_remote_dirs', dirs)
    monkeypatch.setattr(ssh_env.SSHEnvironment, 'init_session', init)
    monkeypatch.setattr(ssh_env, 'FileSyncManager', factory)
    with install_and_reset_profile_terminal_scope(tmp_path):
        env = ssh_env.SSHEnvironment(host='example.invalid', user='alice')
        env._before_execute()
        env.cleanup()
    init.assert_called_once()
    if enabled:
        dirs.assert_called_once()
        factory.assert_called_once()
        assert factory.return_value.sync.call_count == 2
        factory.return_value.sync_back.assert_called_once()
    else:
        dirs.assert_not_called()
        factory.assert_not_called()
        assert env._sync_manager is None


def test_ssh_sync_policy_does_not_leak_from_ambient_env(tmp_path, monkeypatch):
    monkeypatch.setenv('TERMINAL_FILE_SYNC_ENABLED', 'true')
    monkeypatch.setattr(ssh_env.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(ssh_env, '_ensure_ssh_available', lambda: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, '_establish_connection', lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, '_detect_remote_home', lambda self: '/home/alice')
    monkeypatch.setattr(ssh_env.SSHEnvironment, '_ensure_remote_dirs', lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, 'init_session', lambda self: None)
    factory = MagicMock()
    monkeypatch.setattr(ssh_env, 'FileSyncManager', factory)
    token = set_terminal_scope({'TERMINAL_FILE_SYNC_ENABLED': 'false'})
    try:
        env = ssh_env.SSHEnvironment(host='example.invalid', user='alice')
        env._before_execute()
        env.cleanup()
    finally:
        reset_terminal_scope(token)
    factory.assert_not_called()
