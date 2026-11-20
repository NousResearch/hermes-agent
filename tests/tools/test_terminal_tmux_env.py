"""Both local spawners get the tool call's owner; the shared backend retains neither owner."""
import os

import pytest

from gateway.session_context import scoped_current_session_id
from tools.environments.local import _make_run_env, _terminal_env, terminal_tmux_scope


@pytest.mark.skipif(os.name != 'posix', reason='local Unix sockets')
@pytest.mark.parametrize('build_env', [_make_run_env, _terminal_env])
def test_env_uses_physical_owner_without_mutating_shell_command(monkeypatch, tmp_path, build_env):
    from tools import session_tmux

    def resolve(owner):
        return tmp_path / owner

    monkeypatch.setattr(session_tmux, 'session_directory', resolve)
    inherited = {'TMUX': '/outer/server,123,0', 'TMUX_TMPDIR': '/outer'}
    with scoped_current_session_id('physical'), terminal_tmux_scope('route-not-owner', local=True):
        env = build_env({**inherited, 'PATH': os.defpath})
        assert env['TMUX_TMPDIR'] == str(tmp_path / 'physical')
        assert 'TMUX' not in env
    with terminal_tmux_scope('control-plane', local=False):
        env = build_env({**inherited, 'PATH': os.defpath})
        assert all(env[k] == v for k, v in inherited.items())
