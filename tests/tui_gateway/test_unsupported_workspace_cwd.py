"""Unsupported plugin workspaces cannot silently switch to a gateway host cwd."""

import pytest

from tui_gateway import server as session_workdir


@pytest.mark.parametrize('cwd_kind', ['host-existing', 'remote-only'])
def test_custom_backend_cwd_switch_refuses_without_host_validation(cwd_kind, tmp_path, monkeypatch):
    monkeypatch.setattr(session_workdir, '_bound_terminal_backend', lambda home: 'isolated_remote')
    cwd = str(tmp_path) if cwd_kind == 'host-existing' else '/remote-only/project'
    with pytest.raises(ValueError, match='unavailable.*isolated_remote'):
        session_workdir._workspace_cwd(tmp_path / 'profile', cwd)


def test_local_cwd_switch_still_validates_host_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(session_workdir, '_bound_terminal_backend', lambda home: 'local')
    assert session_workdir._workspace_cwd(tmp_path / 'profile', str(tmp_path)) == str(tmp_path)
