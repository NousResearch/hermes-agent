"""Requested worktree isolation must stop a child before it can run in shared files."""
import threading
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import delegate_tool


class Child:
    tool_progress_callback = None
    _credential_pool = None
    _delegate_saved_tool_names = []
    _delegate_role = 'leaf'
    _delegate_depth = 1
    _subagent_id = None
    model = 'fixture'
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_cost_status = 'unknown'

    def __init__(self):
        self.ran = False
        self.closed = False

    def run_conversation(self, **kwargs):
        self.ran = True
        return {'final_response': 'ran', 'completed': True, 'messages': []}

    def get_activity_summary(self):
        return {}

    def close(self):
        self.closed = True


def test_requested_isolation_refuses_non_git_before_running_child(tmp_path, monkeypatch):
    from tools import subagent_worktree
    from tools.terminal_tool import record_session_cwd
    monkeypatch.setattr(delegate_tool, '_get_worktree_isolation', lambda: True)
    monkeypatch.setattr(subagent_worktree, 'local_backend_active', lambda: True)
    child = Child()
    parent = SimpleNamespace(session_id='isolation-parent', _current_task_id='isolation-parent',
                             _active_children=[child], _active_children_lock=threading.Lock())
    record_session_cwd(parent._current_task_id, str(tmp_path))
    result = delegate_tool._run_single_child(0, 'must not run shared', child, parent)
    assert not child.ran, result
    assert child.closed
    assert result['status'] == 'error'
    error = result['error']
    assert 'needs a Git repository' in error and repr(str(tmp_path)) in error, error
    assert 'child was not run' in error


def test_requested_isolation_refuses_when_no_parent_cwd_is_known(monkeypatch):
    from tools import subagent_worktree, terminal_tool
    monkeypatch.setattr(delegate_tool, '_get_worktree_isolation', lambda: True)
    monkeypatch.setattr(delegate_tool, '_resolve_workspace_hint', lambda parent: None)
    monkeypatch.setattr(subagent_worktree, 'local_backend_active', lambda: True)
    monkeypatch.setattr(terminal_tool, 'get_session_cwd', lambda key: None)
    child = Child()
    parent = SimpleNamespace(session_id='no-cwd-parent', _current_task_id='no-cwd-parent',
                             _active_children=[child], _active_children_lock=threading.Lock())
    result = delegate_tool._run_single_child(0, 'must not run without a workspace', child, parent)
    assert not child.ran, result
    assert result['status'] == 'error'
    assert 'none is recorded for this session' in result['error'], result['error']
    assert 'child was not run' in result['error']


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    for args in [('init', '-b', 'main'), ('config', 'user.name', 'Fixture'),
                 ('config', 'user.email', 'fixture@example.invalid')]:
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    (repo / 'base').write_text('base')
    for args in [('add', 'base'), ('commit', '-m', 'base')]:
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    return repo


@pytest.mark.parametrize('mode', ['nonlocal', 'creation_failure', 'cwd_seed_failure', 'success', 'opt_out'])
def test_child_runs_only_after_requested_boundary_is_ready(tmp_path, monkeypatch, mode):
    from tools import subagent_worktree, terminal_tool
    repo = _repo(tmp_path)
    monkeypatch.setattr(delegate_tool, '_get_worktree_isolation', lambda: mode != 'opt_out')
    monkeypatch.setattr(subagent_worktree, 'local_backend_active', lambda: mode != 'nonlocal')
    if mode == 'creation_failure':
        # An unborn repo exercises a real git worktree-add failure.
        empty = tmp_path / 'unborn'
        empty.mkdir()
        subprocess.run(['git', 'init', str(empty)], check=True, capture_output=True)
        repo = empty
    if mode == 'cwd_seed_failure':
        record = terminal_tool.record_session_cwd
        def refuse_tree(session_key, cwd):
            if cwd and '.worktrees' in Path(cwd).parts:
                raise OSError('fixture worktree cwd seed failed')
            return record(session_key, cwd)
        monkeypatch.setattr(terminal_tool, 'record_session_cwd', refuse_tree)
    child = Child()
    parent = SimpleNamespace(session_id='boundary-parent', _current_task_id='boundary-parent',
                             _active_children=[child], _active_children_lock=threading.Lock())
    terminal_tool.record_session_cwd(parent._current_task_id, str(repo))
    observed = []
    def run(**kwargs):
        child.ran = True
        cwd = terminal_tool.get_session_cwd(kwargs['task_id'])
        observed.append(cwd)
        (Path(cwd) / 'child-output').write_text('owned child output')
        return {'final_response': 'ran', 'completed': True, 'messages': []}
    child.run_conversation = run
    result = delegate_tool._run_single_child(0, 'write only in owned workspace', child, parent)
    assert child.closed
    assert child not in parent._active_children
    if mode == 'cwd_seed_failure':
        workspace = result['worktree']
        assert workspace['pruned'] or workspace.get('note'), workspace
    if mode in {'success', 'opt_out'}:
        assert child.ran and result['status'] == 'completed', result
        if mode == 'success':
            assert '.worktrees' in Path(observed[0]).parts
            assert not (repo / 'child-output').exists()
        else:
            assert observed == [str(repo)]
            assert (repo / 'child-output').exists()
    else:
        assert not child.ran, result
        assert result['status'] == 'error'
        expected_cause = {
            'nonlocal': 'needs terminal.backend: local',
            'creation_failure': f'could not add a worktree in {str(repo.resolve())!r}',
            'cwd_seed_failure': 'OSError: fixture worktree cwd seed failed',
        }[mode]
        assert expected_cause in result['error'], result['error']
        assert 'child was not run' in result['error']
        if mode == 'cwd_seed_failure':
            assert "'worktree' result" in result['error'], result['error']
            assert 'delegation.worktree_isolation: false' in result['error'], result['error']
        assert not (repo / 'child-output').exists()
