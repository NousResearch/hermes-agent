"""Workers retain board-owned project identity across profile-local registries."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('reference', ['id', 'slug', 'inherited'])
def test_worker_child_uses_parent_project_from_shared_board_home(tmp_path, monkeypatch, reference):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, projects_db as pdb
    from tools import kanban_tools  # noqa: F401
    from tools.registry import registry

    root = tmp_path / '.hermes'
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(root))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    repo = tmp_path / 'hermes-source'
    repo.mkdir()
    (root / 'profiles' / 'architect').mkdir(parents=True)
    (root / 'profiles' / 'coding-expert').mkdir(parents=True)
    with pdb.connect_closing() as conn:
        project = pdb.create_project(conn, name='Hermes', slug='hermes-agent', primary_path=str(repo))
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title='Discovery', assignee='architect', project_id=project,
                                workspace_kind='dir', workspace_path=str(tmp_path / 'discovery'))
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles/architect'))
    monkeypatch.setenv('HERMES_PROFILE', 'architect')
    monkeypatch.setenv('HERMES_KANBAN_TASK', parent)
    args = dict(title='Repair', assignee='coding-expert', parents=[parent])
    if reference != 'inherited':
        args['project'] = project if reference == 'id' else 'hermes-agent'
    result = json.loads(registry.dispatch('kanban_create', args))
    assert result.get('ok'), result
    with kbc.connect() as conn:
        child = kb.get_task(conn, result['task_id'])
        assert child.project_id == project
        assert Path(child.workspace_path).parent == repo / '.worktrees'
        assert child.workspace_path != str(tmp_path / 'discovery')

    args['project'] = 'unregistered-project'
    with kbc.connect() as conn:
        before = conn.execute('SELECT count(*) FROM tasks').fetchone()[0]
    rejected = json.loads(registry.dispatch('kanban_create', args))
    assert not rejected.get('ok'), rejected
    with kbc.connect() as conn:
        assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == before
