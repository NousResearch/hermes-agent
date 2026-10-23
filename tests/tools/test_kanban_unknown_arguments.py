"""Unknown review arguments must not silently mutate the board (#115641)."""
import json
from pathlib import Path


def test_review_unknown_argument_rejects_before_handoff(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    for key in ('HERMES_KANBAN_TASK', 'HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_RUN_ID'):
        monkeypatch.delenv(key, raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools  # register actual handlers
    from tools.registry import registry
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='review payload', assignee='builder', workspace_kind='scratch')
        assert kb.claim_task(conn, tid) is not None
        workspace = kb.workspaces_root() / tid
        workspace.mkdir(parents=True)
        conn.execute('UPDATE tasks SET workspace_path=? WHERE id=?', (str(workspace), tid))
        conn.commit()
    artifact = workspace / 'result.txt'
    artifact.write_text('verified result', encoding='utf-8')
    result = json.loads(registry.dispatch('kanban_request_review', {
        'task_id': tid, 'summary': 'verified', 'summary_artifacts': [str(artifact)]}))
    with kbc.connect_closing() as conn:
        status = kb.get_task(conn, tid).status
        attachments = kb.list_attachments(conn, tid)
    assert result.get('error') and 'summary_artifacts' in result['error'], (result, status, attachments)
    assert status == 'running'
    assert not attachments
    corrected = json.loads(registry.dispatch('kanban_request_review', {
        'task_id': tid, 'summary': 'verified', 'artifacts': [str(artifact)],
        'metadata': {'custom_key': 'allowed'}}))
    assert corrected.get('ok'), corrected
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == 'review'
        assert kb.list_attachments(conn, tid)


def test_registered_kanban_tools_reject_unknown_keys():
    from tools import kanban_tools
    from tools.registry import registry
    for name, schema, handler, emoji in kanban_tools._TOOLS:
        result = json.loads(registry.dispatch(name, {'misspelled_argument': True}))
        assert 'unknown parameter' in result.get('error', ''), (name, result)
        assert 'misspelled_argument' in result['error']


def test_undeclared_internal_keys_survive_the_strict_check():
    """``project_id`` (legacy alias of ``project``) and ``session_id`` (internal
    provenance) are read by ``_handle_create`` without being in the LLM-facing
    schema; the unknown-key gate must not reject them (#115641 follow-up)."""
    from tools import kanban_tools
    from tools.registry import registry
    # No board in this test: an accepted key set reaches the handler proper and
    # fails on the *next* validation, never on "unknown parameter".
    result = json.loads(registry.dispatch(
        'kanban_create', {'title': 'x', 'project_id': '', 'session_id': 's'}))
    assert 'unknown parameter' not in result.get('error', ''), result
    assert 'project_id' in kanban_tools._UNDECLARED_ARGS['kanban_create']
    # ``title`` is the filename alias ``_handle_attach_url`` still reads (review follow-up).
    result = json.loads(registry.dispatch(
        'kanban_attach_url', {'task_id': 'x', 'url': 'https://example.invalid/y.pdf', 'title': 'spec'}))
    assert 'unknown parameter' not in result.get('error', ''), result
