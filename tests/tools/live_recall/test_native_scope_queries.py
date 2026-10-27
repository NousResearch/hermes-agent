"""Query real SQLite sessions and the recall tool under an actual bound gateway context."""
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def sessions(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(tmp_path / 'config.yaml', {'lsp': {'enabled': False}})
    from hermes_state import SessionDB
    from gateway.recall_scope import canonical_recall_identity

    db = SessionDB(tmp_path / 'state.db')
    chat, topic, other_chat, other_topic = (uuid4().hex for _ in range(4))
    origin = {'platform': 'telegram', 'chat_type': 'group', 'chat_id': chat, 'thread_id': topic, 'profile': 'default'}
    content = (Path(__file__).resolve().parents[3] / 'README.md').read_text(encoding='utf-8')
    routes = [origin, dict(origin), {**origin, 'thread_id': other_topic}, {**origin, 'chat_id': other_chat}, None]
    ids = []
    for route in routes:
        sid = uuid4().hex
        db.create_session(sid, source='telegram', model='gpt-6.1-sol',
                          origin_json=json.dumps(route) if route is not None else None)
        db.append_message(sid, 'user', content)
        ids.append(sid)
    identity = canonical_recall_identity(origin)
    assert identity is not None
    try:
        yield db, ids, origin, identity
    finally:
        db.close()
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


@pytest.mark.parametrize('surface', ['fts', 'recent', 'detail'])
def test_sql_scope_excludes_other_chats_threads_and_unknown_origins(sessions, surface):
    db, ids, _origin, scope = sessions
    if surface == 'fts':
        scoped = db.search_messages('Hermes', recall_scope=scope)
        assert {row['session_id'] for row in scoped} == set(ids[:2])
        assert {row['session_id'] for row in db.search_messages('Hermes')} == set(ids)
    elif surface == 'recent':
        scoped = db.list_recent_sessions_bounded(limit=10, recall_scope=scope)
        assert {row['id'] for row in scoped} == set(ids[:2])
    else:
        assert db.get_session_rich_row(ids[0], recall_scope=scope)['id'] == ids[0]
        for sid in ids[2:]:
            assert db.get_session_rich_row(sid, recall_scope=scope) is None


def test_real_tool_defaults_to_bound_chat_and_requires_explicit_all(sessions):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.session_search_tool import session_search

    db, ids, origin, _scope = sessions
    tokens = set_session_vars(platform=origin['platform'], chat_id=origin['chat_id'],
                              chat_type=origin['chat_type'], thread_id=origin['thread_id'],
                              profile='default', session_id=ids[0], gateway_context=True)
    try:
        browse = json.loads(session_search(db=db, current_session_id=ids[0], limit=10))
        assert browse['success'] is True, browse
        assert {row['session_id'] for row in browse['results']} == {ids[1]}
        denied = json.loads(session_search(db=db, current_session_id=ids[0], session_id=ids[2]))
        assert denied['success'] is False, denied
        allowed = json.loads(session_search(db=db, current_session_id=ids[0], session_id=ids[2], scope='all'))
        assert allowed['success'] is True and allowed['session_id'] == ids[2], allowed
    finally:
        clear_session_vars(tokens)


def test_workspace_maintenance_uses_the_same_filter_owner(sessions, tmp_path):
    db, ids, _origin, _scope = sessions
    root = Path(__file__).resolve().parents[3]
    db.update_session_cwd(ids[0], str(root))
    db.update_session_cwd(ids[1], str(tmp_path))
    matching = db.list_prune_candidates(cwd_prefix=str(root), include_open=True)
    assert {row['id'] for row in matching} == {ids[0]}
    assert {row['id'] for row in db.list_prune_candidates(include_open=True)} == set(ids)


def test_empty_cron_query_executes_with_all_named_parameters(sessions):
    db, _ids, _origin, _scope = sessions
    assert db.list_cron_job_runs(uuid4().hex, limit=1, offset=0) == []
