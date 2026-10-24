"""Tool-heavy observation regressions for aka-lacie/hermes-plugins#3."""
import json
import re

import pytest

from hermes_state import SessionDB
from tools.delegate_session_store import DelegateSessionStore


@pytest.fixture
def history(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('parent', 'cli')
        db.create_session('child', 'subagent', parent_session_id='parent',
                          model_config={'_delegate_from': 'parent'})
        store = DelegateSessionStore(db)
        store.register('parent', 'child', model='test', provider='test', assignment='Inspect')
        yield db, store


def test_tool_heavy_window_keeps_steps_and_headers_without_changing_history(history):
    db, store = history
    for i in range(5):
        db.append_message('child', 'assistant', '', tool_calls=[{
            'id': f'call-{i}', 'type': 'function', 'function': {
                'name': 'terminal', 'arguments': json.dumps({'command': f'cat file-{i}.txt'})}}])
        db.append_message('child', 'tool', f'HEADER-{i}\n' + 'x' * 10000 + f'\nSTATUS-{i}',
                          tool_name='terminal', tool_call_id=f'call-{i}')
    before = db.get_messages('child')
    session = store.get('parent', 'child')
    result = store.observe('parent', 'child', limit=20)
    assert len(result['messages']) == 10
    assert result['truncated']
    for i in range(5):
        call, output = result['messages'][i * 2:i * 2 + 2]
        assert call['tool_calls'] == [{'name': 'terminal', 'command_preview': f'cat file-{i}.txt'}]
        assert output['tool_name'] == 'terminal'
        assert len(output['content']) <= 1500
        assert output['content'].startswith(f'HEADER-{i}\n')
        assert output['content'].endswith(f'\nSTATUS-{i}')
        marker = re.search(r'\n\[([0-9]+) chars omitted\]\n', output['content'])
        assert marker
        assert len(output['content']) - len(marker[0]) + int(marker[1]) == len(before[i * 2 + 1]['content'])
    assert db.get_messages('child') == before
    assert store.get('parent', 'child') == session


@pytest.mark.parametrize('budget', [1, 20, 50, 12000])
def test_observe_bounds_metadata_and_preserves_generous_prose(history, budget):
    db, store = history
    prose = 'Introduction\n' + 'p' * 3000 + '\nConclusion'
    db.append_message('child', 'assistant', prose, tool_calls=[{
        'function': {'name': 'write_file', 'arguments': json.dumps({
            'path': '/project/DESIGN.md', 'content': 'private payload' * 10000})}},
        {'function': {'name': 'broken', 'arguments': '{invalid'}},
    ])
    result = store.observe('parent', 'child', max_chars=budget)
    row = result['messages'][0]
    metadata = {k: v for k, v in row.items() if k not in {'role', 'content', 'timestamp'}}
    assert len(row['content']) + (len(json.dumps(metadata, ensure_ascii=False)) if metadata else 0) <= budget
    assert 'private payload' not in json.dumps(result)
    if budget == 12000:
        assert row['content'] == prose
        assert row['tool_calls'] == [{'name': 'write_file', 'target': '/project/DESIGN.md'}, {'name': 'broken'}]
        assert not result['truncated']
    else:
        assert result['truncated']


def test_many_large_tool_calls_stay_bounded(history):
    db, store = history
    db.append_message('child', 'assistant', '', tool_calls=[{
        'function': {'name': 'terminal', 'arguments': {'command': 'echo ' + 'x' * 10000}}}
        for _ in range(100)])
    result = store.observe('parent', 'child', max_chars=1000)
    row = result['messages'][0]
    assert 0 < len(row['tool_calls']) < 100
    assert len(json.dumps({'tool_calls': row['tool_calls']}, ensure_ascii=False)) <= 1000
    assert all(len(call['command_preview']) <= 200 for call in row['tool_calls'])
    assert result['truncated']


@pytest.mark.parametrize('redaction_enabled', [True, False])
def test_observe_uses_completion_redaction_before_trimming(history, monkeypatch, redaction_enabled):
    """Assistant/tool prose and preview metadata share the completion boundary (P2-5)."""
    from agent.redact import register_vault_redaction_value, clear_vault_redaction_values
    from tools.delegation_redaction import redact_child_text

    db, store = history
    token = 'ghp_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8'
    vault = 'synthetic-vault-value-for-observation'
    monkeypatch.setattr('agent.redact._redact_enabled', lambda: redaction_enabled)
    register_vault_redaction_value(vault)
    try:
        text = f'Found {token} and {vault}'
        command = f'echo {vault} {token}'
        db.append_message('child', 'assistant', text, tool_calls=[{
            'id': 'call', 'type': 'function', 'function': {'name': 'terminal',
                'arguments': json.dumps({'command': command, 'path': vault})}}])
        db.append_message('child', 'tool', text + '\n' + 'padding ' * 1000,
                          tool_name='terminal', tool_call_id='call')
        before = db.get_messages('child')
        result = store.observe('parent', 'child')
        call, output = result['messages']
        assert call['content'] == redact_child_text(text, force=False)
        assert call['tool_calls'][0]['command_preview'] == redact_child_text(command, force=False)
        assert call['tool_calls'][0]['target'] == redact_child_text(vault, force=False)
        assert output['content'].startswith(redact_child_text(text, force=False))
        assert vault not in json.dumps(result)
        assert (token in json.dumps(result)) is (not redaction_enabled)
        assert db.get_messages('child') == before
    finally:
        clear_vault_redaction_values()
