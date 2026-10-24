"""Independent memory decisions, including capacity and entry dependencies."""
import json

from tools import write_approval as wa
from tools.memory_tool import load_on_disk_store, memory_tool
from hermes_cli.write_approval_commands import handle_pending_subcommand


def setup_home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('memory:\n  write_approval: true\n')
    sent = []
    monkeypatch.setattr(wa, '_prompt_for_record', lambda _, rec: sent.append(rec['id']))
    return sent


def test_eight_operations_are_independent(tmp_path, monkeypatch):
    sent = setup_home(tmp_path, monkeypatch)
    store = load_on_disk_store()
    for i in range(8):
        store.add('memory', f'Original {i}')
    result = json.loads(memory_tool(action='batch', target='memory', store=store, operations=[
        {'action': 'patch', 'old_text': f'Original {i}', 'content': f'Revised {i}'} for i in range(8)]))
    assert result['success'] and len(result['pending_ids']) == 8
    assert 'pending_id' not in result
    assert sent == result['pending_ids']
    records = wa.list_pending('memory')
    assert all(r['payload']['action'] == 'replace' and r['revision'] == 1 for r in records)
    handle_pending_subcommand('memory', ['approve', records[5]['id']], memory_store=store)
    assert store.memory_entries == [f'Revised {i}' if i == 5 else f'Original {i}' for i in range(8)]
    handle_pending_subcommand('memory', ['reject', records[2]['id']])
    assert len(wa.list_pending('memory')) == 6
    again = wa.stage_write('memory', {'action': 'batch', 'target': 'memory', 'operations': [
        records[2]['payload'], records[3]['payload']]}, summary='again', origin='pytest')
    assert len(again['pending_ids']) == 1 and len(again['suppressed']) == 1


def test_capacity_and_sequential_entry_dependency(tmp_path, monkeypatch):
    setup_home(tmp_path, monkeypatch)
    from tools.memory_review_records import review_issue
    store = load_on_disk_store()
    store.memory_char_limit = 40
    store.add('memory', 'x' * 30)
    result = json.loads(memory_tool(action='batch', target='memory', store=store, operations=[
        {'action': 'add', 'content': 'new entry'},
        {'action': 'patch', 'old_text': 'x' * 30, 'content': 'short'},
        {'action': 'patch', 'old_text': 'short', 'content': 'final'}]))
    children = [wa.get_pending('memory', pid) for pid in result['pending_ids']]
    assert 'space' in review_issue(children[0], store).lower()
    assert children[2]['depends_on'] == [children[1]['id']]
    assert 'earlier proposal' in review_issue(children[2], store)
    failed = handle_pending_subcommand('memory', ['approve', children[2]['id']], memory_store=store)
    assert 'Approved 0' in failed and store.memory_entries == ['x' * 30]
    handle_pending_subcommand('memory', ['approve', children[1]['id']], memory_store=store)
    assert review_issue(children[2], store) is None
    handle_pending_subcommand('memory', ['approve', children[2]['id']], memory_store=store)
    handle_pending_subcommand('memory', ['approve', children[0]['id']], memory_store=store)
    assert store.memory_entries == ['final', 'new entry']


def test_legacy_aggregate_cannot_be_approved(tmp_path, monkeypatch):
    setup_home(tmp_path, monkeypatch)
    from utils import atomic_json_write
    record = {'id': 'abcdef12', 'payload': {'action': 'batch', 'target': 'memory', 'operations': [
        {'action': 'add', 'content': 'one'}, {'action': 'add', 'content': 'two'}]}}
    atomic_json_write(wa._pending_path('memory', record['id']), record)
    store = load_on_disk_store()
    result = handle_pending_subcommand('memory', ['approve', record['id']], memory_store=store)
    assert 'individually' in result
    assert not store.memory_entries and wa.get_pending('memory', record['id'])


def test_disk_failure_is_not_staged_success(tmp_path, monkeypatch):
    setup_home(tmp_path, monkeypatch)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(wa, 'atomic_json_write', fail)
    result = json.loads(memory_tool(action='add', target='memory', content='new', store=load_on_disk_store()))
    assert result['success'] is False and not result.get('staged') and 'pending_id' not in result


def test_span_patch_review_preserves_other_facts_and_refuses_drift(tmp_path, monkeypatch):
    setup_home(tmp_path, monkeypatch)
    store = load_on_disk_store()
    original = 'Drink tea daily. Keep the controller last.'
    assert store.add('user', original)['success']
    staged = json.loads(memory_tool(action='patch', target='user', old_text='tea',
                                   content='coffee', store=store))
    record = wa.get_pending('memory', staged['pending_id'])
    assert record['payload']['matched_entry'] == original
    assert record['payload']['content'] == 'Drink coffee daily. Keep the controller last.'
    assert store.user_entries == [original]
    # Approval applies exactly the complete reviewed entry, including untouched facts.
    approved = handle_pending_subcommand('memory', ['approve', record['id']], memory_store=store)
    assert 'Approved 1' in approved
    assert store.user_entries == ['Drink coffee daily. Keep the controller last.']
    again = json.loads(memory_tool(action='patch', target='user', old_text='coffee',
                                  content='water', store=store))
    assert store.patch('user', 'daily', 'weekly')['success']
    refused = handle_pending_subcommand('memory', ['approve', again['pending_id']], memory_store=store)
    assert 'Approved 0' in refused
    assert store.user_entries == ['Drink coffee weekly. Keep the controller last.']
    assert wa.get_pending('memory', again['pending_id']) is not None


def test_legacy_pinned_patch_batch_converts_to_full_entry_reviews(tmp_path, monkeypatch):
    setup_home(tmp_path, monkeypatch)
    from tools.memory_review_records import independent_records
    original = 'Drink tea daily. Keep the controller last.'
    records = independent_records({'action': 'batch', 'target': 'user', 'operations': [
        {'action': 'patch', 'matched_entry': original, 'old_text': 'tea', 'content': 'coffee'},
        {'action': 'add', 'content': 'Another fact.'}]}, summary='legacy patch', origin='foreground')
    assert records[0]['payload']['action'] == 'replace'
    assert records[0]['payload']['matched_entry'] == original
    assert records[0]['payload']['content'] == 'Drink coffee daily. Keep the controller last.'
    assert records[1]['payload']['action'] == 'add'
