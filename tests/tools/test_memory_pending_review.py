"""Read-only native memory pending-review contract."""
import copy
import json

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa
from tools.memory_tool import load_on_disk_store, memory_tool
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin


def test_background_replace_full_review(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('memory:\n  write_approval: false\n  user_char_limit: 1375\n')
    store = load_on_disk_store()
    old = 'Existing preference: ' + 'x' * 1350
    assert store.add('user', old)['success']
    token = set_current_write_origin('background_review')
    try:
        staged = json.loads(memory_tool(action='replace', target='user', old_text='Existing preference',
            content=old + ' NEW DETAIL: consent before extending a timed task.', store=store))
    finally:
        reset_current_write_origin(token)
    assert staged['staged']
    pid = staged['pending_id']
    state = copy.deepcopy(vars(store))
    record = wa.get_pending(wa.MEMORY, pid)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    out = handle_pending_subcommand(wa.MEMORY, ['diff', pid], memory_store=store)
    assert vars(store) == state
    assert wa.get_pending(wa.MEMORY, pid) == record
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert isinstance(out, str) and old in out and 'NEW DETAIL' in out
    assert 'exceeds the configured limit' in out
    assert '/memory diff <id>' in handle_pending_subcommand(wa.MEMORY, ['pending'])


def test_projection_uses_disk_not_stale_live_entries(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = load_on_disk_store()
    assert store.add('memory', 'original full entry')['success']
    store.memory_entries = ['stale local entry']
    record = wa.stage_write(wa.MEMORY, {'action': 'remove', 'target': 'memory', 'old_text': 'original'},
                            summary='remove', origin='foreground')
    out = handle_pending_subcommand(wa.MEMORY, ['diff', record['id']], memory_store=store)
    assert '-original full entry' in out
    assert store.memory_entries == ['stale local entry']
    assert load_on_disk_store().memory_entries == ['original full entry']


def test_batch_projection_and_noop_preserve_caller_limit(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = load_on_disk_store()
    store.memory_char_limit = 10
    assert store.add('memory', 'alpha')['success']
    record = wa.stage_write(wa.MEMORY, {'action': 'batch', 'operations': [
        {'action': 'add', 'new_text': 'replacement'}, {'action': 'remove', 'old_text': 'alpha'}]},
        summary='batch', origin='foreground')
    out = handle_pending_subcommand(wa.MEMORY, ['diff', record['id']], memory_store=store)
    assert '+replacement' in out and '-alpha' in out
    assert '/ 10' in out and 'exceeds' in out
    duplicate = wa.stage_write(wa.MEMORY, {'action': 'add', 'content': 'alpha'}, summary='duplicate', origin='foreground')
    assert 'No text changes.' in handle_pending_subcommand(wa.MEMORY, ['diff', duplicate['id']], memory_store=store)


def test_unprojectable_batch_shows_complete_payload_not_partial_diff(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = load_on_disk_store()
    assert store.add('memory', 'first entry')['success']
    assert store.add('memory', 'second entry')['success']
    record = wa.stage_write(wa.MEMORY, {'action': 'batch', 'operations': [
        {'action': 'add', 'content': 'uncommitted'}, {'action': 'remove', 'old_text': 'entry'}]},
        summary='ambiguous', origin='foreground')
    out = handle_pending_subcommand(wa.MEMORY, ['diff', record['id']], memory_store=store)
    assert 'multiple distinct entries' in out
    assert 'uncommitted' in out and 'Full staged proposal' in out
    assert '--- current/' not in out
    assert store._consolidation_failures == 0


def test_missing_and_malformed_records_are_reviewable(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    assert 'Usage:' in handle_pending_subcommand(wa.MEMORY, ['diff'])
    assert 'No pending memory write' in handle_pending_subcommand(wa.MEMORY, ['diff', 'deadbeef'])
    store = load_on_disk_store()
    for payload in [{'action': 'batch', 'operations': []}, {'action': 'batch', 'operations': [42]},
                    {'action': 'remove', 'old_text': 'missing'}, {'action': 'unknown'},
                    {'action': 'add', 'content': 42}]:
        record = wa.stage_write(wa.MEMORY, payload, summary='bad proposal', origin='foreground')
        out = handle_pending_subcommand(wa.MEMORY, ['diff', record['id']], memory_store=store)
        assert 'Cannot project' in out and 'Full staged proposal' in out
        assert wa.get_pending(wa.MEMORY, record['id']) == record


def test_read_exception_preserves_payload(tmp_path, monkeypatch):
    from tools.memory_tool import MemoryStore
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = load_on_disk_store()
    record = wa.stage_write(wa.MEMORY, {'action': 'add', 'content': 'visible proposal'},
                            summary='add', origin='foreground')
    def unavailable(path):
        raise OSError('read denied')
    monkeypatch.setattr(MemoryStore, '_read_raw_checked', staticmethod(unavailable))
    out = handle_pending_subcommand(wa.MEMORY, ['diff', record['id']], memory_store=store)
    assert 'read denied' in out and 'visible proposal' in out


def test_no_live_store_honors_config_and_empty_file(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('memory:\n  user_char_limit: 19\n')
    record = wa.stage_write(wa.MEMORY, {'action': 'add', 'target': 'user', 'content': 'first preference'},
                            summary='new', origin='foreground')
    out = handle_pending_subcommand(wa.MEMORY, ['diff', record['id']])
    assert '+first preference' in out and '/ 19' in out
    assert load_on_disk_store().user_entries == []
