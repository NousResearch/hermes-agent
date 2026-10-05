"""Persistent memory identity and fail-closed management contracts for #86157."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.memory_tool import ENTRY_DELIMITER, MemoryStore, apply_memory_pending, load_on_disk_store
from tools.registry import registry


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from agent import secret_scope
    root = tmp_path / '.hermes'
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    for home in (root, root / 'profiles' / 'a', root / 'profiles' / 'b', root / 'profiles' / 'launch'):
        _config(home)
    monkeypatch.setenv('HERMES_HOME', str(root))
    secret_scope.set_multiplex_active(True)
    try:
        yield root, root / 'profiles' / 'a', root / 'profiles' / 'b'
    finally:
        secret_scope.set_multiplex_active(False)


def _config(home, approval=False):
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'memory': {'memory_char_limit': 40000,
        'user_char_limit': 40000, 'write_approval': approval}}), encoding='utf-8')


@contextmanager
def _served(home):
    from agent import secret_scope
    token = set_hermes_home_override(home)
    secrets = secret_scope.set_secret_scope({})
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secrets)
        reset_hermes_home_override(token)


def _call(route, store, **args):
    if route == 'registry':
        result = registry.dispatch('memory', args, store=store)
    else:
        from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
        agent = SimpleNamespace(_memory_store=store, _memory_manager=None)
        result = INLINE_TOOL_EXECUTORS['memory'](agent, args, InlineToolContext(effective_task_id='memory-review'))
    return json.loads(result)


def _migrate(*flags):
    from hermes_cli.main_agent_cmds import cmd_memory
    from hermes_cli.subcommands.memory import build_memory_parser
    parser = argparse.ArgumentParser()
    build_memory_parser(parser.add_subparsers(), cmd_memory=cmd_memory)
    args = parser.parse_args(['memory', 'migrate-identities', *flags])
    return args.func(args)


def _seed(home):
    directory = home / 'memories'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'MEMORY.md').write_text(f'First {home.name}\n§\nShared cue\n§\nLong note ' + 'z' * 600,
                                       encoding='utf-8-sig')
    (directory / 'USER.md').write_bytes(b'Prefers clear examples.\r\n')


def _list(route, store, target='memory', **kwargs):
    result = _call(route, store, action='list', target=target, **kwargs)
    assert result['success'], result
    return result


def _all(route, store, target='memory'):
    entries, cursor = [], None
    while True:
        page = _list(route, store, target, limit=2, cursor=cursor)
        assert len(page['entries']) <= 2
        entries.extend(page['entries'])
        cursor = page['next_cursor']
        assert page['truncated'] == (cursor is not None)
        if cursor is None:
            assert len(entries) == page['total']
            return entries


def _staged_add(route, store, home):
    from tools import write_approval as wa
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    _config(home, approval=True)
    token = set_current_write_origin('background_review')
    try:
        staged = _call(route, store, action='add', target='user', content='Staged preference.', scope='settings')
    finally:
        reset_current_write_origin(token)
        _config(home)
    assert staged['staged'], staged
    payload = wa.get_pending(wa.MEMORY, staged['pending_id'])['payload']
    assert payload['scope'] == 'settings'
    assert 'Staged preference.' not in store.user_entries
    applied = apply_memory_pending(payload, store)
    assert applied['success'] and applied['scope'] == 'settings'
    assert UUID(applied['entry_id']).version == 4
    _config(home, approval=True)
    token = set_current_write_origin('background_review')
    try:
        staged = _call(route, store, target='user', operations=[
            {'action': 'add', 'content': 'Batch preference.', 'scope': 'batch-settings'},
            {'action': 'replace', 'old_text': 'Batch preference.', 'new_text': 'Reviewed batch preference.'},
        ])
    finally:
        reset_current_write_origin(token)
        _config(home)
    assert staged['staged'], staged
    payload = wa.get_pending(wa.MEMORY, staged['pending_id'])['payload']
    planned = payload['operations'][0]['created_entry_id']
    assert payload['operations'][1]['entry_id'] == planned
    applied = apply_memory_pending(payload, store)
    assert applied['success'] and applied['entry_ids'][2]['entry_id'] == planned
    assert _list(route, store, 'user', scope='batch-settings')['entries'][0]['content'] == 'Reviewed batch preference.'


def _exercise_identity(route, store, home):
    from agent.learning_graph import build_learning_graph
    from agent.learning_mutations import delete_node, edit_node, node_detail
    initial = _all(route, store)
    assert any(entry['content_truncated'] and len(entry['content']) == 512 for entry in initial)
    scoped = _call(route, store, action='add', content='Identical preference.', scope='settings')
    other = _call(route, store, action='add', content='Identical preference.', scope='other')
    assert scoped['success'] and other['success'] and scoped['entry_id'] != other['entry_id']
    assert _call(route, store, action='add', content='Identical preference.', scope='settings')['entry_id'] == scoped['entry_id']
    assert {entry['entry_id'] for entry in _list(route, store, scope='settings')['entries']} == {scoped['entry_id']}
    entry_id = scoped['entry_id']
    result = _call(route, store, action='replace', entry_id=entry_id, scope='settings', new_text='Corrected preference.')
    assert result['success'] and result['entry_id'] == entry_id
    batch = _call(route, store, operations=[
        {'action': 'replace', 'entry_id': other['entry_id'], 'scope': 'other', 'content': 'Another preference.'},
        {'action': 'add', 'new_text': 'Batched note.'},
    ])
    assert batch['success'] and batch['entry_ids']['1']['entry_id'] == other['entry_id']
    assert UUID(batch['entry_ids']['2']['entry_id']).version == 4
    # Existing substring and batch callers must preserve the same identity.
    assert _call(route, store, action='replace', old_text='Corrected preference.', content='Reviewed preference.')['entry_id'] == entry_id
    assert _call(route, store, operations=[{'action': 'replace', 'old_text': 'Reviewed preference.',
                                          'content': 'Reviewed preference again.'}])['entry_ids']['1']['entry_id'] == entry_id
    node_id = next(node['id'] for node in build_learning_graph()['nodes'] if node['id'].endswith(entry_id))
    path = home / 'memories' / 'MEMORY.md'
    header, _, body = path.read_text(encoding='utf-8').partition('\n')
    path.write_text(header + '\n' + ENTRY_DELIMITER.join(reversed(body.split(ENTRY_DELIMITER))), encoding='utf-8')
    restarted = load_on_disk_store()
    assert next(node['id'] for node in build_learning_graph()['nodes'] if node['id'].endswith(entry_id)) == node_id
    assert node_detail(node_id)['content'] == 'Reviewed preference again.'
    assert edit_node(node_id, 'Edited through Journey.')['ok']
    assert node_detail(node_id)['content'] == 'Edited through Journey.'
    assert entry_id in {entry['entry_id'] for entry in _all(route, restarted)}
    assert delete_node(f"memory:memory:{other['entry_id']}")['ok']
    assert node_detail(node_id)['ok']
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(copy_context().run, _call, route, store,
                                   action='add', content=f'Concurrent note {index}.') for index in range(8)]
        results = [future.result(timeout=15) for future in futures]
    assert all(result['success'] for result in results)
    assert len({result['entry_id'] for result in results}) == 8
    assert node_detail(node_id)['content'] == 'Edited through Journey.'
    _verify_memory_import(route, store, home)
    _verify_review_race(route, store, home)
    ids = {entry['entry_id'] for entry in _all(route, load_on_disk_store())}
    assert ids.issuperset(result['entry_id'] for result in results)
    assert all(UUID(value).version == 4 for value in ids)
    return ids


def _assert_identity_survives(homes, tmp_path, monkeypatch, capsys, route, launch_named):
    root, a, b = homes
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles' / 'launch' if launch_named else root))
    expected = {}
    for home in (a, b, a):
        with _served(home):
            store = load_on_disk_store()
            if home in expected:
                assert {entry['entry_id'] for entry in _all(route, store)} == expected[home]
                continue
            _seed(home)
            store.load_from_disk()
            frozen = store.format_for_system_prompt('memory')
            unsupported = _call(route, store, action='list')
            assert not unsupported['success']
            paths = [home / 'memories' / name for name in ('MEMORY.md', 'USER.md')]
            original = {path.name: path.read_bytes() for path in paths}
            assert _migrate('--target', 'all', '--dry-run') == 0
            assert all(path.read_bytes() == original[path.name] for path in paths)
            assert not (home / 'memories' / '.identity-backups').exists()
            assert _migrate('--target', 'all', '--yes') == 0
            backups = list((home / 'memories' / '.identity-backups').glob('*.bak'))
            assert len(backups) == 2
            assert all(path.read_bytes() == original[path.name.split('.')[0] + '.md'] for path in backups)
            migrated = {path: path.read_bytes() for path in paths}
            assert _migrate('--target', 'all', '--yes') == 0
            assert all(path.read_bytes() == migrated[path] for path in paths)
            assert len(list((home / 'memories' / '.identity-backups').glob('*.bak'))) == 2
            _staged_add(route, store, home)
            expected[home] = _exercise_identity(route, store, home)
            assert store.format_for_system_prompt('memory') == frozen
            refreshed = load_on_disk_store()
            prompt = refreshed.format_for_system_prompt('memory')
            assert 'Edited through Journey.' in prompt and 'hermes-memory-' not in prompt
            assert 'settings' not in prompt and not any(value in prompt for value in expected[home])
            assert refreshed._char_count('memory') == len(ENTRY_DELIMITER.join(refreshed.memory_entries))
            restored = tmp_path / f'restored-{home.name}'
            shutil.copytree(home, restored)
            with _served(restored):
                assert {entry['entry_id'] for entry in _all(route, load_on_disk_store())} == expected[home]
    assert expected[a].isdisjoint(expected[b])
    assert not (root / 'memories').exists() and not (root / 'profiles' / 'launch' / 'memories').exists()
    capsys.readouterr()


def _stale_pending(store, target, entry, home, case):
    from tools import write_approval as wa
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    token = set_current_write_origin('background_review')
    try:
        kwargs = {'action': 'remove', 'target': target, 'entry_id': entry['entry_id'], 'scope': entry['scope']}
        if case == 'legacy-approval-recreated':
            kwargs = {'action': 'remove', 'target': target, 'old_text': entry['content']}
        if case == 'batch-approval-recreated':
            kwargs = {'target': target, 'operations': [
                {'action': 'remove', 'entry_id': entry['entry_id'], 'scope': entry['scope']},
                {'action': 'add', 'content': 'Must remain a proposal.'},
            ]}
        staged = _call('registry', store, **kwargs)
    finally:
        reset_current_write_origin(token)
    assert staged['staged'], staged
    payload = wa.get_pending(wa.MEMORY, staged['pending_id'])['payload']
    pinned = payload['operations'][0] if 'operations' in payload else payload
    assert pinned['entry_id'] == entry['entry_id'] and pinned['scope'] == entry['scope']
    assert pinned['matched_entry'] == entry['content']
    if case == 'approval-content-changed':
        assert _call('registry', store, action='replace', target=target, entry_id=entry['entry_id'],
                     scope=entry['scope'], content='Changed after review.')['success']
    else:
        assert _call('registry', store, action='remove', target=target, entry_id=entry['entry_id'], scope=entry['scope'])['success']
        recreated = _call('registry', store, action='add', target=target, content=entry['content'], scope=entry['scope'])
        assert recreated['success'] and recreated['entry_id'] != entry['entry_id']
    return lambda: apply_memory_pending(payload, store)


def _corrupt(path, entries, case):
    first, second = entries[0]['entry_id'], entries[1]['entry_id']
    transforms = {
        'duplicate-id': lambda raw: raw.replace(second, first),
        'missing-header': lambda raw: raw.replace(next(line for line in raw.splitlines() if first in line) + '\n', ''),
        'duplicate-json-key': lambda raw: raw.replace(f'"id":"{first}"', f'"id":"{first}","id":"{first}"'),
        'unsupported-version': lambda raw: raw.replace('identities:v1', 'identities:v2'),
        'invalid-metadata-target': lambda raw: raw.replace('"target":"memory"', '"target":[]').replace('"target":"user"', '"target":[]'),
        'wrong-metadata-target': lambda raw: raw.replace('"target":"memory"', '"target":"user"') if '"target":"memory"' in raw else raw.replace('"target":"user"', '"target":"memory"'),
        'removed-all-metadata': lambda raw: ENTRY_DELIMITER.join(MemoryStore._read_file(path)),
        'invalid-encoding': lambda raw: raw + '\udcff',
    }
    if case not in transforms:
        return False
    content = transforms[case](path.read_text(encoding='utf-8'))
    path.write_bytes(content.encode('utf-8', errors='surrogateescape'))
    return True


def _migration_failure(store, home, target, case, monkeypatch):
    from tools import memory_identity_store as identities
    path = home / 'memories' / ('MEMORY.md' if target == 'memory' else 'USER.md')
    path.write_text(ENTRY_DELIMITER.join(MemoryStore._read_file(path)), encoding='utf-8')
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError('Injected memory persistence failure')

    if case == 'migration-backup-failure':
        monkeypatch.setattr(identities, '_backup_bytes', fail)
    else:
        from tools import memory_store_io
        monkeypatch.setattr(memory_store_io, 'atomic_write_text', fail)

    def migrate():
        result = identities.migrate_target(store, target, commit=True)
        if case == 'migration-write-failure':
            assert Path(result['backup']).read_bytes() == original
        return result

    return migrate


def _request(store, home, target, entries, case, monkeypatch):
    entry = entries[0]
    kwargs = {'action': 'remove', 'target': target, 'entry_id': entry['entry_id'], 'scope': entry['scope']}
    changes = {
        'unknown-id': {'entry_id': str(uuid4())},
        'wrong-target': {'target': 'user' if target == 'memory' else 'memory'},
        'wrong-scope': {'scope': 'other'},
        'invalid-id': {'entry_id': '123e4567-e89b-12d3-a456-426614174000'},
        'invalid-scope': {'scope': []},
        'conflicting-selection': {'old_text': entry['content']},
        'reserved-delimiter': {'action': 'replace', 'content': 'two\n§\nentries'},
        'bad-list-limit': {'action': 'list', 'entry_id': None, 'scope': None, 'limit': True},
        'bad-list-scope': {'action': 'list', 'entry_id': None, 'scope': ''},
    }
    if 'approval' in case:
        return _stale_pending(store, target, entry, home, case)
    if case.startswith('migration-'):
        return _migration_failure(store, home, target, case, monkeypatch)
    path = home / 'memories' / ('MEMORY.md' if target == 'memory' else 'USER.md')
    if _corrupt(path, entries, case):
        return lambda: _call('registry', store, **kwargs)
    special = {
        'changed-cursor': lambda: _cursor_request(store, target),
        'caller-creation-id': lambda: {'target': target, 'operations': [
            {'action': 'add', 'content': 'Not caller assigned.', 'created_entry_id': str(uuid4())}]},
        'stale-removed': lambda: _stale_request(store, target, entry),
        'batch-atomic-failure': lambda: {'target': target, 'operations': [
            {'action': 'add', 'content': 'Must not be committed.'},
            {'action': 'remove', 'entry_id': entry['entry_id'], 'scope': 'other'}]},
    }
    kwargs = special[case]() if case in special else {**kwargs, **changes[case]}
    return lambda: _call('registry', store, **kwargs)


def _assert_unverifiable_preserves_disk(homes, capsys, monkeypatch, target, case):
    _, home, _ = homes
    with _served(home):
        _seed(home)
        store = load_on_disk_store()
        unsupported = _call('registry', store, action='list', target=target)
        assert not unsupported['success']
        assert _migrate('--yes') == 0
        assert _call('registry', store, action='add', target=target, content='Scoped preference.', scope='settings')['success']
        assert _call('registry', store, action='add', target=target, content='Other entry.', scope='other')['success']
        entries = sorted(_all('registry', store, target), key=lambda entry: entry['scope'] != 'settings')
        request = _request(store, home, target, entries, case, monkeypatch)
        paths = list((home / 'memories').glob('*.md'))
        before = {path: path.read_bytes() for path in paths}
        result = request()
        assert not result['success'], result
        assert all(path.read_bytes() == before[path] for path in paths)
        assert 'error' in result
    capsys.readouterr()


def _cursor_request(store, target):
    cursor = _list('registry', store, target, limit=1)['next_cursor']
    assert _call('registry', store, action='add', target=target, content='Concurrent new entry.')['success']
    return {'action': 'list', 'target': target, 'cursor': cursor, 'limit': 1}


def _stale_request(store, target, entry):
    kwargs = {'action': 'remove', 'target': target, 'entry_id': entry['entry_id'], 'scope': entry['scope']}
    assert _call('registry', store, **kwargs)['success']
    recreated = _call('registry', store, action='add', target=target, content=entry['content'], scope=entry['scope'])
    assert recreated['entry_id'] != entry['entry_id']
    return kwargs


_INVALID_CASES = [
    'unknown-id', 'wrong-target', 'wrong-scope', 'invalid-id', 'invalid-scope', 'conflicting-selection',
    'reserved-delimiter', 'bad-list-limit', 'bad-list-scope', 'changed-cursor', 'batch-atomic-failure',
    'duplicate-id', 'missing-header', 'duplicate-json-key', 'unsupported-version', 'invalid-metadata-target',
    'wrong-metadata-target', 'removed-all-metadata', 'invalid-encoding', 'approval-content-changed',
    'approval-recreated', 'legacy-approval-recreated', 'batch-approval-recreated',
    'caller-creation-id', 'stale-removed',
    'migration-backup-failure', 'migration-write-failure',
]
_CONTRACTS = [pytest.param('lifetime', route, named, None, None, id=f'{route}-launch-named-{named}')
              for route in ('registry', 'inline') for named in (False, True)]
_CONTRACTS += [pytest.param('rejected', 'registry', False, target, case, id=f'{target}-{case}')
               for target in ('memory', 'user') for case in _INVALID_CASES]


@pytest.mark.parametrize('contract,route,launch_named,target,case', _CONTRACTS)
def test_memory_management_never_redirects_identity(homes, tmp_path, monkeypatch, capsys, contract,
                                                    route, launch_named, target, case):
    if contract == 'lifetime':
        _assert_identity_survives(homes, tmp_path, monkeypatch, capsys, route, launch_named)
    else:
        _assert_unverifiable_preserves_disk(homes, capsys, monkeypatch, target, case)


def _verify_memory_import(route, store, home):
    from hermes_cli.agent_import import AgentImporter, SUPPORTED_AGENTS
    source = home / "import-notes.md"
    source.write_text("Imported standing note.\n\nAnother imported note.", encoding="utf-8")
    path = home / "memories" / "MEMORY.md"
    before = path.read_bytes()
    identities = {entry["entry_id"] for entry in _all(route, store)}
    preview = AgentImporter(SUPPORTED_AGENTS[0], source.parent, home)
    preview.import_context_file(source, "memory-notes")
    assert preview.items[0]["status"] == "imported" and path.read_bytes() == before
    actual = AgentImporter(SUPPORTED_AGENTS[0], source.parent, home, execute=True)
    actual.import_context_file(source, "memory-notes")
    assert actual.items[0]["status"] == "imported"
    current = _all(route, store)
    assert {entry["entry_id"] for entry in current}.issuperset(identities)
    assert {"Imported standing note.", "Another imported note."}.issubset(entry["content"] for entry in current)
    assert Path(actual.items[0]["backup"]).read_bytes() == before
    valid = path.read_bytes()
    identifier = current[0]["entry_id"]
    path.write_text(valid.decode().replace(f'"id":"{identifier}"',
                    f'"id":"{identifier}","id":"{identifier}"'), encoding="utf-8")
    broken = path.read_bytes()
    actual.import_context_file(source, "memory-notes")
    assert actual.items[-1]["status"] == "error" and path.read_bytes() == broken
    path.write_bytes(valid)


def _verify_review_race(route, store, home):
    from unittest.mock import patch
    from tools import write_approval as wa
    from tools.memory_identity_store import edit_entry_id

    for target in ('memory', 'user'):
        for allow in (True, False):
            created = _call(route, store, action='add', target=target,
                            content='Preference before review.', scope='review-race')
            assert created['success'], created
            path = home / 'memories' / ('MEMORY.md' if target == 'memory' else 'USER.md')
            changed = []

            def decide(*args, **kwargs):
                assert edit_entry_id(store, 'replace', target, created['entry_id'],
                    scope='review-race', content='Preference changed during review.')['success']
                changed.append(path.read_bytes())
                return SimpleNamespace(allow=allow, blocked=False, message='Approval required.')

            with patch.object(wa, 'evaluate_gate', decide):
                result = _call(route, store, action='remove', target=target,
                               entry_id=created['entry_id'], scope='review-race')
            assert not result['success'], result
            assert changed and path.read_bytes() == changed[0]
            assert 'changed since it was reviewed' in result['error']
