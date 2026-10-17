"""Capacity recovery uses valid, independent proposals from the owning file."""
import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import write_approval as wa
from tools.memory_tool import load_on_disk_store, memory_tool
from tools.memory_space_plan import space_plan
from utils import atomic_json_write


def test_capacity_plans_and_producer_recovery_are_scoped_and_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(wa, '_prompt_for_record', lambda *_: None)
    homes = [tmp_path / name for name in ('alpha', 'beta')]
    for home in homes:
        home.mkdir()
        (home / 'config.yaml').write_text(json.dumps({'memory': {
            'write_approval': True, 'user_char_limit': 50, 'memory_char_limit': 50}}))
    records = {}
    for home in (homes[0], homes[1], homes[0]):
        token = set_hermes_home_override(home)
        try:
            store = load_on_disk_store()
            if home not in records:
                original = 'A useful policy with details.'
                assert store.add('user', original)['success']
                assert store.add('user', 'Other facts')['success']
                before = store._path_for('user').read_bytes()
                result = json.loads(memory_tool(action='add', target='user', content='New facts for later', store=store))
                record = wa.get_pending('memory', result['pending_id'])
                records[home] = record
                recovery, = result['capacity_recovery']
                assert recovery['pending_id'] == record['id']
                assert recovery['target_path'] == str(home / 'memories/USER.md')
                assert recovery['characters_needed'] == 15 and recovery['companion_ids'] == []
                assert recovery['current_entries'] == store.user_entries
                assert 'independent full-text approval' in recovery['instruction']
                assert store._path_for('user').read_bytes() == before
            record = records[home]
            if home == homes[0]:
                companion = wa.stage_write('memory', {'action': 'replace', 'target': 'user',
                    'old_text': 'A useful policy with details.', 'matched_entry': 'A useful policy with details.',
                    'content': 'Useful policy'}, summary='make room', origin='pytest')
                plan = space_plan(record, store)
                assert plan['after'] == 49 and plan['steps'][0]['id'] == companion['id']
                assert plan['steps'][0]['saved_chars'] == 16
                # Snoozed, stale and wrong-file candidates cannot promise usable space.
                bad = [dict(companion, id='11111111', snooze_until=100),
                       dict(companion, id='22222222', payload={**companion['payload'], 'matched_entry': 'stale'}),
                       dict(companion, id='33333333', payload={**companion['payload'], 'target': 'memory'})]
                assert space_plan(record, store, bad, now=10)['steps'] == []
                remove = dict(companion, id='44444444', payload={'action': 'remove', 'target': 'user',
                    'old_text': 'A useful policy with details.', 'matched_entry': 'A useful policy with details.'})
                assert space_plan(record, store, [remove])['steps'][0]['saved_chars'] == 32
                blocked_replace = dict(record, payload={'action': 'replace', 'target': 'user',
                    'matched_entry': 'A useful policy with details.', 'old_text': 'A useful policy with details.',
                    'content': 'x' * 45})
                assert space_plan(blocked_replace, store, [companion])['steps'] == []
                # Clean up only the synthetic companion so A→B→A repeats the invariant.
                wa.discard_pending('memory', companion['id'])
            else:
                assert space_plan(record, store)['steps'] == []
        finally:
            reset_hermes_home_override(token)
