"""Conversion journals retain exact proposals through crashes and decisions."""
import json

import pytest
import gateway.run # bootstrap before isolated-home guards

from tools import write_approval as wa
from tools import memory_review_conversion as conversion
from tools.memory_tool import load_on_disk_store
from hermes_cli.write_approval_commands import handle_pending_subcommand
from utils import atomic_json_write


def seed(tmp_path, monkeypatch, count=8):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    record = dict(id='abcdef12', subsystem='memory', origin='background_review', created_at=123,
                  snooze_until=9999999999, payload=dict(action='batch', target='user', operations=[
                      dict(action='add', content=f'Memory {i} 😀\ncomplete text {i}') for i in range(count)]))
    atomic_json_write(wa._pending_path('memory', record['id']), record)
    return record


def test_conversion_preserves_every_byte_owner_clock_and_fixed_ids(tmp_path, monkeypatch):
    original = seed(tmp_path, monkeypatch)
    journals = conversion.convert_pending_batches()
    assert len(journals) == 1 and journals[0]['original'] == original
    records = wa.list_pending('memory')
    assert len(records) == 8 and not wa.get_pending('memory', original['id'])
    assert [r['payload']['content'] for r in records] == [op['content'] for op in original['payload']['operations']]
    assert all(r['snooze_until'] == original['snooze_until'] and r['created_at'] == 123 for r in records)
    ids = {r['id'] for r in records}
    conversion.convert_pending_batches()
    assert {r['id'] for r in wa.list_pending('memory')} == ids
    assert load_on_disk_store().user_entries == []


def test_crash_between_child_writes_recovers_without_duplicate(tmp_path, monkeypatch):
    original = seed(tmp_path, monkeypatch)
    real_write = conversion.atomic_json_write
    writes = 0
    def crash(path, value):
        nonlocal writes
        writes += 1
        if writes == 4:
            raise OSError('simulated disk failure')
        return real_write(path, value)
    monkeypatch.setattr(conversion, 'atomic_json_write', crash)
    with pytest.raises(OSError):
        conversion.convert_pending_batches()
    prepared = wa.list_pending('memory')
    child = next(r for r in prepared if r['id'] != original['id'])
    result = handle_pending_subcommand('memory', ['approve', child['id']], memory_store=load_on_disk_store())
    assert 'Approved 0' in result
    monkeypatch.setattr(conversion, 'atomic_json_write', real_write)
    conversion.convert_pending_batches()
    assert len(wa.list_pending('memory')) == 8
    assert child['id'] in {r['id'] for r in wa.list_pending('memory')}


def test_crash_after_child_decision_never_resurrects_child(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch, count=2)
    conversion.convert_pending_batches()
    path = conversion.journal_dir() / 'abcdef12.json'
    journal = json.loads(path.read_text())
    # Simulate a crash after children became actionable but before the final journal write.
    journal['state'] = 'children_written'
    atomic_json_write(path, journal)
    first, second = wa.list_pending('memory')
    handle_pending_subcommand('memory', ['reject', first['id']])
    conversion.convert_pending_batches()
    assert [r['id'] for r in wa.list_pending('memory')] == [second['id']]
    assert json.loads(path.read_text())['original']['payload']['operations'][0] == journal['original']['payload']['operations'][0]


def test_retire_combined_controls_once_without_erasing_comparison(tmp_path, monkeypatch):
    from tests.fakes.platforms.telegram_standin import TelegramStandin
    from tests.tools.test_memory_prompt_cadence import configure
    from tools.memory_prompt_cadence import tick
    original = seed(tmp_path, monkeypatch, count=2)
    original.pop('snooze_until')
    with TelegramStandin() as server:
        configure(tmp_path, server)
        # Stand-in accepts edits of the old known message; conversion doesn't approve its contents.
        original['prompt'] = dict(chat_id='111', bot_id='123456789', thread_id=None, message_ids=['20'])
        atomic_json_write(wa._pending_path('memory', original['id']), original)
        assert tick() == 2
        assert len(server.calls_of('editMessageReplyMarkup')) == 1
        assert not server.calls_of('editMessageText')
        assert len(server.calls_of('sendMessage')) == 3 # replacement notice + two independent cards
        assert tick() == 0
        assert len(server.calls_of('sendMessage')) == 3
        assert load_on_disk_store().user_entries == []
