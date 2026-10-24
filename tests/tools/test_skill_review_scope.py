"""Real profile switches, external creation paths and durable retry isolation."""
import json
import time
from pathlib import Path

import gateway.run
import pytest
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tests.fakes.platforms.telegram_standin import TelegramStandin
from tests.tools.test_skill_review import ORIGINAL, REPLACEMENT
from tools import skill_manager_tool as sm, skill_review as review, write_approval as wa


def configure(home, **skills):
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'skills': skills}))
    from agent import skill_utils
    skill_utils._raw_config_cache_clear()
    skill_utils._external_dirs_cache_clear()


def create():
    assert json.loads(sm.skill_manage('create', 'sample', content=ORIGINAL))['success']


def stage():
    result = json.loads(sm.skill_manage('', '', operations=[
        {'action': 'patch', 'name': 'sample', 'old_string': 'original rule', 'new_string': 'revised rule'}]))
    assert result['staged'], result
    return review.inspect(result['pending_id'])


@pytest.mark.parametrize('kind', ['local', 'external', 'shared_create', 'creation_drift', 'delete_added_file'])
def test_a_b_a_and_existing_skill_resolution_are_preserved(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    homes = [tmp_path / '.hermes', tmp_path / '.hermes/profiles/b']
    monkeypatch.setenv('HERMES_HOME', str(homes[0]))
    external = tmp_path / 'shared'
    settings = {'create_dir': str(external)} if kind == 'shared_create' else {}
    configure(homes[0], **settings)
    if kind != 'creation_drift':
        create()
    if kind == 'external':
        external.mkdir()
        (homes[0] / 'skills/sample').rename(external / 'sample')
        settings = {'external_dirs': [str(external)]}
    configure(homes[0], **settings, write_approval=True)
    if kind == 'creation_drift':
        result = json.loads(sm.skill_manage('create', 'sample', content=ORIGINAL))
        record = review.inspect(result['pending_id'])
        configure(homes[0], create_dir=str(external), write_approval=True)
        with pytest.raises(ValueError, match='destination changed'):
            review.decide(record['id'], 1, 0, 'approve')
        assert not (external / 'sample').exists()
        return
    if kind == 'delete_added_file':
        result = json.loads(sm.skill_manage('delete', 'sample'))
        record = review.inspect(result['pending_id'])
        (homes[0] / 'skills/sample/extra.txt').write_text('Not reviewed')
        with pytest.raises(ValueError, match='files in skill'):
            review.decide(record['id'], 1, 0, 'approve')
        assert (homes[0] / 'skills/sample/extra.txt').exists()
        return
    first = stage()
    expected_root = external / 'sample' if kind != 'local' else homes[0] / 'skills/sample'
    assert first['skill_review']['files'][0]['path'] == str(expected_root / 'SKILL.md')
    configure(homes[1])
    token = set_hermes_home_override(homes[1])
    try:
        create()
        configure(homes[1], write_approval=True)
        second = stage()
        assert second['skill_review']['files'][0]['path'] == str(homes[1] / 'skills/sample/SKILL.md')
        assert wa.get_pending('skills', first['id']) is None
        review.decide(second['id'], 1, 0, 'approve')
    finally:
        reset_hermes_home_override(token)
    assert review.inspect(first['id'])['skill_review']['files'][0]['before'] == ORIGINAL
    review.decide(first['id'], 1, 0, 'approve')
    assert (expected_root / 'SKILL.md').read_text() == REPLACEMENT
    assert (homes[1] / 'skills/sample/SKILL.md').read_text() == REPLACEMENT


@pytest.mark.parametrize('kind', ['retry', 'legacy', 'disabled', 'snoozed'])
def test_scheduler_respects_scope_snooze_and_legacy_records(tmp_path, monkeypatch, kind):
    from tools.memory_prompt_cadence import tick
    from tools.skill_prompt_send import send_prompt
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    monkeypatch.setenv('HERMES_HOME', str(home))
    configure(home)
    create()
    configure(home, write_approval=True)
    record = stage()  # No configured transport: a real missed immediate prompt.
    assert not record.get('last_prompt_at')
    if kind == 'legacy':
        record.pop('skill_review')
        record.pop('revision')
        review.save(record)
    elif kind == 'snoozed':
        record['snooze_until'] = time.time() + 300
        review.save(record)
    with TelegramStandin() as server:
        cfg = {'skills': {'write_approval': True}, 'platforms': {'telegram': {'enabled': True,
            'token': server.token, 'home_channel': {'platform': 'telegram', 'chat_id': '111'},
            'extra': {'base_url': server.api_base, 'base_file_url': server.file_base}}}}
        if kind == 'disabled':
            cfg['tools'] = {'write_approval': {'nag_hours': 0}}
        (home / 'config.yaml').write_text(json.dumps(cfg))
        assert tick() == (1 if kind == 'retry' else 0)
        if kind == 'legacy':
            assert send_prompt(review.inspect(record['id']))
        assert tick() == 0
        assert len(server.calls_of('sendMessage')) == (1 if kind in {'retry', 'legacy'} else 0)
        wa.discard_pending('skills', record['id'])
        before = len(server.calls_of('sendMessage'))
        assert tick(now=time.time() + 100000) == 0
        assert len(server.calls_of('sendMessage')) == before
