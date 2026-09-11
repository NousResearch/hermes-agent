"""Onboarding facts cross profiles only through an explicit, bounded transfer."""
from pathlib import Path
from unittest.mock import Mock

import pytest

import tui_gateway.server as srv
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.memory_tool import load_on_disk_store


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    guide = home / 'profiles' / 'hermes-setup'
    guide.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(guide))
    return home, guide


def remember(answers):
    return srv._methods['profiles.remember_onboarding'](1, {'answers': answers})


def test_agreed_facts_reach_future_default_sessions_without_sharing_or_rewriting_live_memory(homes):
    home, guide = homes
    token = set_hermes_home_override(home)
    try:
        before = load_on_disk_store()
        before.add('user', 'Existing preference: concise replies.')
        frozen = before.format_for_system_prompt('user')
    finally:
        reset_hermes_home_override(token)
    answers = {'name': 'Ada', 'context': 'A garden tracker', 'connectors': ['Calendar'],
               'focus': ['research'], 'api_key': 'must-not-transfer'}
    assert remember(answers)['result']['saved'] is True
    assert remember(answers)['result']['saved'] is True
    assert not (guide / 'memories' / 'USER.md').exists()
    token = set_hermes_home_override(home)
    try:
        future = load_on_disk_store()
        text = future.format_for_system_prompt('user')
        assert 'Ada' in text and 'A garden tracker' in text and 'Calendar' in text
        assert 'Focus areas: research' in text
        assert text.count('Ada') == 1 and 'Existing preference' in text
        assert 'must-not-transfer' not in text
        assert before.format_for_system_prompt('user') == frozen
    finally:
        reset_hermes_home_override(token)


def test_answers_without_focus_or_theme_persist_the_agreed_facts(homes):
    home, _guide = homes
    assert remember({'name': 'Ada', 'context': 'x', 'connectors': []})['result']['saved'] is True
    token = set_hermes_home_override(home)
    try:
        text = load_on_disk_store().format_for_system_prompt('user')
        assert 'User prefers to be called: Ada' in text
        assert 'Working on: x' in text
    finally:
        reset_hermes_home_override(token)


def test_disabled_or_over_budget_memory_refuses_without_claiming_saved(homes):
    home, _guide = homes
    (home / 'config.yaml').write_text('memory:\n  user_profile_enabled: false\n')
    result = remember({'name': 'Ada'})
    assert 'error' in result
    assert not (home / 'memories' / 'USER.md').exists()
    (home / 'config.yaml').write_text('memory:\n  user_char_limit: 10\n')
    assert 'error' in remember({'name': 'Ada Lovelace'})
    assert not (home / 'memories' / 'USER.md').exists()



def test_setup_profile_no_alias_keeps_creation_local_to_its_profile(homes, monkeypatch):
    from hermes_cli import profiles

    home, _guide = homes
    wrapper = Mock()
    collision = Mock(return_value=False)
    monkeypatch.setattr(profiles, "create_wrapper_script", wrapper)
    monkeypatch.setattr(profiles, "check_alias_collision", collision)
    response = srv.handle_request({
        "id": "profile", "method": "profiles.create",
        "params": {"name": "guide-without-alias", "no_alias": True, "no_skills": True},
    })
    assert response["result"]["ok"] is True
    assert (home / "profiles" / "guide-without-alias" / "config.yaml").exists()
    wrapper.assert_not_called()
    collision.assert_not_called()
