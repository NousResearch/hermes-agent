"""Model arguments cannot carry relay identity or general room operations."""
import json

import pytest


@pytest.mark.parametrize('extra', ['room_id', 'owner', 'profile', 'sender', 'platform', 'token', 'method'])
def test_model_cannot_supply_authority(extra, monkeypatch):
    import tools.bot_room_delegate
    from tools.registry import registry
    result = json.loads(registry.dispatch('bot_room_delegate', {
        'recipient': 'server', 'message': 'task', extra: 'forged'}))
    assert result['status'] == 'denied'


def test_disabled_tool_cannot_execute_even_with_identity_kwargs(tmp_path, monkeypatch):
    import tools.bot_room_delegate as tool
    from tools.registry import registry
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('{}')
    assert tool.check_requirements() is False
    result = json.loads(registry.dispatch('bot_room_delegate', {'recipient': 'server', 'message': 'task'},
                                         user_id='owner', profile='default'))
    assert result['status'] == 'denied'
    schema = registry.get_schema('bot_room_delegate')['parameters']
    assert set(schema['properties']) == {'recipient', 'message'}
    assert schema['properties']['recipient']['enum'] == ['server', 'flohmarkt', 'buchhaltung', 'listing']
    assert schema['additionalProperties'] is False


def test_result_tool_schema_is_read_only_reference_and_in_telegram_bundle():
    import tools.bot_room_delegate
    from tools.registry import registry
    from toolsets import TOOLSETS

    schema_record = registry.get_schema('bot_room_result')
    assert schema_record is not None
    schema = schema_record['parameters']
    assert set(schema['properties']) == {'request_id'}
    assert schema['required'] == ['request_id']
    assert schema['additionalProperties'] is False
    assert 'bot_room_delegate' in TOOLSETS['hermes-telegram']['tools']
    assert 'bot_room_result' in TOOLSETS['hermes-telegram']['tools']
    raw = registry.dispatch('bot_room_result', {'request_id': 'a' * 64, 'room_id': 'forged'})
    result = json.loads(raw) if isinstance(raw, str) else raw
    assert result['status'] == 'denied'
