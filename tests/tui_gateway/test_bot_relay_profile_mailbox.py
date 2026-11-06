"""Legacy relay RPCs and bot tools must use the same profile mailbox."""
import json

import pytest

import tui_gateway.server as srv
from tools import bot_relay


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    home = root / 'profiles' / 'linkedin'
    home.mkdir(parents=True)
    (root / 'config.yaml').write_text('{}\n', encoding='utf-8')
    (home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(root))
    return root, home


def result(method, **params):
    frame = srv.handle_request({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params})
    assert 'error' not in frame, frame
    return frame['result']


def test_roster_sync_reaches_named_bot_without_replacing_default(homes):
    root, home = homes
    rows = [{'profile': 'cowork', 'handle': 'cowork', 'connection_id': 'pro'}]
    bot_relay.write_remote_roster(root, [{'profile': 'other', 'connection_id': 'other'}])
    assert result('bot_relay.roster.sync', profile='linkedin', agents=rows)['count'] == 1
    assert bot_relay.resolve_remote_target('cowork', bot_relay.read_remote_roster(home))
    assert bot_relay.read_remote_roster(root)[0]['profile'] == 'other'


def test_outbox_drain_and_reply_stay_with_sending_profile(homes):
    root, home = homes
    target = {'profile': 'cowork', 'handle': 'cowork', 'connection_id': 'pro'}
    named = bot_relay.enqueue_envelope(home, target=target, message='synthetic named',
                                      sender_profile='linkedin', sender_handle='linkedin')
    default = bot_relay.enqueue_envelope(root, target=target, message='synthetic default',
                                        sender_profile='default', sender_handle='hermes')
    assert [e['id'] for e in result('bot_relay.outbox.drain', profile='linkedin')['envelopes']] == [named['id']]
    result('bot_relay.reply', profile='linkedin', id=named['id'], reply='synthetic acknowledgement')
    reply = bot_relay.relay_root(home) / 'replies' / (named['id'] + '.json')
    assert json.loads(reply.read_text(encoding='utf-8-sig'))['reply'] == 'synthetic acknowledgement'
    assert not (bot_relay.relay_root(root) / 'replies' / reply.name).exists()
    assert [e['id'] for e in result('bot_relay.outbox.drain')['envelopes']] == [default['id']]


def test_roster_sync_uses_ticket_profile_when_param_is_absent(homes):
    from tui_gateway.ws_legacy_fallback import _connection_profile_home
    root, home = homes
    token = _connection_profile_home.set(str(home))
    try:
        result('bot_relay.roster.sync', agents=[{'profile': 'cowork', 'connection_id': 'pro'}])
    finally:
        _connection_profile_home.reset(token)
    assert bot_relay.read_remote_roster(home)[0]['profile'] == 'cowork'
    assert bot_relay.read_remote_roster(root) == []
