"""Named-profile relay behavior against the shared Desktop mailbox."""
import json
from types import SimpleNamespace

from tools import bot_mode_dm, bot_mode_probe, bot_relay


def test_named_sender_queues_in_install_root_when_profile_roster_disagrees(tmp_path, monkeypatch):
    """The sender's private roster may differ, but the Desktop drains only the install root."""
    root = tmp_path / 'state'
    sender = root / 'profiles' / 'sender'
    sender.mkdir(parents=True)
    (sender / 'profile.yaml').write_text('ui_meta:\n  hermes-bots:\n    shape: cloud\n')
    own = dict(profile='remote', handle='remote', connection_id='own', title='Owned remote')
    other = dict(profile='remote', handle='remote', connection_id='foreign', title='Foreign remote')
    bot_relay.write_remote_roster(sender, [own])
    bot_relay.write_remote_roster(root, [other])
    agent = SimpleNamespace(_session_db=SimpleNamespace(db_path=str(sender / 'state.db')),
                            _session_title_hint='Bot Chat')
    commands = []
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery', lambda command, *a, **kw:
                        commands.append(command) or json.dumps({'status': 'sent'}))

    result = json.loads(bot_mode_dm.message_agent_tool('remote', 'hello', agent=agent))
    assert result['status'] == 'sent'
    envelopes = bot_relay.claim_pending_envelopes(root)
    assert len(envelopes) == 1 and envelopes[0]['target_connection'] == 'foreign'
    assert envelopes[0]['message'] == 'Message from 🤖 sender (@sender): hello'
    assert bot_relay.claim_pending_envelopes(sender) == []
    # Prompt discovery is separately profile-scoped; it cannot choose the mailbox.
    section = bot_mode_probe.get_bot_mode_protocol_section(sender, force_refresh=True)
    assert 'Owned remote' in section and 'Foreign remote' not in section


def _sender_agent(home):
    """An agent whose session db sits in ``home`` — the only thing ``_agent_home`` reads."""
    from types import SimpleNamespace as _NS

    return _NS(_session_db=_NS(db_path=str(home / 'state.db')), _session_title_hint='Bot Chat')


def _spy_delivery(monkeypatch):
    commands = []
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery', lambda command, *a, **kw:
                        commands.append(command) or json.dumps({'status': 'sent'}))
    return commands


def test_named_profile_falls_back_to_the_machine_root_relay(tmp_path, monkeypatch):
    """A named profile's home holds no relay folder — the Desktop only ever writes the machine
    root — so a cross-device DM must resolve against the root roster and queue there. Without the
    fallback the sender reports "No teammate named ..." for a row the root roster lists."""
    root = tmp_path / 'state'
    sender = root / 'profiles' / 'sender'
    sender.mkdir(parents=True)
    (sender / 'profile.yaml').write_text('ui_meta:\n  hermes-bots:\n    shape: cloud\n')
    bot_relay.write_remote_roster(root, [dict(profile='remote', handle='remote',
                                              connection_id='homelab', title='Homelab bot')])
    assert bot_relay.read_remote_roster(sender) == []  # nothing profile-local to resolve against
    _spy_delivery(monkeypatch)

    result = json.loads(bot_mode_dm.message_agent_tool('remote@homelab', 'hello', agent=_sender_agent(sender)))
    assert result.get('status') == 'sent', result
    envelopes = bot_relay.claim_pending_envelopes(root)
    assert len(envelopes) == 1 and envelopes[0]['target_connection'] == 'homelab'
    assert envelopes[0]['message'] == 'Message from 🤖 sender (@sender): hello'
    assert bot_relay.claim_pending_envelopes(sender) == []


def test_root_fallback_covers_a_bare_target_too(tmp_path, monkeypatch):
    """The same fallback at the unknown/same-name call site: a bare handle that no local profile
    answers to still resolves against the root roster (replies are addressed by handle)."""
    root = tmp_path / 'state'
    sender = root / 'profiles' / 'sender'
    sender.mkdir(parents=True)
    (sender / 'profile.yaml').write_text('ui_meta:\n  hermes-bots:\n    shape: cloud\n')
    bot_relay.write_remote_roster(root, [dict(profile='remote', handle='remote',
                                              connection_id='homelab', title='Homelab bot')])
    _spy_delivery(monkeypatch)

    result = json.loads(bot_mode_dm.message_agent_tool('remote', 'hello', agent=_sender_agent(sender)))
    assert result.get('status') == 'sent', result
    envelopes = bot_relay.claim_pending_envelopes(root)
    assert len(envelopes) == 1 and envelopes[0]['target_connection'] == 'homelab'


def test_stale_profile_only_relay_target_does_not_queue_an_undrainable_dm(tmp_path, monkeypatch):
    """A target removed from the shared roster cannot be revived by a stale local roster."""
    root = tmp_path / 'state'
    sender = root / 'profiles' / 'sender'
    sender.mkdir(parents=True)
    (sender / 'profile.yaml').write_text('ui_meta:\n  hermes-bots:\n    shape: cloud\n')
    bot_relay.write_remote_roster(sender, [dict(profile='remote', handle='remote', connection_id='own')])
    bot_relay.write_remote_roster(root, [dict(profile='remote', handle='remote', connection_id='homelab')])
    _spy_delivery(monkeypatch)

    result = json.loads(bot_mode_dm.message_agent_tool('remote@own', 'hello', agent=_sender_agent(sender)))
    assert 'error' in result and 'No teammate named' in result['error'], result
    assert bot_relay.claim_pending_envelopes(sender) == []
    assert bot_relay.claim_pending_envelopes(root) == []


def test_genuine_miss_keeps_the_roster_error(tmp_path, monkeypatch):
    """No row anywhere: the sender still gets today's roster error, not a silent success."""
    root = tmp_path / 'state'
    sender = root / 'profiles' / 'sender'
    sender.mkdir(parents=True)
    (sender / 'profile.yaml').write_text('ui_meta:\n  hermes-bots:\n    shape: cloud\n')
    bot_relay.write_remote_roster(root, [dict(profile='remote', handle='remote', connection_id='homelab')])
    _spy_delivery(monkeypatch)

    result = json.loads(bot_mode_dm.message_agent_tool('nobody@nowhere', 'hello', agent=_sender_agent(sender)))
    assert 'error' in result and 'No teammate named' in result['error'], result
