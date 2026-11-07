"""Profile outboxes and unacknowledged claims must wake the Desktop relay."""
import pytest
from tui_gateway import server

@pytest.fixture
def watcher(tmp_path, monkeypatch):
    (tmp_path/'config.yaml').write_text('display: {}\n', encoding='utf-8')
    monkeypatch.setattr(server, '_watcher_home', lambda: tmp_path)
    monkeypatch.setattr(server, '_change_sigs', {})
    monkeypatch.setattr(server, '_change_checked_at', {})
    monkeypatch.setattr(server, '_change_broadcast_at', {})
    monkeypatch.setattr(server, '_pairing_roots_cache', None)
    events = []
    monkeypatch.setattr(server, '_broadcast_global_event', lambda event, payload=None: events.append((event, payload)))
    return tmp_path, events

def pending(events):
    return [event for event in events if event[0] == 'bot_relay.outbox.pending']

@pytest.mark.parametrize("launch", ["default", "linkedin"])
def test_sibling_outbox_wakes_relay_even_on_root_socket(watcher, monkeypatch, launch):
    home, events = watcher
    profile = home/'profiles/linkedin'
    profile.mkdir(parents=True)
    (profile/'config.yaml').write_text('display: {}\n', encoding='utf-8')
    monkeypatch.setattr(server, '_watcher_home', lambda: home if launch == 'default' else profile)
    server._broadcast_watched_changes(now=0)
    outbox = profile/'bot_relay/outbox'
    outbox.mkdir(parents=True)
    (outbox/('a'*32+'.json')).write_text('{}', encoding='utf-8')
    server._broadcast_watched_changes(now=2)
    assert pending(events) == [('bot_relay.outbox.pending', {})]

def test_existing_pending_claim_repeats_until_acknowledged(watcher):
    home, events = watcher
    claimed = home/'bot_relay/claimed'
    claimed.mkdir(parents=True)
    key = 'b'*32+'.json'
    (claimed/key).write_text('{}', encoding='utf-8')
    server._broadcast_watched_changes(now=0)
    assert len(pending(events)) == 1
    server._broadcast_watched_changes(now=2)
    assert len(pending(events)) == 1
    server._broadcast_watched_changes(now=31)
    assert len(pending(events)) == 2
    replies = home/'bot_relay/replies'
    replies.mkdir()
    (replies/key).write_text('{}', encoding='utf-8')
    server._broadcast_watched_changes(now=62)
    assert len(pending(events)) == 2

def test_startup_pending_outbox_is_not_silently_seeded(watcher):
    home, events = watcher
    outbox = home/'bot_relay/outbox'
    outbox.mkdir(parents=True)
    (outbox/('c'*32+'.json')).write_text('{}', encoding='utf-8')
    server._broadcast_watched_changes(now=0)
    assert pending(events) == [('bot_relay.outbox.pending', {})]
