"""Preserve a proved current mission tab across separate exec calls."""
import sys
import pytest

from tools import browser_tab_capture as capture
from tools.browser_tab_ownership import OwnershipRegistry
from tests.tools.test_browser_tab_capture import WS, harness


@pytest.mark.parametrize('target_id,session_id,expected', [
    ('second', 'retained-session', []),
    ('foreign', 'foreign-session', ['first']),
    ('gone', 'stale-session', ['first']),
    ('second', None, ['first']),
    (None, 'orphan-session', ['first']),
])
def test_entry_retains_current_live_exact_mission_target(tmp_path, monkeypatch, target_id, session_id, expected):
    path = tmp_path / 'tabs.sqlite'
    registry = OwnershipRegistry(path)
    previous = registry.admit('owner', 'g', WS, 'mission')
    registry.record_created(previous, 'first')
    registry.record_created(previous, 'second')
    registry.record_created(previous, 'gone')
    registry.finish(previous)
    foreign = registry.admit('owner', 'g', WS, 'different-mission')
    registry.record_created(foreign, 'foreign')
    registry.finish(foreign)
    token = registry.admit('owner', 'g', WS, 'mission')
    attached, requests = harness(monkeypatch, {'first', 'second', 'foreign'})
    helpers = sys.modules['browser_harness.helpers']
    original = helpers._send

    def send(req, *args, **kwargs):
        if req.get('meta') == 'connection_status':
            requests.append(req)
            return {'target_id': target_id, 'session_id': session_id,
                    'page': {'targetId': target_id}}
        return original(req, *args, **kwargs)

    helpers._send = send
    capture.begin_exec(path, token)
    assert attached == expected
    if not expected:
        assert not any(req.get('meta') == 'set_session' for req in requests)
    assert registry.call(token)['inflight'] == 0
    assert registry.call(token)['state'] == 'active'
