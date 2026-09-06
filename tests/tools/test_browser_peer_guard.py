"""Observed CDP peers must gate navigation even when the URL stays public."""
import json
from types import SimpleNamespace

import pytest
from tools import browser_supervisor as bs, browser_tool as bt
from tools import browser_tool_cloud as cloud, browser_tool_session as session


@pytest.mark.parametrize('peer,allowed,success', [
    ('127.0.0.1', False, False), ('169.254.169.254', True, False),
    ('not-an-ip', True, False), ('93.184.216.34', False, True),
    ('127.0.0.1', True, True),
])
def test_navigation_checks_observed_peer(monkeypatch, peer, allowed, success):
    records = []
    def rotate():
        old = tuple(records)
        records.clear()
        return old
    supervisor = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
        start_network_response_window=rotate, flush_network_events=lambda: True,
    )
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, 'get', lambda _: supervisor)
    monkeypatch.setattr(bt, '_is_camofox_mode', lambda: False)
    monkeypatch.setattr(bt, '_navigation_session_key', lambda task, url: task)
    monkeypatch.setattr(bt, '_url_policy_error', lambda *a, **k: None)
    monkeypatch.setattr(bt, '_attach_auto_snapshot', lambda *a: None)
    monkeypatch.setattr(cloud, '_is_local_backend', lambda: False)
    monkeypatch.setattr(cloud, '_allow_private_urls', lambda: allowed)
    monkeypatch.setattr(session, '_get_session_info', lambda _: {'_first_nav': False})
    def command(task, command, args, **kwargs):
        if args != ['about:blank']:
            records.append(SimpleNamespace(ts=0, remote_ip=peer, url='https://public.example/'))
        return {'success': True, 'data': {'url': args[0], 'title': 'browser content'}}
    monkeypatch.setattr(session, '_run_browser_command', command)
    result = json.loads(bt.browser_navigate('https://public.example/', 'peer-test'))
    assert result['success'] is success
    if not success:
        assert 'browser content' not in json.dumps(result)


@pytest.mark.parametrize('peer,private,blocked', [
    ('93.184.216.34', False, False), ('2606:4700:4700::1111', False, False),
    ('127.0.0.1', False, True), ('127.0.0.1', True, False),
    ('fd00::1', False, True), ('fd00::1', True, False),
    ('169.254.169.254', True, True), ('::ffff:169.254.169.254', True, True),
    ('bad-peer', False, True), ('bad-peer', True, True),
])
def test_reported_peer_uses_shared_ip_policy(peer, private, blocked):
    from tools.url_safety import ip_address_block_reason
    assert bool(ip_address_block_reason(peer, allow_private=private)) is blocked
