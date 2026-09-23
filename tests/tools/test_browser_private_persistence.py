"""Private browser continuity: real files/locks, mocked provider transport only."""
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

import pytest
import requests

from plugins.browser.browser_use.persistence import Lease
from plugins.browser.browser_use.provider import BrowserUseBrowserProvider


def session():
    return {'id': 'browser-1', 'status': 'active', 'cdpUrl': 'wss://example.invalid/cdp',
            'timeoutAt': (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()}


def provider():
    p = BrowserUseBrowserProvider()
    p._post_create = Mock(return_value=Mock(ok=True, status_code=200, json=lambda: session()))
    return p


def test_restart_recovers_same_browser_and_locks_profile(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    p = provider(); cfg = {'api_key': 'synthetic', 'base_url': 'https://example.invalid'}
    lease = Lease('profile-a', 'owner-a')
    try:
        first = lease.acquire(p, cfg)
        with pytest.raises(RuntimeError, match='already in use'):
            Lease('profile-a', 'owner-a')
        assert p._post_create.call_args.args[2]['profileId'] == 'profile-a'
        lease.checkpoint()
    finally:
        lease.release()  # process crash releases OS lock, not browser
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: Mock(status_code=200, json=lambda: first))
    resumed = Lease('profile-a', 'owner-a')
    try:
        assert resumed.acquire(p, cfg)['id'] == first['id']
        assert p._post_create.call_count == 1
        assert 'cdpUrl' not in resumed.path.read_text()
    finally:
        resumed.release()
    wrong_owner = Lease('profile-a', 'owner-b')
    try:
        with pytest.raises(RuntimeError, match='another pending conversation'):
            wrong_owner.acquire(p, cfg)
        assert p._post_create.call_count == 1
    finally:
        wrong_owner.release()


def test_unknown_create_never_repeats_and_other_profile_is_independent(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    p = provider(); p._post_create.side_effect = requests.Timeout()
    cfg = {'api_key': 'synthetic', 'base_url': 'https://example.invalid'}
    lease = Lease('profile-a', 'owner-a')
    try:
        with pytest.raises(requests.Timeout): lease.acquire(p, cfg)
    finally:
        lease.release()
    lease = Lease('profile-a', 'owner-a')
    try:
        with pytest.raises(RuntimeError, match='outcome unknown'): lease.acquire(p, cfg)
        assert p._post_create.call_count == 1
        other = Lease('profile-b', 'owner-b')
        try: assert not other.record
        finally: other.release()
    finally:
        lease.release()


def test_configured_provider_failure_does_not_fall_back(monkeypatch):
    from tools import browser_tool_session as sessions
    p = provider()
    monkeypatch.setattr(p, 'requires_explicit_recovery', lambda: True)
    monkeypatch.setattr(p, 'create_session', Mock(side_effect=requests.Timeout()))
    local = Mock()
    monkeypatch.setattr(sessions, '_create_local_session', local)
    with pytest.raises(requests.Timeout):
        sessions._create_cloud_session_or_fallback('task', p)
    local.assert_not_called()


def test_conversation_owners_allow_group_and_cron_without_cross_takeover(monkeypatch):
    from plugins.browser.browser_use.persistence import owner
    from gateway import session_context
    context = {'HERMES_SESSION_PLATFORM': 'telegram', 'HERMES_SESSION_CHAT_ID': '-100',
               'HERMES_SESSION_USER_ID': '1', 'HERMES_SESSION_CHAT_TYPE': 'supergroup',
               'HERMES_SESSION_KEY': 'group-thread', 'HERMES_SESSION_THREAD_ID': '4'}
    monkeypatch.setattr(session_context, 'get_session_env', lambda name: context.get(name, ''))
    group = owner('task-a')
    context['HERMES_SESSION_USER_ID'] = '2'
    assert owner('task-b') == group  # Same authorized conversation, another participant.
    context['HERMES_SESSION_THREAD_ID'] = '5'
    assert owner('task-b') != group
    context.update(HERMES_CRON_SESSION='1', HERMES_SESSION_PLATFORM='', HERMES_SESSION_CHAT_ID='')
    assert owner('cron:job:run-a') != owner('cron:job:run-b')
    assert owner('cron:job:run-a') != group
    with pytest.raises(RuntimeError, match='execution task ID'):
        owner('default')
    p = provider()
    p._persistent_leases['browser-1'] = Mock()
    assert p.keep_session('browser-1') is False
    p._persistent_leases['browser-1'].checkpoint.assert_not_called()


def test_private_owner_compatibility_and_no_anonymous_owner(monkeypatch):
    import hashlib
    from plugins.browser.browser_use.persistence import owner
    from gateway import session_context
    context = {'HERMES_SESSION_PLATFORM': 'telegram', 'HERMES_SESSION_CHAT_ID': '1',
               'HERMES_SESSION_USER_ID': '1', 'HERMES_SESSION_CHAT_TYPE': 'private',
               'HERMES_SESSION_KEY': 'private-thread', 'HERMES_SESSION_THREAD_ID': ''}
    monkeypatch.setattr(session_context, 'get_session_env', lambda name: context.get(name, ''))
    assert owner('task') == hashlib.sha256(json.dumps(['1', '1', '', 'private-thread']).encode()).hexdigest()
    context['HERMES_SESSION_USER_ID'] = '2'
    with pytest.raises(RuntimeError, match='does not match'):
        owner('task')
    context.clear()
    with pytest.raises(RuntimeError, match='runtime task ID'):
        owner('default')
    assert owner('cli-session-a') != owner('cli-session-b')


def test_operator_gateway_uses_only_scoped_token(monkeypatch):
    from plugins.browser.browser_use import persistence
    from plugins.browser.browser_use import provider as module
    monkeypatch.setattr(persistence, 'settings', lambda: {'gateway_url':'https://agents.example/browser-runtime/'})
    read = Mock(return_value='scoped-token')
    monkeypatch.setattr(module, 'get_secret', read)
    config = provider()._get_config_or_none()
    assert config == {'api_key':'scoped-token','base_url':'https://agents.example/browser-runtime','managed_mode':False,'tenant_gateway':True}
    read.assert_called_once_with('BROWSER_USE_GATEWAY_TOKEN')
    read.return_value = None
    assert provider()._get_config_or_none() is None
    monkeypatch.setattr(persistence, 'settings', lambda: {'gateway_url':'http://unsafe.example'})
    with pytest.raises(ValueError, match='HTTPS'):
        provider()._get_config_or_none()


def test_gateway_reconciliation_preserves_operation_identity(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    p = provider(); p._post_create.side_effect = requests.Timeout()
    cfg = {'api_key':'scoped', 'base_url':'https://gateway.invalid', 'tenant_gateway':True}
    lease = Lease('client-a','owner-a')
    try:
        with pytest.raises(requests.Timeout):lease.acquire(p,cfg)
        operation = lease.record['operation_id']
    finally:lease.release()
    p._post_create.side_effect = None
    resumed = Lease('client-a','owner-a')
    try:
        assert resumed.acquire(p,cfg)['id'] == 'browser-1'
        assert p._post_create.call_args.args[2]['metadata']['hermes_operation'] == operation
    finally:resumed.release()
