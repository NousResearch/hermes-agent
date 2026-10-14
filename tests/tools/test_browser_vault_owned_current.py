"""Vault writes stay on the completed mission's exact page across user prompts."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as supervisors
from tools import browser_vault_tool as vault
from tools.browser_tab_ownership import OwnershipRegistry
from tools.registry import registry


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setattr('hermes_constants.get_hermes_home', lambda: tmp_path)
    ledger = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    ws = 'ws://localhost/devtools/browser/exact'
    token = ledger.admit('session:a', 'a', ws, 'mission')
    ledger.record_created(token, 'owned')
    ledger.record_created(token, 'other')
    ledger.publish_current(token, 'owned')
    ledger.finish(token)
    supervisor = supervisors.CDPSupervisor('a', ws)
    supervisor._vault_owned_call = (str(ledger.path), token)
    supervisor._active = True
    supervisor._ws = object()
    supervisor._loop = SimpleNamespace(is_running=lambda: True)
    supervisor._page_session_id = 'foreign-session'
    calls = []

    async def cdp(method, params=None, **kwargs):
        calls.append((method, params, kwargs))
        if method == 'Target.attachToTarget':
            assert params['targetId'] == 'owned'
            return {'result': {'sessionId': 'owned-session'}}
        if method == 'Target.detachFromTarget':
            assert params['sessionId'] == 'owned-session'
            return {}
        assert method == 'Runtime.evaluate'
        assert kwargs['session_id'] == 'owned-session'
        expression = params['expression']
        if expression == 'window.location.href':
            value = 'https://example.com/code'
        elif 'const elements = Array.from' in expression:
            value = [{'index': 0, 'type': 'text', 'name': 'otp',
                      'autocomplete': 'one-time-code', 'label': 'Code'}]
        else:
            value = {'filled': 1}
        return {'result': {'result': {'type': 'object', 'value': value}}}

    monkeypatch.setattr(supervisor, '_cdp', cdp)
    monkeypatch.setattr(supervisors, '_schedule', lambda coro, loop, **kw: asyncio.run(coro))
    monkeypatch.setattr(supervisors.SUPERVISOR_REGISTRY, 'get', lambda task: supervisor)
    monkeypatch.setattr(vault, '_bot_desktop_browser_session', lambda task: None)
    monkeypatch.setattr(supervisor, 'focus_page', lambda *a, **k: pytest.fail('tab scan'))
    from tools import browser_tool_session
    monkeypatch.setattr(browser_tool_session, '_run_browser_command', lambda *a, **k: pytest.fail('CLI fallback'))
    from agent.vault_backends import unlock
    monkeypatch.setattr(unlock, 'can_prompt_here', lambda: True)
    monkeypatch.setattr(unlock, 'get_code_prompt_callback', lambda: lambda *a: '123456')
    return ledger, token, supervisor, calls


def enter_code():
    return json.loads(registry.get_entry('browser_vault').handler({'action': 'enter_code'}, task_id='a'))


def test_owned_fill_uses_exact_session_even_when_default_focus_changes(owned, monkeypatch):
    _, _, supervisor, calls = owned
    from agent.vault_backends import unlock

    def prompt(*args):
        supervisor._page_session_id = 'another-foreign-session'
        return '123456'

    monkeypatch.setattr(unlock, 'get_code_prompt_callback', lambda: prompt)
    result = enter_code()
    assert result['success'] and result['filled_fields'] == 1, result
    assert all(call[2]['session_id'] == 'owned-session' for call in calls if call[0] == 'Runtime.evaluate')


def invalidate(ledger, token, supervisor, state):
    if state == 'held':
        ledger.hold(supervisor.cdp_url, 'owned', 'opaque-live-actor')
    elif state in ('unknown', 'unresolved'):
        field, value = ('owner', 'unknown:unscoped') if state == 'unknown' else ('generation', 'unresolved')
        with ledger._db() as db:
            for table in ('owners', 'calls', 'targets'):
                db.execute(f'UPDATE {table} SET {field}=?', (value,))
    elif state == 'binding':
        supervisor._vault_owned_call = (str(ledger.path), 'another-call')
    elif state == 'transport':
        supervisor._ws = object()
    elif state == 'endpoint':
        supervisor.cdp_url += '-other'
    elif state == 'retired':
        ledger.retire('session:a', 'a', evidence='fixture:finished')
    elif state == 'nonlatest':
        newer = ledger.admit('session:a', 'a', supervisor.cdp_url, 'mission')
        ledger.finish(newer)
    else:
        with ledger._db() as db:
            if state in ('active', 'quarantined'):
                db.execute('UPDATE calls SET state=? WHERE token=?', (state, token))
            elif state == 'pending':
                db.execute('UPDATE targets SET pending_close=1')
            elif state == 'foreign':
                db.execute("UPDATE targets SET owner='someone-else'")
            elif state == 'current':
                db.execute("UPDATE calls SET current_target='other' WHERE token=?", (token,))


@pytest.mark.parametrize('state', ['active', 'quarantined', 'nonlatest', 'pending', 'retired', 'foreign', 'unknown', 'unresolved', 'held'])
def test_unavailable_owned_current_never_attaches_or_falls_back(owned, state):
    ledger, token, supervisor, calls = owned
    invalidate(ledger, token, supervisor, state)
    assert not enter_code()['success']
    assert calls == []


@pytest.mark.parametrize('state', ['binding', 'current', 'active', 'quarantined', 'nonlatest',
                                  'pending', 'retired', 'foreign', 'transport', 'endpoint', 'unknown', 'unresolved', 'held'])
def test_prompt_change_refuses_before_secret_evaluation(owned, monkeypatch, state):
    ledger, token, supervisor, calls = owned
    from agent.vault_backends import unlock
    before = []

    def prompt(*args):
        before.extend(calls)
        invalidate(ledger, token, supervisor, state)
        return '123456'

    monkeypatch.setattr(unlock, 'get_code_prompt_callback', lambda: prompt)
    result = enter_code()
    assert not result['success']
    evaluations = [call for call in calls if call[0] != "Target.detachFromTarget"]
    assert before and evaluations == before


def test_attachment_failure_does_not_fall_back(owned, monkeypatch):
    _, _, supervisor, _ = owned

    async def unavailable(*args, **kwargs):
        raise RuntimeError('target closed')

    monkeypatch.setattr(supervisor, '_cdp', unavailable)
    assert not enter_code()['success']


def test_unlock_prompt_cannot_rebind_fill(owned, monkeypatch):
    ledger, token, supervisor, calls = owned
    from agent import vault_backends
    backend = SimpleNamespace(name='fixture', needs_unlock=True, is_unlocked=lambda: False,
                              get_meta=lambda handle: pytest.fail('continued after changed prompt'))
    monkeypatch.setattr(vault_backends, 'backend_for_handle', lambda handle: backend)

    def unlock(name):
        invalidate(ledger, token, supervisor, 'binding')
        return json.dumps({'success': True})

    monkeypatch.setattr(vault, 'browser_vault_unlock', unlock)
    # Validation after unlock must happen before metadata or secret resolution.
    result = json.loads(vault.browser_vault_fill('handle', task_id='a'))
    assert not result['success']
    assert not any(call[0] == 'Runtime.evaluate' for call in calls)


@pytest.mark.parametrize('missing', ['supervisor', 'handoff'])
def test_owned_mode_without_handoff_never_enters_legacy_path(owned, monkeypatch, missing):
    from tools import browser_use_cli
    _, _, supervisor, calls = owned
    monkeypatch.setattr(browser_use_cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(browser_use_cli, 'is_browser_use_cli_mode', lambda: True)
    if missing == 'supervisor':
        monkeypatch.setattr(supervisors.SUPERVISOR_REGISTRY, 'get', lambda task: None)
    else:
        supervisor._vault_owned_call = None
    result = enter_code()
    assert not result['success'] and result['error_type'] == 'owned_current_unavailable'
    assert calls == []


def test_other_profile_cannot_consume_owned_handoff(owned, tmp_path, monkeypatch):
    _, _, _, calls = owned
    monkeypatch.setattr('hermes_constants.get_hermes_home', lambda: tmp_path / 'other-profile')
    result = enter_code()
    assert not result['success'] and result['error_type'] == 'owned_current_unavailable'
    assert calls == []
