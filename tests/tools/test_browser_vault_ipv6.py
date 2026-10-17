"""IPv6 saved bindings use exact origins without rebinding legacy ambiguity."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore
from tools import browser_vault_tool


def _save(store):
    return store.add_item('login', 'IPv6', {
        'identifier_type': 'email', 'identifier': 'ipv6@example.test', 'password': 'test-canary',
    }, origin='https://[::1]:8443')


def test_ambiguous_legacy_binding_refuses_before_resolution(tmp_path):
    store = VaultStore(tmp_path / 'vault')
    meta = _save(store)
    # Reproduce the persisted record written by the old normalizer; never guess
    # which host/port it meant or silently rewrite it during metadata reads.
    records = store._read_all()
    records[0]['origin'] = 'https://::1:8443'
    store._write_all(records)
    reloaded = VaultStore(tmp_path / 'vault')
    supervisor = SimpleNamespace(focus_page=lambda *_a, **_kw: {'ok': False})
    before = reloaded._vault_path.read_bytes()
    with patch('agent.vault_store.get_vault_store', return_value=reloaded), \
         patch.object(reloaded, 'resolve_secret', wraps=reloaded.resolve_secret) as resolve, \
         patch.object(browser_vault_tool, '_ensure_supervisor', return_value=supervisor), \
         patch.object(browser_vault_tool, '_current_page_origin', return_value='https://[::1]:8443'), \
         patch.object(browser_vault_tool, '_eval_js_secret') as write:
        result = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id='legacy-ipv6'))
    assert result['error_type'] == 'origin_mismatch'
    assert resolve.call_count == write.call_count == 0
    assert reloaded.get_meta(meta.id).origin == 'https://::1:8443'
    assert reloaded._vault_path.read_bytes() == before


@pytest.mark.parametrize('child', ['http://[::1]:8443', 'https://[::2]:8443', 'https://[::1]:8444'])
def test_ipv6_exact_pair_decline_never_resolves_or_falls_back(tmp_path, child):
    store = VaultStore(tmp_path / 'vault')
    meta = _save(store)
    route = {'page_session_id': 'page', 'frame_id': 'selected',
             'frame_session_id': 'child', 'frame_loader_id': 'document'}
    supervisor = SimpleNamespace(focus_page=lambda *_a, **_kw: {
        'ok': True, 'url': meta.origin, 'frame_origin': child, 'route': route,
    })
    with patch('agent.vault_store.get_vault_store', return_value=store), \
         patch.object(store, 'resolve_secret', wraps=store.resolve_secret) as resolve, \
         patch.object(browser_vault_tool, '_ensure_supervisor', return_value=supervisor), \
         patch.object(browser_vault_tool, '_eval_js_in_route') as inspect, \
         patch.object(browser_vault_tool, '_eval_js_secret') as write, \
         patch('tools.approval_prompt.request_elicitation_consent', return_value='decline') as consent:
        result = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id='exact-ipv6'))
    assert result['error_type'] == 'cross_origin_declined'
    assert consent.call_count == 1
    assert meta.origin in consent.call_args.args[0] and child in consent.call_args.args[0]
    assert resolve.call_count == inspect.call_count == write.call_count == 0
