"""Event-boundary and unlock-lease regression probes."""
import pytest
from tests.tools.test_browser_vault_signup_real import (
    pytestmark, signup_runtime, dispatch, leaf_doc, chrome, tab_tmp,
)


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('event', ['input', 'change'])
def test_first_field_mutation_stops_confirmation(signup_runtime, monkeypatch, event):
    c = signup_runtime
    monkeypatch.setattr('tools.approval_prompt.request_elicitation_consent', lambda *a, **k: 'accept')
    saved = dispatch('browser_vault_save_login', {'label': 'Synthetic event boundary'})
    wire, doc = leaf_doc(c)
    try:
        import json
        doc.evaluate("window.changes=0;let p=document.querySelector('input[name=password]');p.addEventListener('change',()=>window.changes++);p.addEventListener(" + json.dumps(event) + ",()=>p.remove(),{once:true});true")
        result = dispatch('browser_vault_fill', {'handle': saved['handle']})
        assert not result['success'] and result['filled_fields'] == 1
        assert doc.evaluate("document.querySelector('input[name=confirmPassword]').value === ''")
        if event == 'input':
            assert doc.evaluate('window.changes') == 0
    finally:
        wire.close()


@pytest.mark.live_system_guard_bypass
def test_takeover_and_handback_during_unlock_never_resolves(signup_runtime, monkeypatch):
    c = signup_runtime
    from tools import browser_vault_tool as vault
    from tools.bot_desktop import lease
    from agent.vault_backends.local import LocalLoginBackend
    meta = c.store.add_item('login', 'Synthetic leaf', {
        'identifier': 'fixture@example.test', 'identifier_type': 'email',
        'password': c.password}, origin=c.leaf_origin)
    monkeypatch.setattr(vault, '_bot_desktop_browser_session', lambda task: True)
    monkeypatch.setattr(LocalLoginBackend, 'needs_unlock', True)
    monkeypatch.setattr(LocalLoginBackend, 'is_unlocked', lambda self: False)
    unlocked = []
    def unlock(backend):
        assert backend == 'local'
        unlocked.append(True)
        lease.acquire('synthetic-unlock-viewer')
        lease.release('synthetic-unlock-viewer')
        return '{"success": true}'
    monkeypatch.setattr(vault, 'browser_vault_unlock', unlock)
    resolved = []
    monkeypatch.setattr(LocalLoginBackend, 'resolve_password', lambda *a: resolved.append(True))
    result = dispatch('browser_vault_fill', {'handle': meta.id})
    assert unlocked == [True]
    assert not result['success'] and not resolved
