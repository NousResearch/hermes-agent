"""Contradictory/masked MFA fields must refuse before resolving a password."""
import json
from unittest.mock import Mock
import pytest
from agent.vault_store import VaultItemMeta
from tools import browser_vault_tool as vault

@pytest.mark.parametrize('name,label,autocomplete',[
    ('new_password','','current-password'), ('','Confirm password','current-password'),
    ('otp','',''), ('otp','','current-password'),
])
def test_excluded_fields_never_resolve_password(monkeypatch,name,label,autocomplete):
    backend=Mock()
    backend.needs_unlock=False
    backend.get_meta.return_value=VaultItemMeta(id='op:test',kind='login',label='Synthetic',origin='https://example.com',created_at='')
    monkeypatch.setattr('agent.vault_backends.backend_for_handle',lambda _:backend)
    monkeypatch.setattr(vault,'_focus_bound_origin',lambda *_:None)
    monkeypatch.setattr(vault,'_current_page_origin',lambda _:'https://example.com')
    monkeypatch.setattr(vault,'_eval_js',lambda *_:{'success':True,'result':[
        {'type':'password','name':name,'label':label,'autocomplete':autocomplete,'index':0},
        {'type':'email','name':'email','index':1}]})
    result=json.loads(vault.browser_vault_fill('op:test',task_id='synthetic'))
    assert result['success'] is False
    backend.resolve_password.assert_not_called()


@pytest.mark.parametrize('mixed', [False, True])
def test_text_password_candidate_never_outranks_real_password(monkeypatch, mixed):
    backend = Mock(needs_unlock=False)
    backend.name = 'onepassword'
    backend.get_meta.return_value = VaultItemMeta(id='op:test', kind='login', label='Synthetic',
                                                 origin='https://example.com', created_at='')
    backend.resolve_password.return_value = 'synthetic-only'
    monkeypatch.setattr('agent.vault_backends.backend_for_handle', lambda _: backend)
    monkeypatch.setattr(vault, '_focus_bound_origin', lambda *_: None)
    monkeypatch.setattr(vault, '_current_page_origin', lambda _: 'https://example.com')
    controls = [{'type': 'text', 'name': 'password', 'autocomplete': 'current-password', 'index': 0}]
    if mixed:
        controls.append({'type': 'password', 'name': 'password', 'index': 1})
    monkeypatch.setattr(vault, '_eval_js', lambda *_: {'success': True, 'result': controls})
    secret_eval = Mock(return_value={'success': True, 'result': '{"filled":1}'})
    monkeypatch.setattr(vault, '_eval_js_secret', secret_eval)
    from tools.registry import registry
    result = json.loads(registry.dispatch('browser_vault_fill', {'handle': 'op:test'}, task_id='synthetic'))
    if not mixed:
        assert result['success'] is False
        backend.resolve_password.assert_not_called()
        secret_eval.assert_not_called()
    else:
        assert result['success'] is True
        backend.resolve_password.assert_called_once()
        source = secret_eval.call_args.args[1]
        # Observe the generated payload, not implementation source text.
        fills = json.loads(source.split('const fills = ', 1)[1].split(';', 1)[0])
        assert [f['index'] for f in fills] == [1]
