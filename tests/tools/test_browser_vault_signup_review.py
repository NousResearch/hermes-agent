"""Regression probes from security review; real temporary Chromium only."""
import pytest
from tests.tools.test_browser_vault_signup_real import pytestmark
from tests.tools.test_browser_vault_signup_real import (
    signup_runtime, dispatch, leaf_doc, chrome, tab_tmp,
)


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('attribute,value', [('autocomplete','current-password'), ('name','differentPassword')])
def test_input_event_cannot_redirect_second_password(signup_runtime,monkeypatch,attribute,value):
    c=signup_runtime
    monkeypatch.setattr('tools.approval_prompt.request_elicitation_consent',lambda *a,**k:'accept')
    saved=dispatch('browser_vault_save_login',{'label':'Synthetic event probe'})
    monkeypatch.setattr('tools.approval_prompt.request_elicitation_consent',lambda *a,**k:'accept')
    wire,doc=leaf_doc(c)
    try:
        import json
        doc.evaluate("document.querySelector('input[name=password]').addEventListener('input',()=>document.querySelector('input[name=confirmPassword]').setAttribute("+json.dumps(attribute)+","+json.dumps(value)+"),{once:true});true")
        result=dispatch('browser_vault_fill',{'handle':saved['handle']})
        assert not result['success']
        assert result['filled_fields']==1
        assert doc.evaluate("document.querySelectorAll('input[type=password]')[1].value === ''")
    finally:
        wire.close()


@pytest.mark.live_system_guard_bypass
def test_takeover_during_consent_never_resolves(signup_runtime,monkeypatch):
    c=signup_runtime
    monkeypatch.setattr('tools.approval_prompt.request_elicitation_consent',lambda *a,**k:'accept')
    saved=dispatch('browser_vault_save_login',{'label':'Synthetic takeover'})
    from tools import browser_vault_tool as vault
    from tools.bot_desktop import lease
    monkeypatch.setattr(vault,'_bot_desktop_browser_session',lambda task:True)
    def takeover(*args,**kwargs):
        lease.acquire('synthetic-viewer')
        return 'accept'
    monkeypatch.setattr('tools.approval_prompt.request_elicitation_consent',takeover)
    from agent.vault_backends.local import LocalLoginBackend
    resolved=[]
    monkeypatch.setattr(LocalLoginBackend,'resolve_password',lambda *a:resolved.append(True))
    result=dispatch('browser_vault_fill',{'handle':saved['handle']})
    assert not result['success'] and resolved==[]
    lease.release('synthetic-viewer')
    wire,doc=leaf_doc(c)
    try:
        assert doc.evaluate("Array.from(document.querySelectorAll('input[type=password]')).every(e=>e.value==='')")
    finally:
        wire.close()


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('iframe', ['about:blank', 'data:text/html,<input type=password>'])
def test_login_with_unrelated_iframe_keeps_masked_save(signup_runtime,monkeypatch,iframe):
    c=signup_runtime
    from tools.browser_vault_document import OwnedWire
    wire = OwnedWire(c.supervisor)
    try:
        wire.document(wire.frames()[0]).evaluate("document.body.innerHTML='<form><input type=email value=fixture@example.test><input type=password autocomplete=current-password></form><iframe src=about:blank></iframe>';true")
        import json
        wire.document(wire.frames()[0]).evaluate('document.querySelector("iframe").src=' + json.dumps(iframe) + ';true')
    finally:
        wire.close()
    prompts=[]
    def secret_prompt(*args):
        prompts.append(args)
        return {'identifier':'fixture@example.test','password':'synthetic-ordinary-login-password!'}
    monkeypatch.setattr('agent.vault_backends.unlock.get_save_login_prompt_callback',lambda:secret_prompt)
    monkeypatch.setattr('agent.vault_backends.unlock.can_prompt_here',lambda:True)
    result=dispatch('browser_vault_save_login',{'label':'Synthetic ordinary login'})
    assert result['success'] is True and prompts
    assert c.store.get_meta(result['handle']).origin==c.root_origin
    assert dispatch('browser_vault_fill', {'handle': result['handle']})['success']
