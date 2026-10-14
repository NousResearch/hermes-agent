"""Real isolated Chromium + registered native Vault + cross-origin signup."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tests.tools.test_browser_tab_real_chromium import tab_tmp, chrome
from tests.waiting import wait_until

pytestmark = [pytest.mark.platforms('linux'),
              pytest.mark.skipif(os.environ.get('HERMES_E2E_BROWSER') != '1',
                                 reason='set HERMES_E2E_BROWSER=1 for disposable Chromium')]


@pytest.fixture(params=['same-site','oopif'])
def signup_runtime(chrome, tab_tmp, monkeypatch, request):
    from tools import browser_vault_tool as vault
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    from tools.browser_tab_ownership import OwnershipRegistry
    from agent.vault_store import get_vault_store
    browser, ws = chrome
    home = tab_tmp / 'hermes-signup'
    home.mkdir()
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(vault, '_bot_desktop_browser_session', lambda task: False)
    fixture_email = 'fixture@example.test'
    fixture_password = 'synthetic-signup-password!'
    leaf_origin = ''
    class Page(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/root':
                body = '<html><body><iframe style="width:600px;height:400px" src="' + leaf_origin + '/signup"></iframe></body></html>'
            else:
                body = '<html><body><form><input type="email" value="fixture@example.test"><label>Password<input type="password" name="password" autocomplete="new-password"></label><label>Confirm Password<input type="password" name="confirmPassword"></label></form></body></html>'
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.end_headers()
            self.wfile.write(body.encode())
        def log_message(self,format,*args):
            pass
    root_server = ThreadingHTTPServer(('127.0.0.1',0),Page)
    leaf_server = ThreadingHTTPServer(('127.0.0.1',0),Page)
    root_origin = f'http://127.0.0.1:{root_server.server_port}'
    leaf_host = 'localhost' if request.param == 'oopif' else '127.0.0.1'
    leaf_origin = f'http://{leaf_host}:{leaf_server.server_port}'
    threads = [threading.Thread(target=s.serve_forever,daemon=True) for s in (root_server,leaf_server)]
    for t in threads:
        t.start()
    r = OwnershipRegistry(home / 'browser_tabs.sqlite')
    token = r.admit('session:signup-test','signup-test',ws,'signup-daemon')
    target = browser.command('Target.createTarget',url=root_origin+'/root')['targetId']
    r.record_created(token,target)
    r.publish_current(token,target)
    r.finish(token)
    sup = SUPERVISOR_REGISTRY.get_or_start('signup-test',ws)
    sup._vault_owned_call = (str(r.path),token)

    source = get_vault_store().add_item('login','Synthetic source',{'identifier_type':'email',
        'identifier':fixture_email,'password':fixture_password},origin=root_origin)
    from tools.browser_vault_document import OwnedWire, password_documents
    def loaded():
        wire = OwnedWire(sup)
        try:
            return bool(password_documents(wire))
        finally:
            wire.close()
    context = SimpleNamespace(browser=browser, supervisor=sup, source=source,
        root_origin=root_origin, leaf_origin=leaf_origin, password=fixture_password, store=get_vault_store())
    try:
        wait_until(loaded, timeout=30, what='synthetic signup document')
        yield context
    finally:
        SUPERVISOR_REGISTRY.stop('signup-test')
        root_server.shutdown(); leaf_server.shutdown()
        root_server.server_close(); leaf_server.server_close()
        for t in threads:
            t.join(timeout=5)


def dispatch(name,args):
    from tools.registry import registry
    result = registry.dispatch('browser_vault', {'action': name.removeprefix('browser_vault_'), **args},task_id='signup-test')
    return json.loads(result) if isinstance(result,str) else result


def leaf_doc(ctx):
    from tools.browser_vault_document import OwnedWire,password_documents
    wire=OwnedWire(ctx.supervisor)
    docs=password_documents(wire)
    assert len(docs)==1
    return wire,docs[0]


@pytest.mark.live_system_guard_bypass
def test_cross_origin_requires_separate_explicitly_approved_save(signup_runtime,monkeypatch):
    c=signup_runtime
    import tools.approval_prompt as prompt
    from agent.vault_backends.local import LocalLoginBackend
    calls=[]
    resolve=LocalLoginBackend.resolve_password
    def recorded(self,handle):
        calls.append(handle)
        return resolve(self,handle)
    monkeypatch.setattr(LocalLoginBackend,'resolve_password',recorded)
    denied=dispatch('browser_vault_fill',{'handle':c.source.id})
    assert denied['error_type']=='frame_origin_mismatch'
    assert calls==[]
    monkeypatch.setattr(prompt,'request_elicitation_consent',lambda *a,**k:'decline')
    declined=dispatch('browser_vault_save_login',{'label':'Synthetic signup'})
    assert not declined['success'] and calls==[]
    assert len(c.store.list_items())==1
    monkeypatch.setattr(prompt,'request_elicitation_consent',lambda *a,**k:'accept')
    saved=dispatch('browser_vault_save_login',{'label':'Synthetic signup'})
    assert saved['success'] and saved['saved'] and not saved['filled']
    assert saved['origin']==c.leaf_origin
    assert c.store.get_meta(c.source.id).origin==c.root_origin
    assert c.store.get_meta(saved['handle']).origin==c.leaf_origin
    filled=dispatch('browser_vault_fill',{'handle':saved['handle']})
    assert filled['success'] and filled['filled_fields']==2 and filled['registration']
    assert c.password not in json.dumps(saved)+json.dumps(filled)
    wire,doc=leaf_doc(c)
    try:
        assert doc.evaluate("[...document.querySelectorAll('input[type=password]')].every(e=>e.value==='synthetic-signup-password!')") is True
    finally:
        wire.close()


@pytest.mark.live_system_guard_bypass
def test_original_field_replacement_refuses_before_resolution(signup_runtime,monkeypatch):
    c=signup_runtime
    leaf=c.store.add_item('login','Synthetic leaf',{'identifier_type':'email',
        'identifier':'fixture@example.test','password':c.password},origin=c.leaf_origin)
    import tools.approval_prompt as prompt
    from agent.vault_backends.local import LocalLoginBackend
    def must_not_resolve(*args):
        pytest.fail('secret resolved after original field replacement')
    monkeypatch.setattr(LocalLoginBackend,'resolve_password',must_not_resolve)
    def consent(*args,**kwargs):
        wire,doc=leaf_doc(c)
        try:
            doc.evaluate("(()=>{let e=document.querySelector('input[name=confirmPassword]');e.replaceWith(e.cloneNode(true));return true})()")
        finally:
            wire.close()
        return 'accept'
    monkeypatch.setattr(prompt,'request_elicitation_consent',consent)
    result=dispatch('browser_vault_fill',{'handle':leaf.id})
    assert not result['success'] and result['error_type']=='document_changed'


@pytest.mark.live_system_guard_bypass
def test_wire_change_refuses_original_document(signup_runtime):
    c=signup_runtime
    from tools.browser_vault_document import DocumentRefused
    wire,doc=leaf_doc(c)
    old=c.supervisor._ws
    c.supervisor._ws=object()
    try:
        with pytest.raises(DocumentRefused):
            doc.evaluate('true')
    finally:
        c.supervisor._ws=old
        wire.close()
