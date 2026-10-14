"""Owned nested login/registration support for existing native Vault handlers.

Origin mismatch never resolves a secret. A separate save-login operation can
reuse a parent-bound password ONLY after explicit human approval to create a
new exact-origin item; the original binding is never changed.
"""
from __future__ import annotations

import json
import secrets

from agent.vault_login_classifier import (LoginControl, build_inspection_js,
    classify_login_control, select_password_fill)
from agent.vault_registration import registration_pair, confirm_registration
from tools.browser_vault_document import OwnedWire, password_documents


REFS = "window.__hermesVaultOriginal"


def inspect(doc):
    nonce = secrets.token_hex(16)
    expression = build_inspection_js(nonce)
    # Metadata and original references must come from the same renderer turn.
    controls = json.loads(doc.evaluate(f"(()=>{{const controls=({expression});const refs=Array.from(document.querySelectorAll('input, select'));{REFS}={{nonce:{json.dumps(nonce)},doc:document,refs,attrs:refs.map(e=>[e.name,e.id,e.getAttribute('autocomplete'),e.form])}};return controls;}})()"))
    return nonce, [LoginControl.from_dict(c) for c in controls]


def validate_js(nonce, indices, origin, *, require_empty=False):
    return """(() => {
      const s=window.__hermesVaultOriginal;
      const ids=__IDS__;
      if(location.origin!==__ORIGIN__ || !s || s.nonce!==__NONCE__ || s.doc!==document) return false;
      return ids.every(i=>{ const e=s.refs[i], a=s.attrs[i]; return e && a && e.isConnected && e.ownerDocument===document &&
        e.name===a[0] && e.id===a[1] && e.getAttribute('autocomplete')===a[2] && e.form===a[3] &&
        e.type==='password' && !e.disabled && !e.readOnly && e.getAttribute('data-hermes-vault-slot')===s.nonce+':'+i &&
        e.getClientRects().length>0 && getComputedStyle(e).visibility!=='hidden' &&
        (!__EMPTY__ || e.value===''); });
    })()""".replace('__IDS__', json.dumps(indices)).replace('__ORIGIN__', json.dumps(origin)).replace(
        '__NONCE__', json.dumps(nonce)).replace('__EMPTY__', 'true' if require_empty else 'false')


def fill_js(nonce, indices, origin, password, *, signup=False, identity_probe=None):
    guard = validate_js(nonce, indices, origin, require_empty=signup)
    if identity_probe is not None:
        guard = '(' + guard + ') && (' + identity_probe + ')'
    field_guard = validate_js(nonce, indices, origin)
    if signup:
        field_guard = '(' + field_guard + ') && ids.every((index,j)=>s.refs[index].value===(j<filled?value:""))'
    if identity_probe is not None:
        field_guard = '(' + field_guard + ') && (' + identity_probe + ')'
    return """(() => {
      if(!(__GUARD__)) return JSON.stringify({refused:'document_changed',filled:0});
      const s=window.__hermesVaultOriginal, ids=__IDS__, value=__PASSWORD__;
      let filled=0;
      for(const i of ids) {
        const e=s.refs[i];
        if(!(__FIELD_GUARD__))
          return JSON.stringify({refused:'field_changed',filled});
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,value);
        filled++;
        e.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertText'}));
        if(!(__FIELD_GUARD__)) return JSON.stringify({refused:'field_changed',filled});
        e.dispatchEvent(new Event('change',{bubbles:true}));
        if(!(__FIELD_GUARD__)) return JSON.stringify({refused:'field_changed',filled});
      }
      delete window.__hermesVaultOriginal;
      document.querySelectorAll('[data-hermes-vault-slot]').forEach(e=>e.removeAttribute('data-hermes-vault-slot'));
      return JSON.stringify({filled});
    })()""".replace('__GUARD__', guard).replace('__FIELD_GUARD__', field_guard).replace('__IDS__', json.dumps(indices)).replace('__PASSWORD__', json.dumps(password))


def registration_identity_js(nonce, controls, pair, identity):
    emails = [c for c in controls if c.type == 'email' and c.form_index == pair[0].form_index]
    if len(emails) != 1 or not identity:
        return None
    return """(() => {
      const s=window.__hermesVaultOriginal, i=__INDEX__;
      if(!s || s.nonce!==__NONCE__ || s.doc!==document) return false;
      const e=s.refs[i], a=s.attrs[i], p=s.refs[__PASSWORD_INDEX__];
      return !!e && !!a && !!p && e.isConnected && e.ownerDocument===document &&
        e.type==='email' && e.form===p.form && e.form===a[3] && e.name===a[0] && e.id===a[1] &&
        e.value===__IDENTITY__;
    })()""".replace('__INDEX__', str(emails[0].index)).replace('__NONCE__',json.dumps(nonce)).replace(
        '__PASSWORD_INDEX__',str(pair[0].index)).replace('__IDENTITY__',json.dumps(identity))


def top_level_login(wire):
    """An ordinary root login does not depend on unrelated child documents."""
    root = wire.document(wire.frames()[0])
    nonce, controls = inspect(root)
    classified = [c for x in controls if (c := classify_login_control(x)) is not None]
    if not registration_pair(controls) and select_password_fill(classified, 'selection-only'):
        return root, nonce, controls
    return None


def nested_fill(supervisor, backend, meta):
    wire = OwnedWire(supervisor)
    try:
        ordinary = top_level_login(wire)
        docs = [ordinary[0]] if ordinary else password_documents(wire)
        allowed = set(meta.allowed_origins or ([meta.origin] if meta.origin else []))
        matching = [d for d in docs if d.live_origin() in allowed]
        if len(matching) != 1:
            return {'success':False,'error_type':'frame_origin_mismatch' if docs and not matching else 'form_ambiguous',
                    'error':'No unique password document on the saved exact origin.',
                    'field_origins':sorted({d.origin for d in docs})}
        doc = matching[0]
        nonce, controls = ordinary[1:] if ordinary else inspect(doc)
        pair = registration_pair(controls)
        identity_probe = None
        if pair:
            identity_probe = registration_identity_js(nonce, controls, pair, meta.identifier)
            if identity_probe is None or doc.evaluate(identity_probe) is not True:
                return {'success':False,'error_type':'identifier_mismatch','error':'Registration account does not match the saved login.'}
            if not confirm_registration(meta.label, doc.origin):
                return {'success':False,'error_type':'registration_declined','error':'Registration fill was not approved.'}
            indices = [c.index for c in pair]
        else:
            classified = [c for x in controls if (c:=classify_login_control(x)) is not None]
            # Selection uses no credential bytes during discovery.
            selected = select_password_fill(classified, 'selection-only')
            indices = [x['index'] for x in selected]
        if not indices:
            return {'success':False,'error_type':'no_password_field','error':'No supported password field.'}
        if pair and doc.evaluate(identity_probe) is not True:
            return {'success':False,'error_type':'identifier_mismatch','error':'Registration account changed during approval.'}
        if not doc.evaluate(validate_js(nonce, indices, doc.origin, require_empty=bool(pair))):
            return {'success':False,'error_type':'document_changed','error':'Original password fields changed.'}
        if doc.live_origin() not in allowed:
            return {'success':False,'error_type':'origin_mismatch','error':'Document origin changed.'}
        wire.check_profile()
        wire.check()
        password = backend.resolve_password(meta.id)
        from agent.redact import register_vault_redaction_value
        register_vault_redaction_value(password)
        try:
            result = json.loads(doc.evaluate(fill_js(nonce, indices, doc.origin, password, signup=bool(pair), identity_probe=identity_probe)))
        except Exception:
            return {'success':False,'error_type':'document_changed','error':'Protected document fill refused.'}
        finally:
            del password
        count = result.get('filled', 0)
        return {'success':count==len(indices) and not result.get('refused'), 'filled_fields':count,
                'kind':'login', 'origin':doc.origin, 'registration':bool(pair)}
    finally:
        wire.close()


def save_nested_registration(supervisor, label):
    """Metadata-only discovery, then separately approved server-side password reuse."""
    wire = OwnedWire(supervisor)
    try:
        if top_level_login(wire):
            return None
        docs = password_documents(wire)
        root=wire.document(wire.frames()[0])
        parent_origin=root.live_origin()
        # A normal top-level login plus an unrelated iframe must retain its
        # existing masked save flow. Choose registration by actual controls.
        inspected = [(doc, *inspect(doc)) for doc in docs]
        registrations = [(doc, nonce, controls, registration_pair(controls))
                         for doc, nonce, controls in inspected if registration_pair(controls)]
        if not registrations:
            if not docs or any(doc.frame == root.frame for doc in docs):
                return None
            return {'success':False,'error_type':'nested_login_unsupported','error':'No supported nested registration pair.'}
        if len(registrations)!=1 or len(docs)!=1:
            return {'success':False,'error_type':'form_ambiguous','error':'No unique registration document.'}
        doc, nonce, controls, pair = registrations[0]
        assert pair is not None
        if doc.origin == parent_origin:
            return None
        identity=doc.evaluate("document.querySelector('input[type=email]')?.value || ''")
        identity_probe=registration_identity_js(nonce,controls,pair,identity)
        if identity_probe is None or doc.evaluate(identity_probe) is not True:
            return {'success':False,'error_type':'identifier_mismatch','error':'No unique original registration identity.'}
        from agent.vault_backends import enabled_backends
        sources=[]
        for backend in enabled_backends():
            for meta in backend.list_items():
                if meta.kind=='login' and meta.identifier==identity and parent_origin in set(meta.allowed_origins or [meta.origin]):
                    sources.append((backend,meta))
        if len(sources)!=1 or doc.origin==parent_origin:
            return {'success':False,'error_type':'saved_source_unavailable','error':'No unique parent-bound saved login for this account.'}
        backend, source=sources[0]
        from tools.approval_prompt import request_elicitation_consent
        consent=request_elicitation_consent(
            f"Save login for {identity} on {doc.origin} using the saved password for {parent_origin}",
            "Create a NEW encrypted Vault item for this exact form origin using the existing saved password. "
            "The original site's binding remains unchanged. No password is shown and no registration is submitted. "
            "Approve only if you intend to reuse this password on the stated destination.",
            surface='vault-registration-save',title='Confirm exact-origin credential save?')
        if consent!='accept':
            return {'success':False,'error_type':'save_declined','error':'Credential reuse was not approved.'}
        indices=[c.index for c in pair]
        if (root.live_origin()!=parent_origin or doc.live_origin()!=doc.origin or
            doc.evaluate(identity_probe) is not True or
            not doc.evaluate(validate_js(nonce,indices,doc.origin,require_empty=True))):
            return {'success':False,'error_type':'document_changed','error':'Registration changed before save.'}
        wire.check_profile()
        wire.check()
        password=backend.resolve_password(source.id)
        from agent.vault_store import get_vault_store
        from agent.redact import register_vault_redaction_value
        register_vault_redaction_value(password)
        try:
            wire.check()
            wire.check_profile()
            if (root.live_origin()!=parent_origin or doc.live_origin()!=doc.origin or doc.evaluate(identity_probe) is not True or
                    not doc.evaluate(validate_js(nonce,indices,doc.origin,require_empty=True))):
                return {'success':False,'error_type':'document_changed','error':'Registration changed before encrypted save.'}
            item=get_vault_store().add_item('login',label or 'Registration login',
                {'identifier':identity,'identifier_type':source.identifier_type or 'email','password':password}, origin=doc.origin)
        except Exception:
            return {'success':False,'error_type':'save_failed','error':'Encrypted credential save failed.'}
        finally:
            del password
        # Verify metadata readback before claiming that the store write succeeded.
        saved=get_vault_store().get_meta(item.id)
        if saved is None or saved.origin!=doc.origin or saved.identifier!=identity:
            return {'success':False,'error_type':'save_unknown','error':'Credential metadata readback failed.'}
        return {'success':True,'saved':True,'filled':False,'handle':item.id,'origin':item.origin}
    finally:
        wire.close()
