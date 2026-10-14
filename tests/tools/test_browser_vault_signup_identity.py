"""Registration refusal when identity changes; synthetic accounts only."""
import pytest
from tests.tools.test_browser_vault_signup_real import pytestmark
from tests.tools.test_browser_vault_signup_real import signup_runtime, dispatch, leaf_doc, chrome, tab_tmp


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('during_prompt', [False, True])
def test_registration_identity_must_match_before_and_after_consent(signup_runtime, monkeypatch, during_prompt):
    c=signup_runtime
    meta=c.store.add_item('login','Synthetic leaf', {'identifier_type':'email',
        'identifier':'fixture@example.test','password':c.password},origin=c.leaf_origin)
    from agent.vault_backends.local import LocalLoginBackend
    import tools.approval_prompt as prompt
    def do_not_resolve(*args):
        pytest.fail('credential resolved for wrong account')
    monkeypatch.setattr(LocalLoginBackend,'resolve_password',do_not_resolve)
    def change_identity():
        wire,doc=leaf_doc(c)
        try:
            doc.evaluate("document.querySelector('input[type=email]').value='different@example.test';true")
        finally:
            wire.close()
    def consent(*args,**kwargs):
        assert during_prompt, 'unexpected prompt for already mismatched account'
        change_identity()
        return 'accept'
    if not during_prompt:
        change_identity()
    monkeypatch.setattr(prompt,'request_elicitation_consent',consent)
    result=dispatch('browser_vault_fill',{'handle':meta.id})
    assert not result['success'] and result['error_type']=='identifier_mismatch'
