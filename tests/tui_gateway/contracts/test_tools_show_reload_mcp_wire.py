"""``tools.show`` / ``reload.mcp``: the declared wire must match what each transport serves.

The canonical owner (``gateway/session_ancillary.py``) routes on ``profile`` for both verbs and
refuses ``always`` on ``reload.mcp``; the standalone sidecar handler (``tui_gateway/methods_tools.py``)
honors ``always`` and ignores ``profile``. One shared session-scoped model cannot describe both
without lying to whoever generates a client from it, so ``tools.show`` gets its own params model and
``reload.mcp`` gains a canonical override.
"""
import pytest

from tui_gateway.contracts.registry import (
    CANONICAL_METHODS,
    METHODS,
    check_params_accepted,
    validate_params,
)


def test_tools_show_accepts_the_profile_the_owner_routes_on():
    fields = METHODS['tools.show'].params.model_fields
    assert set(fields) == {'session_id', 'profile'}
    assert validate_params(METHODS['tools.show'], {'session_id': 's'})[1] is None
    assert validate_params(METHODS['tools.show'], {'session_id': 's', 'profile': 'rockbot'})[1] is None
    # A handler that accepted it must not be reported as narrower than its contract.
    check_params_accepted(METHODS['tools.show'], {'session_id': 's', 'profile': 'rockbot'})
    # The key is still closed: an undeclared field is a client bug, not a silent ignore.
    assert 'always' in (validate_params(METHODS['tools.show'], {'session_id': 's', 'always': True})[1] or '')


def test_reload_mcp_legacy_model_keeps_always_and_gains_profile():
    fields = METHODS['reload.mcp'].params.model_fields
    assert fields['always'].default is False
    assert 'profile' in fields
    # The sidecar handler honors ``always``: its own model must keep declaring it.
    assert validate_params(METHODS['reload.mcp'], {'session_id': 's', 'confirm': True, 'always': True})[1] is None
    assert validate_params(METHODS['reload.mcp'], {'session_id': 's', 'profile': 'rockbot'})[1] is None


def test_canonical_reload_mcp_declares_profile_and_not_always():
    canonical = CANONICAL_METHODS['reload.mcp']
    assert set(canonical.params.model_fields) == {'session_id', 'confirm', 'rev', 'profile'}
    assert validate_params(canonical, {'session_id': 's', 'confirm': True, 'profile': 'rockbot'})[1] is None
    refused = validate_params(canonical, {'session_id': 's', 'confirm': True, 'always': True})[1]
    assert refused is not None and 'always' in refused
    # The canonical handler's answer for ``always`` is the contract's own refusal.
    check_params_accepted(canonical, {'session_id': 's', 'confirm': True, 'profile': 'rockbot'})


def test_the_shared_session_scoped_model_stays_narrow():
    for name in ('tools.list', 'toolsets.list'):
        assert set(METHODS[name].params.model_fields) == {'session_id'}, name
        assert 'profile' in (validate_params(METHODS[name], {'session_id': 's', 'profile': 'x'})[1] or ''), name


@pytest.mark.asyncio
async def test_canonical_dispatch_refuses_always_and_routes_profile(tmp_path):
    """The authority's own gate answers ``always`` with the contract's refusal (4001 invalid_params)."""
    from types import SimpleNamespace

    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(db_path=tmp_path / 'state.db')
    viewer = None
    try:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test', instance_id='test', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})
        refused = await viewer.dispatch({'id': 1, 'method': 'reload.mcp',
                                         'params': {'session_id': 's', 'confirm': True, 'always': True}})
        assert refused['error']['code'] == 4001
        assert refused['error']['data']['reason'] == 'invalid_params'
        assert 'always' in refused['error']['data']['fields']
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()
