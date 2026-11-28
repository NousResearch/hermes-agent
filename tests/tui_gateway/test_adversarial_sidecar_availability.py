"""Native interactive viewers retain registered presentation sidecars."""
from types import SimpleNamespace
import pytest
from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from tui_gateway import server
from tui_gateway.ws import _dispatch_request
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

@pytest.mark.asyncio
# The two namespaces only this test pins: the legacy fallback gate's own tests do not cover them.
@pytest.mark.parametrize('method,params', [('pet.info.meta', {}), ('projects.list', {})])
async def test_registered_presentation_sidecar_remains_available_on_canonical_transport(method, params, tmp_path, monkeypatch):
    monkeypatch.setattr(server, '_LONG_HANDLERS', server._LONG_HANDLERS - {method})
    home = str(server._launch_home())
    store = storage_handle(tmp_path)
    authority = SessionAuthority(SimpleNamespace(), profile_id=home, instance_id='review', persistence=store,
                                 epoch=store.runtime.begin_epoch(instance_id='review'))
    viewer = AuthorityConnection(authority, None, {'user_id': 'human', 'profile_id': home,
        'capabilities': _PURPOSE_CAPABILITIES['interactive']})
    try:
        req = {'jsonrpc': '2.0', 'id': 7, 'method': method, 'params': params}
        old = await _dispatch_request(None, req, method, None)
        assert old is not None and old.get('error', {}).get('code') != -32601
        result = await _dispatch_request(viewer, req, method, None)
        assert result is not None and result.get('error', {}).get('code') != -32601
    finally:
        await viewer.close()
