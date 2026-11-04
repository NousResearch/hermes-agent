"""EP-13: a session verb the authority does not serve keeps its -32601 on an authority socket.

Authority sessions are never in the legacy ``server._sessions``, so the legacy handler failed with
``4001 session not found`` (``image.detach``, ``input.detect_drop``, ...) or ignored the session and
answered for the launch profile (``tools.list``, ``toolsets.list``).
"""
import asyncio
import types


def _call(monkeypatch, method, params):
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from tui_gateway import ws
    monkeypatch.setattr(ws, 'dispatch_legacy', lambda server, req, transport, actor: {'id': 1, 'result': 'legacy'})

    async def unknown(req):
        return {'id': 1, 'error': {'code': -32601, 'message': 'unknown method'}}

    authority = types.SimpleNamespace(dispatch=unknown, actor=types.SimpleNamespace(
        capabilities=frozenset(_PURPOSE_CAPABILITIES['interactive']), profile_id=''))
    req = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
    return asyncio.run(ws._dispatch_request(authority, req, method, types.SimpleNamespace()))


def test_session_verbs_on_authority_sessions_keep_unknown_method(monkeypatch):
    for method in ('clipboard.paste', 'handoff.fail', 'handoff.request', 'handoff.state', 'image.attach',
                   'image.detach', 'input.detect_drop', 'pdf.attach', 'terminal.resize', 'tools.configure',
                   'tools.list', 'toolsets.list'):
        assert _call(monkeypatch, method, {'session_id': 'authority-sid'})['error']['code'] == -32601, method


def test_host_wide_verbs_still_fall_back(monkeypatch):
    for method, params in (('tools.list', {}), ('pet.info', {}),
                           ('browser.manage', {'session_id': 'authority-sid', 'action': 'status'})):
        assert _call(monkeypatch, method, params)['result'] == 'legacy', method
