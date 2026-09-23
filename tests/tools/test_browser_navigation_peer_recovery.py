"""Failed navigation must retain peer evidence and the previous task owner."""
import json
import pytest
from tools import browser_tool as bt, browser_supervisor as bs
from tools import browser_tool_cloud as cloud, browser_tool_session as session


@pytest.mark.parametrize('case', ['open_exception', 'blank_exception', 'snapshot_exception', 'snapshot_peer'])
def test_failed_navigation_retains_observation_and_owner(monkeypatch, case):
    url = 'https://public.example/'
    supervisor = bs.CDPSupervisor(task_id='recovery', cdp_url='ws://127.0.0.1/inert')
    def record():
        supervisor._on_network_response_received({'response': {'remoteIPAddress': '127.0.0.1', 'url': url}})
    if case in {'open_exception', 'blank_exception'}:
        record()
    calls = []
    def command(task, action, args, **kwargs):
        calls.append((action, args))
        if action == 'open':
            if args == ['about:blank']:
                if case == 'blank_exception':
                    raise RuntimeError('blank failed')
                return {'success': True}
            if case in {'open_exception', 'blank_exception'}:
                raise RuntimeError('PRIVATE_ERROR')
            return {'success': True, 'data': {'url': url, 'title': 'PRIVATE_TITLE'}}
        record()
        if case == 'snapshot_exception':
            raise RuntimeError('PRIVATE_SNAPSHOT')
        return {'success': True, 'data': {'snapshot': 'PRIVATE_BODY'}}
    monkeypatch.setitem(bt._last_active_session_key, 'recovery', 'previous-session')
    for obj, name, value in [
        (bs.SUPERVISOR_REGISTRY, 'get', lambda _: supervisor),
        (supervisor, 'flush_network_events', lambda: True),
        (bt, '_is_camofox_mode', lambda: False),
        (bt, '_navigation_session_key', lambda task, url: task),
        (bt, '_url_policy_error', lambda *a, **k: None),
        (bt, '_secret_url_error_normalized', lambda u: (u, None)),
        (bt._cdp, '_ensure_cdp_supervisor', lambda *a: None),
        (cloud, '_is_local_backend', lambda: False),
        (cloud, '_allow_private_urls', lambda: False),
        (bt, '_is_local_sidecar_key', lambda *a: False),
        (session, '_get_session_info', lambda _: {'_first_nav': False}),
        (session, '_run_browser_command', command),
    ]:
        monkeypatch.setattr(obj, name, value)
    result = json.loads(bt.browser_navigate(url, 'recovery'))
    assert result['success'] is False
    assert 'PRIVATE_' not in json.dumps(result)
    assert bt._last_active_session_key['recovery'] == 'previous-session'
    assert ('open', ['about:blank']) in calls
    if case == 'blank_exception':
        assert supervisor.snapshot().network_responses
