"""Mission-scoped entry selection against a real temporary ownership ledger."""
import sys
from types import ModuleType

import pytest

from tools import browser_tab_capture as capture
from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry

WS = 'ws://127.0.0.1:9222/devtools/browser/fixture'


def harness(monkeypatch, live):
    helpers = ModuleType('browser_harness.helpers')
    attached = []
    requests = []

    def send(req, *args, **kwargs):
        requests.append(req)
        method = req.get('method')
        if method == 'Target.getTargets':
            return {'result': {'targetInfos': [
                {'targetId': target, 'type': 'page'} for target in live]}}
        if method == 'Target.createTarget':
            live.add('fresh')
            return {'result': {'targetId': 'fresh'}}
        if method == 'Target.attachToTarget':
            attached.append(req['params']['targetId'])
            return {'result': {'sessionId': 'fixture-session'}}
        if req.get('meta') == 'set_session':
            return {'session_id': req['session_id']}
        return {'result': {}}

    helpers._send = send
    helpers.cdp = lambda method, **params: helpers._send(
        {'method': method, 'params': params})['result']
    package = ModuleType('browser_harness')
    package.helpers = helpers
    monkeypatch.setitem(sys.modules, 'browser_harness', package)
    monkeypatch.setitem(sys.modules, 'browser_harness.helpers', helpers)
    monkeypatch.setattr(capture, '_verify_daemon', lambda *args: None)
    return attached, requests


@pytest.mark.parametrize('existing_b', [False, True])
def test_entry_never_attaches_quarantined_same_owner_other_mission(tmp_path, monkeypatch, existing_b):
    path = tmp_path / 'tabs.sqlite'
    registry = OwnershipRegistry(path)
    a = registry.admit('owner', 'g', WS, 'mission-a')
    registry.record_created(a, 'a')
    registry.request_started(a)
    registry.quarantine(a)
    live = {'a', 'external'}
    if existing_b:
        previous_b = registry.admit('owner', 'g', WS, 'mission-b')
        registry.record_created(previous_b, 'b')
        registry.finish(previous_b)
        live.add('b')
    b = registry.admit('owner', 'g', WS, 'mission-b')
    attached, _ = harness(monkeypatch, live)
    capture.begin_exec(path, b)
    expected = 'b' if existing_b else 'fresh'
    assert attached == [expected]
    assert registry.call(a)['state'] == 'quarantined'
    assert registry.call(a)['inflight'] == 1
    # Broad owner ownership remains available to lifecycle/cleanup callers.
    assert set(registry.owned(b)) == {'a', expected}
    registry.finish(b)
    again = registry.admit('owner', 'g', WS, 'mission-b')
    attached, _ = harness(monkeypatch, live)  # fresh helpers on each exec
    capture.begin_exec(path, again)
    assert attached == [expected]


@pytest.mark.parametrize('owner,generation,browser,daemon', [
    ('foreign', 'g', WS, 'mission-b'),
    ('owner', 'old', WS, 'mission-b'),
    ('owner', 'g', WS + '-other', 'mission-b'),
    ('owner', 'g', WS, 'mission-a'),
])
def test_entry_does_not_reuse_foreign_creation(tmp_path, monkeypatch, owner, generation, browser, daemon):
    path = tmp_path / 'tabs.sqlite'
    registry = OwnershipRegistry(path)
    foreign = registry.admit(owner, generation, browser, daemon)
    registry.record_created(foreign, 'foreign')
    registry.finish(foreign)
    token = registry.admit('owner', 'g', WS, 'mission-b')
    attached, _ = harness(monkeypatch, {'foreign', 'external'})
    capture.begin_exec(path, token)
    assert attached == ['fresh']


@pytest.mark.parametrize('state', ['drained', 'quarantined'])
def test_entry_rejects_nonactive_call_before_browser_requests(tmp_path, monkeypatch, state):
    path = tmp_path / 'tabs.sqlite'
    registry = OwnershipRegistry(path)
    token = registry.admit('owner', 'g', WS, 'mission')
    if state == 'drained':
        registry.finish(token)
    else:
        registry.quarantine(token)
    _, requests = harness(monkeypatch, {'external'})
    with pytest.raises(OwnershipBusy, match='call is not active'):
        capture.begin_exec(path, token)
    assert requests == []
