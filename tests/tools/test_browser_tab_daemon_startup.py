"""Daemon-created startup blanks need creation proof, not attachment inference."""
import sys
from types import ModuleType

import pytest

from tools import browser_tab_capture as capture
from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry
from tests.tools.test_browser_tab_capture import WS, harness


def setup(tmp_path, monkeypatch, receipt, *, url='about:blank', bound=True):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'mission')
    if bound:
        registry.bind_daemon(token, 123, '123-start')
    attached, requests = harness(monkeypatch, {'startup', 'external'})
    helpers = sys.modules['browser_harness.helpers']
    original = helpers._send

    def send(req, *args, **kwargs):
        if req.get('meta') == 'connection_status':
            requests.append(req)
            return {'target_id': 'startup', 'session_id': 'startup-session', 'page': {'targetId': 'startup'}}
        if req.get('method') == 'Target.getTargetInfo':
            requests.append(req)
            assert req['params']['targetId'] == 'startup'
            return {'result': {'targetInfo': {'targetId': 'startup', 'type': 'page', 'url': url}}}
        return original(req, *args, **kwargs)

    helpers._send = send
    ipc = ModuleType('browser_harness._ipc')
    log = tmp_path / 'mission.log'
    log.write_text(receipt)
    ipc.log_path = lambda name: log
    sys.modules['browser_harness']._ipc = ipc
    monkeypatch.setitem(sys.modules, 'browser_harness._ipc', ipc)
    return registry, token, attached, requests


def test_exact_startup_creation_reused_and_captured(tmp_path, monkeypatch):
    registry, token, attached, requests = setup(tmp_path, monkeypatch,
        'named daemon mission: created dedicated tab (startup)\n')
    capture.begin_exec(registry.path, token)
    assert registry.owned_for_daemon(token) == ['startup']
    assert attached == []  # preserve the daemon's already enabled CDP session
    assert not any(r.get('method') == 'Target.createTarget' for r in requests)
    assert registry.call(token)['inflight'] == 0


@pytest.mark.parametrize('receipt,url,bound', [
    ('', 'about:blank', True),
    ('named daemon foreign: created dedicated tab (startup)\n', 'about:blank', True),
    ('named daemon mission: created dedicated tab (different)\n', 'about:blank', True),
    ('prefix named daemon mission: created dedicated tab (startup)\n', 'about:blank', True),
    ('named daemon mission: created dedicated tab (startup)\n', 'https://signed-in.test/', True),
    ('named daemon mission: created dedicated tab (startup)\n', 'about:blank', False),
])
def test_attachment_without_exact_creation_proof_never_adopted(tmp_path, monkeypatch, receipt, url, bound):
    registry, token, attached, _ = setup(tmp_path, monkeypatch, receipt, url=url, bound=bound)
    capture.begin_exec(registry.path, token)
    assert registry.owned_for_daemon(token) == ['fresh']
    assert attached == ['fresh']


def test_restart_startup_registered_but_existing_continuation_reattached(tmp_path, monkeypatch):
    registry, token, attached, requests = setup(tmp_path, monkeypatch,
        'named daemon mission: created dedicated tab (startup)\n')
    registry.record_created(token, 'external')  # surviving exact mission creation
    capture.begin_exec(registry.path, token)
    assert registry.owned_for_daemon(token) == ['external', 'startup']
    assert attached == ['external']
    assert not any(r.get('method') == 'Target.createTarget' for r in requests)


def test_foreign_ledger_target_cannot_be_claimed_by_startup_receipt(tmp_path, monkeypatch):
    registry, token, attached, _ = setup(tmp_path, monkeypatch,
        'named daemon mission: created dedicated tab (startup)\n')
    foreign = registry.admit('other', 'g', WS, 'other-mission')
    registry.record_created(foreign, 'startup')
    registry.quarantine(foreign)
    with pytest.raises(OwnershipBusy, match='already attributed'):
        capture.begin_exec(registry.path, token)
    assert attached == []
    assert registry.call(token)['state'] == 'quarantined'
    assert registry.call(foreign)['state'] == 'quarantined'


@pytest.mark.parametrize('release', ['held', 'failed'])
def test_released_startup_target_allows_fresh_navigation(tmp_path, monkeypatch, release):
    receipt = 'named daemon mission: created dedicated tab (startup)\n'
    registry, token, _, _ = setup(tmp_path, monkeypatch, receipt)
    capture.begin_exec(registry.path, token)
    registry.finish(token)
    if release == 'held':
        registry.hold(WS, 'startup', 'live-actor')
    registry.request_owner_close('owner', 'g')
    if release == 'failed':
        def failed_close(target):
            raise TimeoutError('close reply lost')
        registry.cleanup(WS, lambda: {'startup'}, failed_close)
    registry, next_call, attached, _ = setup(tmp_path, monkeypatch, receipt)
    capture.begin_exec(registry.path, next_call)
    assert attached == ['fresh']
    assert registry.owned_for_daemon(next_call) == ['fresh']
    assert registry.call(next_call)['state'] == 'active'
    registry.finish(next_call)
    assert registry.call(next_call)['state'] == 'drained'
