import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.browser_tab_ownership import OwnershipRegistry, OwnershipBusy


WS = 'ws://127.0.0.1:9222/devtools/browser/11111111-1111-1111-1111-111111111111'


@pytest.fixture
def registry(tmp_path):
    return OwnershipRegistry(tmp_path / 'ownership.sqlite')


def created(registry, owner='session:a', target='a', browser=WS):
    call = registry.admit(owner, 'generation-1', browser, owner)
    registry.record_created(call, target)
    registry.finish(call)
    return call


def test_unknown_owner_and_active_call_never_close(registry):
    call = registry.admit('session:a', 'g', WS, 'daemon')
    registry.record_created(call, 'a')
    assert registry.pending(WS) == []
    assert not registry.retire('session:a', 'g', evidence='fixture:terminal')
    registry.quarantine(call)
    registry.finish(call)  # caller exit does not drain a timed-out daemon
    assert not registry.retire('session:a', 'g', evidence='fixture:terminal')
    with pytest.raises(OwnershipBusy):
        registry.admit('session:a', 'g', WS, 'daemon')


def test_exact_holds_and_terminal_fence(registry):
    created(registry, target='a')
    created(registry, target='b')
    registry.hold(WS, 'a', 'child:sa_1')
    assert registry.retire('session:a', 'generation-1', evidence='fixture:terminal')
    assert [r['target'] for r in registry.pending(WS)] == ['b']
    with pytest.raises(OwnershipBusy):
        registry.hold(WS, 'b', 'late-child')
    with pytest.raises(OwnershipBusy):
        registry.admit('session:a', 'generation-1', WS, 'daemon')
    registry.release(WS, 'a', 'child:sa_1')
    assert {r['target'] for r in registry.pending(WS)} == {'a', 'b'}


def test_close_requires_absence_and_retries(registry):
    created(registry)
    registry.retire('session:a', 'generation-1', evidence='fixture:terminal')
    live = {'a', 'external'}
    attempts = []

    def close(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise TimeoutError('secret URL must not be recorded')
        live.remove(target)

    registry.cleanup(WS, lambda: live.copy(), close)
    assert registry.pending(WS)[0]['diagnostic'] == 'TimeoutError'
    registry.cleanup(WS, lambda: live.copy(), close)
    registry.cleanup(WS, lambda: live.copy(), close)
    assert attempts == ['a', 'a']
    assert live == {'external'}
    assert registry.pending(WS) == []


def test_generation_and_profile_isolation(registry, tmp_path):
    created(registry)
    registry.retire('session:a', 'generation-1', evidence='fixture:terminal')
    newer = WS.replace('11111111-', '22222222-', 1)
    assert registry.pending(newer) == []
    other = OwnershipRegistry(tmp_path / 'other.sqlite')
    assert other.pending(WS) == []
    assert len(registry.pending(WS)) == 1


def test_creation_is_immutable_and_needs_admission(registry):
    created(registry)
    call = registry.admit('session:b', 'g', WS, 'b')
    with pytest.raises(OwnershipBusy):
        registry.record_created(call, 'a')
    with pytest.raises(OwnershipBusy):
        registry.record_created('invented-token', 'unknown')
    registry.finish(call)
    registry.retire('session:b', 'g', evidence='fixture:terminal')
    assert registry.pending(WS) == []


def test_competing_admissions_are_serialized(registry):
    def enter(_):
        try:
            return registry.admit('session:a', 'g', WS, 'daemon')
        except OwnershipBusy:
            return None
    with ThreadPoolExecutor(2) as pool:
        assert sum(x is not None for x in pool.map(enter, range(2))) == 1


def test_restart_preserves_pending_and_calls(registry):
    created(registry)
    registry.retire('session:a', 'generation-1', evidence='fixture:terminal')
    registry.pending(WS)
    reopened = OwnershipRegistry(registry.path)
    assert reopened.pending(WS)[0]['target'] == 'a'
    with sqlite3.connect(registry.path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
