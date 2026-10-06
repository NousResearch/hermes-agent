"""Durable capture admission without lifecycle inference or tab deletion."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry

WS = 'ws://127.0.0.1:9222/devtools/browser/fixture'


def test_competing_admissions_are_serialized(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    def enter(_):
        try:
            return registry.admit('owner', 'g', WS, 'daemon')
        except OwnershipBusy:
            return None
    with ThreadPoolExecutor(2) as pool:
        assert sum(token is not None for token in pool.map(enter, range(2))) == 1


def test_creation_requires_admission_and_is_immutable(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    a = registry.admit('a', 'g', WS, 'a')
    registry.record_created(a, 'target')
    registry.record_created(a, 'target')
    registry.finish(a)
    b = registry.admit('b', 'g', WS, 'b')
    with pytest.raises(OwnershipBusy):
        registry.record_created(b, 'target')
    with pytest.raises(OwnershipBusy):
        registry.record_created('invented', 'foreign')
    with pytest.raises(OwnershipBusy):
        registry.record_created(a, 'after-drain')
    assert registry.owned(a) == ['target']
    assert registry.owned(b) == []


def test_profile_and_browser_identity_isolation(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'a.sqlite')
    token = registry.admit('owner', 'g', WS, 'a')
    registry.record_created(token, 'target')
    registry.finish(token)
    newer = registry.admit('owner', 'g', WS + '-new', 'b')
    assert registry.owned(newer) == []
    other = OwnershipRegistry(tmp_path / 'b.sqlite')
    assert other.owned(other.admit('owner', 'g', WS, 'a')) == []
    reopened = OwnershipRegistry(registry.path)
    assert reopened.owned(token) == ['target']
    with sqlite3.connect(registry.path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_daemon_binding_cannot_change_within_call(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'a')
    registry.bind_daemon(token, 123, 'start')
    registry.bind_daemon(token, 123, 'start')
    with pytest.raises(OwnershipBusy):
        registry.bind_daemon(token, 123, 'recycled')
    registry.finish(token)
    with pytest.raises(OwnershipBusy):
        registry.bind_daemon(token, 123, 'start')
