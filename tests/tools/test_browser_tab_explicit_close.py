"""Explicit release through native registry; no production browser or Hermes home."""
import json
import os

import pytest

from tools import browser_use_cli  # registers browser_exec
from tools.browser_tab_lifecycle import close_owned, sweep
from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry
from tools.registry import registry
from hermes_state import SessionDB
from tests.tools.test_browser_tab_real_chromium import chrome, tab_tmp, harness_cli, stop_test_daemons


@pytest.fixture
def scope(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    db = SessionDB(home / 'state.db')
    db.create_session('a', 'cli')
    db.create_session('b', 'cli')
    r = OwnershipRegistry(home / 'browser_tabs.sqlite')
    yield home, db, r
    db.close()


def create(r, ws, target, owner='session:a', generation='a', daemon='named'):
    token = r.admit(owner, generation, ws, daemon)
    r.record_created(token, target)
    r.finish(token)
    return token


def dispatch(**args):
    return json.loads(registry.get_entry('browser_exec').handler(args, session_id='a', task_id='a'))


@pytest.mark.platforms("linux")
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER") != "1", reason="requires HERMES_E2E_BROWSER=1")
def test_native_registered_close_and_same_owner_reentry(scope, chrome, monkeypatch):
    home, db, r = scope
    browser, ws = chrome
    unknown = browser.targets()
    a = browser.command('Target.createTarget', url='about:blank')['targetId']
    b = browser.command('Target.createTarget', url='about:blank')['targetId']
    create(r, ws, a)
    create(r, ws, b, 'session:b', 'b', 'other')
    # Not even CLI discovery/routing should run for close; no startup blank.
    monkeypatch.setattr(browser_use_cli, '_find_cli', lambda: pytest.fail('spawn attempted'))
    result = dispatch(action='close_owned')
    assert result['success'] and result['closed'] == 1 and result['remaining'] == 0
    assert browser.targets() == unknown | {b}
    assert db.get_session('a')['ended_at'] is None
    assert dispatch(action='close_owned')['closed'] == 0
    # Reentry can create resources under the SAME compression-lineage owner.
    c = browser.command('Target.createTarget', url='about:blank')['targetId']
    create(r, ws, c)
    assert dispatch(action='close_owned')['closed'] == 1
    assert browser.targets() == unknown | {b}


@pytest.mark.parametrize('state', ['active', 'quarantined'])
def test_busy_owner_is_not_marked(scope, state):
    home, _, r = scope
    token = r.admit('session:a', 'a', 'ws', 'd')
    r.record_created(token, 'a')
    if state == 'quarantined':
        r.quarantine(token)
    result = dispatch(action='close_owned')
    assert not result['success'] and result['reason'] == 'owner_busy'
    with r._db() as db:
        assert db.execute('SELECT pending_close FROM targets').fetchone()[0] == 0


@pytest.mark.platforms("linux")
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER") != "1", reason="requires HERMES_E2E_BROWSER=1")
def test_held_close_survives_restart_and_retries(scope, chrome):
    home, _, r = scope
    browser, ws = chrome
    target = browser.command('Target.createTarget', url='about:blank')['targetId']
    create(r, ws, target)
    r.hold(ws, target, 'opaque-live-actor')
    result = close_owned(home, session_id='a')
    assert not result['success'] and result['remaining'] == 1 and result['closed'] == 0
    assert target in browser.targets()
    r = OwnershipRegistry(r.path)
    r.release(ws, target, 'opaque-live-actor')
    sweep(home)
    assert target not in browser.targets()


def test_disable_unknown_and_code_do_not_fallback(scope, monkeypatch):
    home, _, r = scope
    assert close_owned(home)['reason'] == 'owner_unresolved'
    assert 'error' in dispatch(action='close_owned', code="print('must not execute')")
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: false\n')
    assert dispatch(action='close_owned')['reason'] == 'ownership_disabled'


def test_close_failure_keeps_pending_and_other_retired_owner_is_untouched(scope):
    home, _, r = scope
    ws = 'ws://127.0.0.1:9222/devtools/browser/exact'
    create(r, ws, 'a')
    create(r, ws, 'b', 'session:b', 'b', 'b')
    r.retire('session:b', 'b', evidence='fixture:terminal')
    class Browser:
        socket = None
        def __init__(self, ws):
            self.socket = self
        def targets(self):
            return {'a', 'b'}
        def close_target(self, target):
            assert target == 'a'
            raise TimeoutError('do not expose endpoint')
        def close(self):
            pass
    result = close_owned(home, session_id='a', connect=Browser)
    assert result['closed'] == 0 and result['remaining'] == 1
    with r._db() as db:
        assert db.execute("SELECT diagnostic FROM targets WHERE target='a'").fetchone()[0] == 'TimeoutError'
        assert db.execute("SELECT pending_close FROM targets WHERE target='b'").fetchone()[0] == 0


@pytest.mark.live_system_guard_bypass
@pytest.mark.platforms("linux")
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER") != "1", reason="requires HERMES_E2E_BROWSER=1")
def test_installed_harness_close_then_same_session_navigation(scope, chrome, tab_tmp, monkeypatch, harness_cli):
    import os
    home, db, r = scope
    browser, ws = chrome
    monkeypatch.setattr(browser_use_cli, '_find_cli', lambda: harness_cli)
    monkeypatch.setattr(browser_use_cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(browser_use_cli, '_base_subprocess_env', lambda: {
        'PATH': os.environ['PATH'], 'HOME': str(tab_tmp), 'TMPDIR': str(tab_tmp),
        'BU_CDP_WS': ws, 'BH_RUNTIME_DIR': str(tab_tmp), 'BH_RUNTIME_DIR_SHARED': '1',
        'BH_TMP_DIR': str(tab_tmp), 'BH_TMP_DIR_SHARED': '1', 'BU_NO_AUTOUPDATE': '1'})
    monkeypatch.setattr(browser_use_cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(browser_use_cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(browser_use_cli, '_workspace_dir', lambda *args: str(home / 'workspace'))
    unknown = browser.targets()
    try:
        for title in ('First', 'AfterClose'):
            result = dispatch(code=f"new_tab('data:text/html,<title>{title}</title>'); print(page_info()['title'])",
                              session='mission', timeout_s=60)
            assert result.get('success'), title + ': ' + json.dumps(result)
            assert title in result['output']
            released = dispatch(action='close_owned')
            assert released['success'] and released['closed'] > 0, released
            assert browser.targets() == unknown
            assert db.get_session('a')['ended_at'] is None
    finally:
        stop_test_daemons(r, tab_tmp)


def test_generation_and_profile_scope(scope, tmp_path):
    home, _, r = scope
    create(r, 'ws', 'old', generation='old')
    other = OwnershipRegistry(tmp_path / 'other' / 'browser_tabs.sqlite')
    create(other, 'ws', 'other')
    assert close_owned(home, session_id='a')['closed'] == 0
    with r._db() as db:
        assert db.execute('SELECT pending_close FROM targets').fetchone()[0] == 0
    with other._db() as db:
        assert db.execute('SELECT pending_close FROM targets').fetchone()[0] == 0
