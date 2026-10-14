"""Real installed Harness + disposable Chromium: no startup orphan blanks."""
import json
import os

import pytest

from tests.tools.test_browser_tab_real_chromium import tab_tmp, chrome, harness_cli, stop_test_daemons


@pytest.mark.platforms("linux")
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER") != "1", reason="requires HERMES_E2E_BROWSER=1")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('release', [None, 'held', 'failed'])
def test_real_startup_single_owned_target_and_continuation(chrome, tab_tmp, monkeypatch, harness_cli, release):
    from tools import browser_use_cli as cli
    from tools import browser_tab_lifecycle as lifecycle
    from tools.browser_tab_ownership import OwnershipRegistry
    from hermes_state import SessionDB
    browser, ws = chrome
    home = tab_tmp / 'hermes'
    home.mkdir()
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: home)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(cli, '_find_cli', lambda: harness_cli)
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {
        'PATH': os.environ['PATH'], 'HOME': str(tab_tmp), 'TMPDIR': str(tab_tmp),
        'BU_CDP_WS': ws, 'BU_NO_AUTOUPDATE': '1',
        'BH_RUNTIME_DIR': str(tab_tmp), 'BH_RUNTIME_DIR_SHARED': '1',
        'BH_TMP_DIR': str(tab_tmp), 'BH_TMP_DIR_SHARED': '1'})
    monkeypatch.setattr(cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: str(home / 'workspace'))
    monkeypatch.setattr(cli, '_configured_ipc_response_timeout', lambda: 30)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle, 'ensure_scheduled', lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle, 'after_call', lambda *args, **kwargs: None)
    db = SessionDB(home / 'state.db')
    db.create_session('startup-owner', 'cli')
    registry = OwnershipRegistry(home / 'browser_tabs.sqlite')
    initial = browser.targets()
    try:
        def execute(code):
            result = json.loads(cli.browser_exec(code, session='startup', task_id='startup-owner',
                session_id='startup-owner', timeout_s=90))
            assert result.get('success'), result
            return result
        execute("print('startup')")
        with registry._db() as conn:
            targets = {r[0] for r in conn.execute('SELECT target FROM targets')}
        assert len(targets) == 1
        assert browser.targets() == initial | targets  # no hidden dedicated blank
        execute("print('continuation')")
        assert browser.targets() == initial | targets
        if release:
            startup = next(iter(targets))
            if release == 'held':
                registry.hold(ws, startup, 'live-actor')
            registry.request_owner_close('session:startup-owner', 'startup-owner')
            if release == 'failed':
                def failed_close(target):
                    raise TimeoutError('close reply lost')
                registry.cleanup(ws, browser.targets, failed_close)
            execute("print('navigation after pending close')")
            assert initial | targets < browser.targets()
            assert len(browser.targets() - initial - targets) == 1
            registry.release(ws, startup, 'live-actor')
        # Explicit native retirement only in the disposable test ledger; the
        # production session completion contract is intentionally unchanged.
        assert registry.retire('session:startup-owner', 'startup-owner', evidence='fixture:mission-finished')
        from tests.waiting import wait_until
        def cleaned():
            registry.cleanup(ws, browser.targets, browser.close_target)
            try:
                return browser.targets() == initial
            except (TimeoutError, OSError):
                return False
        wait_until(cleaned, timeout=60, what='owned targets absent')
        print(json.dumps({'startup_owned': len(targets), 'orphan_blanks': 0,
                          'continuation_preserved': True, 'unknown_preserved': True}))
    finally:
        stop_test_daemons(registry, tab_tmp)
        db.close()
