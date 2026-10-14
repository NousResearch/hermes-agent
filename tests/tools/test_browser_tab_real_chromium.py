"""Real no-auth Chromium + installed Harness. No live relay or personal profile."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.request import urlopen

import pytest
from tests.waiting import wait_until

from tools.browser_tab_lifecycle import PinnedBrowser
from tools.browser_tab_ownership import OwnershipRegistry

pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER") != "1",
                                 reason="set HERMES_E2E_BROWSER=1 for disposable browser tests")]


@pytest.fixture
def harness_cli():
    executable = os.environ.get("HERMES_TEST_BROWSER_HARNESS")
    resolved = shutil.which(executable) if executable else None
    if not resolved:
        pytest.skip("set HERMES_TEST_BROWSER_HARNESS to an installed Harness executable")
    return [resolved]


def stop_test_daemons(registry, home):
    import psutil
    def process_identity(pid):
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]

    with registry._db() as conn:
        rows = conn.execute("SELECT DISTINCT daemon_pid,daemon_start,daemon FROM calls "
                            "WHERE daemon_pid IS NOT NULL").fetchall()
    for pid, start, name in rows:
        try:
            process = psutil.Process(pid)
            env = process.environ()
            if (process_identity(pid) != start or env.get("BU_NAME") != name
                    or env.get("HOME") != str(home)):
                continue
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                if process_identity(pid) == start:
                    process.kill()
                    process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass



@pytest.fixture
def tab_tmp():
    import tempfile
    root = Path(os.environ.get('TMPDIR', tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='bt-', dir=root))
    try:
        yield Path(directory)
    finally:
        # A failed browser drain keeps its disposable profile for diagnosis.
        if not list(directory.rglob('.chromium-live')):
            shutil.rmtree(directory)


@contextmanager
def isolated_chrome(tmp_path, port=0):
    if os.environ.get("HERMES_E2E_BROWSER") != "1":
        pytest.skip("set HERMES_E2E_BROWSER=1 for disposable browser tests")
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = shutil.which('chromium')
    if not binary:
        pytest.skip('Chromium is not installed')
    profile = tmp_path / 'chrome'
    log = (tmp_path / 'chrome.log').open('w')
    preservation = tmp_path / '.chromium-live'
    preservation.touch()  # Keep the profile unless teardown proves drainage.
    proc = subprocess.Popen([binary, '--headless=new', '--no-sandbox', '--disable-gpu',
        '--no-first-run', '--no-default-browser-check', '--remote-debugging-address=127.0.0.1',
        '--remote-debugging-port=' + str(port), '--disable-background-networking',
        '--user-data-dir=' + str(profile), 'about:blank'], stdout=log, stderr=log,
        start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        active = profile / 'DevToolsActivePort'
        ws = None
        def discover_ws():
            ws = None
            if proc.poll() is not None:
                raise RuntimeError('isolated Chromium exited')
            try:
                if port:
                    with urlopen(f'http://127.0.0.1:{port}/json/version', timeout=.5) as response:
                        ws = json.load(response)['webSocketDebuggerUrl']
                elif active.exists():
                    actual_port, path = active.read_text().splitlines()[:2]
                    ws = 'ws://127.0.0.1:' + actual_port + path
            except (OSError, ValueError, KeyError):
                pass
            return ws
        ws = wait_until(discover_ws, timeout=20, what='Chromium CDP endpoint')
        browser = None
        def connect_browser():
            try:
                return PinnedBrowser(ws, response_timeout=10)
            except (TimeoutError, OSError):
                return None
        browser = wait_until(connect_browser, timeout=20, what='Chromium websocket')
        try:
            yield browser, ws
        finally:
            browser.socket.close()
    finally:
        # Snapshot only this fixture's descendants before stopping its parent.
        import psutil
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        _, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, survivors = psutil.wait_procs(alive, timeout=5)
        live = []
        for child in survivors:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    live.append(child)
            except psutil.NoSuchProcess:
                pass
        assert not live, 'Disposable Chromium descendants did not drain'
        assert proc.poll() is not None
        preservation.unlink()
        log.close()


@pytest.fixture
def chrome(tab_tmp):
    with isolated_chrome(tab_tmp) as browser:
        yield browser


def targets_unrecorded(registry, targets):
    """Closure is settled only when exact target receipts are gone too."""
    with registry._db() as conn:
        recorded = {row[0] for row in conn.execute('SELECT target FROM targets')}
    return not targets & recorded


def wait_for_targets(browser, predicate, retry=None):
    """Poll exact absence; never replay target creation after an ambiguous reply."""
    def settled():
        if retry is not None:
            retry()
        try:
            return predicate(browser.targets())
        except (TimeoutError, OSError):
            return False
    wait_until(settled, timeout=60, what='exact Chromium target state')


def test_pinned_real_close_preserves_unknown(chrome, tmp_path):
    browser, ws = chrome
    unknown = browser.targets()
    a = browser.command('Target.createTarget', url='about:blank')['targetId']
    b = browser.command('Target.createTarget', url='about:blank')['targetId']
    r = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    for owner, target in [('a', a), ('b', b)]:
        token = r.admit(owner, 'run', ws, owner)
        r.record_created(token, target)
        r.finish(token)
    r.retire('a', 'run', evidence='fixture:native-finished')
    wait_for_targets(browser, lambda targets: targets == unknown | {b} and targets_unrecorded(r, {a}),
                     retry=lambda: r.cleanup(ws, browser.targets, browser.close_target))
    assert browser.targets() == unknown | {b}
    print(json.dumps({'closed': a, 'active': b, 'unknown': sorted(unknown)}))


@pytest.mark.live_system_guard_bypass  # Harness forks; teardown verifies exact temporary HOME/name/PID.
def test_real_browser_exec_exact_capture_and_next_owner(chrome, tab_tmp, monkeypatch, harness_cli):
    tmp_path = tab_tmp
    from tools import browser_use_cli as cli
    from tools.browser_tab_lifecycle import sweep
    from hermes_state import SessionDB
    browser, ws = chrome
    home = tmp_path / 'hermes'
    home.mkdir()
    cfg = {'backend': 'browser-use', 'tab_cleanup_enabled': True}
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: home)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: cfg)
    monkeypatch.setattr(cli, '_find_cli', lambda: harness_cli)
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {
        'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'TMPDIR': str(tmp_path),
        'BU_CDP_WS': ws,
        'BH_RUNTIME_DIR': str(tmp_path), 'BH_RUNTIME_DIR_SHARED': '1',
        'BH_TMP_DIR': str(tmp_path), 'BH_TMP_DIR_SHARED': '1',
        'BU_NO_AUTOUPDATE': '1'})
    monkeypatch.setattr(cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: str(home / 'workspace'))
    monkeypatch.setattr(cli, '_configured_ipc_response_timeout', lambda: 30)
    db = SessionDB(home / 'state.db')
    db.create_session('a', 'cli')
    db.create_session('b', 'cli')
    unknown = browser.targets()
    r = OwnershipRegistry(home / 'browser_tabs.sqlite')
    daemons = set()
    try:
        def execute(task, code):
            result = json.loads(cli.browser_exec(code, session='named', task_id=task,
                session_id=task, timeout_s=90))
            with r._db() as conn:
                daemons.update(row[0] for row in conn.execute('SELECT DISTINCT daemon FROM calls'))
            assert result.get('success'), json.dumps(result, ensure_ascii=False)
            return result
        execute('a', "print(cdp('Target.createTarget', url='about:blank')['targetId'])")
        with r._db() as conn:
            a = {row[0] for row in conn.execute("SELECT target FROM targets WHERE owner='session:a'")}
        assert len(a) == 2
        # A third-party CDP client creates a target while A's actual exec is running.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            marker = tmp_path / 'running'
            release = tmp_path / 'release'
            code = (f"from pathlib import Path; import time\nPath({str(marker)!r}).touch()\n"
                    f"while not Path({str(release)!r}).exists(): time.sleep(.02)\nprint('done')")
            future = pool.submit(execute, 'a', code)
            try:
                wait_until(marker.exists, timeout=30, what='mission running')
                assert not future.done()
                external = browser.command('Target.createTarget', url='about:blank')['targetId']
            finally:
                release.touch()
            future.result(timeout=30)
        execute('b', "new_tab('data:text/html,<title>B</title>'); print(page_info()['title'])")
        # Explicit reuse of an existing blank is navigation, never creation proof.
        execute('a', f"switch_tab({external!r}); new_tab('data:text/html,<title>Reused</title>'); print(page_info()['title'])")
        execute('a', "print(cdp('Target.getTargets')['targetInfos'] != [])")
        with r._db() as conn:
            assert {row[0] for row in conn.execute("SELECT target FROM targets WHERE owner='session:a'")} == a
            b = {row[0] for row in conn.execute("SELECT target FROM targets WHERE owner='session:b'")}
        assert not unknown & (a | b)
        assert external not in a | b
        db.end_session('a', 'new_session')
        wait_for_targets(browser, lambda targets: not a & targets and targets_unrecorded(r, a),
                         retry=lambda: sweep(home))
        assert not a & browser.targets()
        assert unknown | b | {external} <= browser.targets()
        db.create_session('next', 'cli')
        execute('next', "print(page_info()['url'])")
        assert unknown | b | {external} <= browser.targets()
        print(json.dumps({'a_closed': sorted(a), 'b_retained': sorted(b), 'external': external}))
    finally:
        stop_test_daemons(r, tmp_path)
        db.close()


def test_real_parent_hold_cancel_retry_and_profile_isolation(chrome, tmp_path, monkeypatch):
    from tools.browser_tab_lifecycle import sweep, native_child_finished
    from hermes_state import SessionDB
    browser, ws = chrome
    home = tmp_path / 'owner-home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    db = SessionDB(home / 'state.db')
    db.create_session('parent', 'cli')
    r = OwnershipRegistry(home / 'browser_tabs.sqlite')
    target = browser.command('Target.createTarget', url='about:blank')['targetId']
    token = r.admit('session:parent', 'parent', ws, 'd')
    r.record_created(token, target)
    r.finish(token)
    actor = json.dumps(['child-task:continuation', 'continuation'])
    r.hold(ws, target, actor)
    db.end_session('parent', 'new_session')
    sweep(home)
    assert target in browser.targets()  # parent reset cannot close native continuation
    native_child_finished('continuation', home)  # includes cancellation after native exit
    # Reopen the durable registry, as after restart. The exact hold/receipt survives.
    r = OwnershipRegistry(home / 'browser_tabs.sqlite')
    wait_for_targets(browser, lambda targets: target not in targets and targets_unrecorded(r, {target}),
                     retry=lambda: sweep(home))
    assert target not in browser.targets()
    db.close()

    active = browser.command('Target.createTarget', url='about:blank')['targetId']
    other = OwnershipRegistry(tmp_path / 'other-home' / 'browser_tabs.sqlite')
    token = other.admit('session:parent', 'parent', ws, 'd')
    other.record_created(token, active)
    other.finish(token)
    sweep(home)
    assert active in browser.targets()  # identical IDs in another profile are independent
    other.retire('session:parent', 'parent', evidence='fixture:native-exit')
    def ambiguous_close(target):
        browser.close_target(target)
        raise TimeoutError('reply lost')
    other.cleanup(ws, browser.targets, ambiguous_close)
    wait_for_targets(browser, lambda targets: active not in targets)
    assert active not in browser.targets()
    with other._db() as conn:
        assert conn.execute('SELECT pending_close FROM targets').fetchone()[0] == 1
    other.cleanup(ws, browser.targets, browser.close_target)
    with other._db() as conn:
        assert not conn.execute('SELECT 1 FROM targets').fetchone()


def test_real_browser_identity_is_not_port_identity(chrome, tmp_path):
    browser, ws = chrome
    target = browser.command('Target.createTarget', url='about:blank')['targetId']
    r = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = r.admit('old', 'g', ws + '-old-generation', 'old')
    r.record_created(token, target)
    r.finish(token)
    r.retire('old', 'g', evidence='fixture:native-exit')
    # Even the exact same target ID cannot be closed in a different browser generation.
    r.cleanup(ws, browser.targets, browser.close_target)
    assert target in browser.targets()


def test_real_restart_same_port_never_retargets_old_registry(tab_tmp, monkeypatch):
    from urllib.parse import urlsplit
    from tools.browser_tab_lifecycle import sweep
    home = tab_tmp / 'hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('browser:\n  tab_cleanup_enabled: true\n')
    r = OwnershipRegistry(home / 'browser_tabs.sqlite')
    with isolated_chrome(tab_tmp / 'first') as (first, old_ws):
        port = urlsplit(old_ws).port
        target = first.command('Target.createTarget', url='about:blank')['targetId']
        token = r.admit('old', 'run', old_ws, 'd')
        r.record_created(token, target)
        r.finish(token)
        r.retire('old', 'run', evidence='fixture:native-exit')
    with isolated_chrome(tab_tmp / 'second', port) as (second, new_ws):
        assert urlsplit(new_ws).port == port and new_ws != old_ws
        before = second.targets()
        sweep(home)  # persisted old WS fails; it must never rediscover the reused port
        assert second.targets() == before
        with r._db() as conn:
            assert conn.execute('SELECT target FROM targets').fetchone()[0] == target
