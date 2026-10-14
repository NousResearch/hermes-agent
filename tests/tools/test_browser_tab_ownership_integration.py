from types import SimpleNamespace

import pytest

from tools.browser_tab_ownership import OwnershipRegistry, OwnershipBusy
from tools.browser_tab_capture import install_capture, prepare_exec


WS = 'ws://127.0.0.1:9222/devtools/browser/11111111-1111-1111-1111-111111111111'


def test_shared_send_captures_exact_create_only_and_quarantines(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('session:a', 'g', WS, 'daemon')
    requests = []

    def send(req, response_timeout=5):
        requests.append(req)
        if req.get('method') == 'Target.createTarget':
            return {'result': {'targetId': 'exact'}}
        if req.get('method') == 'Page.navigate':
            raise TimeoutError('private endpoint')
        return {'result': {}}

    helpers = SimpleNamespace(_send=send)
    install_capture(helpers, registry, token)
    helpers._send({'method': 'Target.getTargets'})
    assert registry.owned(token) == []  # a snapshot is never evidence of creation
    helpers._send({'method': 'Target.createTarget'})
    assert registry.owned(token) == ['exact']
    with pytest.raises(TimeoutError):
        helpers._send({'method': 'Page.navigate'})
    registry.finish(token)
    assert registry.call(token)['state'] == 'quarantined'


def test_admission_exists_before_cli_and_namespaces_profile_owner(tmp_path):
    env = {'BU_CDP_WS': WS, 'BU_NAME': 'named', 'PYTHONPATH': 'bad-venv'}
    a, code = prepare_exec(env, 'print(1)', tmp_path / 'a', ('session:s', 'g'))
    assert a.registry.call(a.token)['state'] == 'active'
    assert env['BU_CDP_WS'] == WS
    assert env['PYTHONPATH'].split(':')[0] == 'bad-venv'  # preserve trusted builder path
    first_name = env['BU_NAME']
    with pytest.raises(OwnershipBusy):
        prepare_exec({'BU_CDP_WS': WS, 'BU_NAME': 'named'}, '', tmp_path / 'a', ('session:s', 'g'))
    b_env = {'BU_CDP_WS': WS, 'BU_NAME': 'named'}
    b, _ = prepare_exec(b_env, '', tmp_path / 'b', ('session:s', 'g'))
    assert b_env['BU_NAME'] != first_name
    a.registry.finish(a.token)
    again_env = {'BU_CDP_WS': WS, 'BU_NAME': 'named'}
    again, _ = prepare_exec(again_env, '', tmp_path / 'a', ('session:s', 'g'))
    assert again_env['BU_NAME'] == first_name


def test_inflight_request_prevents_retirement_after_client_exit(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('s', 'g', WS, 'd')
    registry.request_started(token)
    registry.finish(token)
    assert registry.call(token)['state'] == 'quarantined'
    assert not registry.retire('s', 'g', evidence='fixture:terminal')


@pytest.mark.parametrize('session', ['', 'named'])
def test_browser_exec_entry_restores_owned_target_and_preserves_ipc_timeout(tmp_path, monkeypatch, session):
    import json
    import sys
    import types
    from pathlib import Path
    from tools import browser_use_cli as cli, browser_tab_capture as capture

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True, 'ipc_response_timeout_seconds': 27})
    monkeypatch.setattr(cli, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {'BU_CDP_WS': WS})
    monkeypatch.setattr(cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args, **kwargs: None)
    monkeypatch.setattr(capture, '_verify_daemon', lambda *args, **kwargs: None)
    live = {'external': {'targetId': 'external', 'type': 'page'}}
    attached = []
    names = []
    timeouts = []
    created = []
    tokens = []

    def run(cmd, code, env, timeout):
        names.append(env['BU_NAME'])
        helpers = types.ModuleType('browser_harness.helpers')
        def send(req, response_timeout=5):
            timeouts.append(response_timeout)
            method = req.get('method')
            if method == 'Target.getTargets':
                return {'result': {'targetInfos': list(live.values())}}
            if method == 'Target.createTarget':
                tid = 'owned-' + str(len(created))
                created.append(tid)
                live[tid] = {'targetId': tid, 'type': 'page'}
                return {'result': {'targetId': tid}}
            if method == 'Target.attachToTarget':
                attached.append(req['params']['targetId'])
                return {'result': {'sessionId': 'attached'}}
            if req.get('meta') == 'connection_status':
                return {'target_id': None, 'session_id': None, 'page': None}
            if req.get('meta') == 'connection_status':
                return {'target_id': None, 'session_id': None, 'page': None}
            if req.get('meta') == 'set_session':
                return {'session_id': req['session_id']}
            return {'result': {}}
        def cdp(method, session_id=None, _response_timeout=5, **params):
            return helpers._send({'method': method, 'params': params}, response_timeout=_response_timeout).get('result', {})
        helpers._send, helpers.cdp = send, cdp
        package = types.ModuleType('browser_harness')
        package.helpers = helpers
        monkeypatch.setitem(sys.modules, 'browser_harness', package)
        monkeypatch.setitem(sys.modules, 'browser_harness.helpers', helpers)
        # Execute the actual entry point's generated code, including IPC settings and capture.
        exec(code, {'cdp': cdp})
        registry = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
        with registry._db() as db:
            tokens.append(db.execute("SELECT token FROM calls WHERE state='active'").fetchone()[0])
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(cli, '_run_cli_killing_process_group', run)
    for _ in range(2):
        assert json.loads(cli.browser_exec("cdp('Page.navigate', url='https://fixture.invalid')", session=session, task_id='caller'))['success']
    assert attached == ['owned-0', 'owned-0']
    assert len(set(names)) == 1
    assert set(timeouts) == {27}
    del live['owned-0']
    assert json.loads(cli.browser_exec('pass', session=session, task_id='caller'))['success']
    assert attached[-1] == 'owned-1'
    registry = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    assert 'external' not in registry.owned(tokens[-1])
    assert all(registry.call(token)['state'] == 'drained' for token in tokens)


def test_browser_exec_timeout_keeps_admission_quarantined(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from tools import browser_use_cli as cli
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(cli, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {'BU_CDP_WS': WS})
    monkeypatch.setattr(cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args, **kwargs: None)
    def run(*args):
        raise subprocess.TimeoutExpired('fixture-cli', 1)
    monkeypatch.setattr(cli, '_run_cli_killing_process_group', run)
    result = json.loads(cli.browser_exec('pass', task_id='caller'))
    assert 'timed out' in result['error']
    registry = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    with registry._db() as db:
        assert db.execute('SELECT state FROM calls').fetchone()[0] == 'quarantined'
    result = json.loads(cli.browser_exec('pass', task_id='caller'))
    assert 'OwnershipBusy' in result['error']


@pytest.mark.parametrize('code', [
    'undefined_fixture_name',
    "send({'method': 'Page.navigate'}); undefined_fixture_name",
])
def test_browser_exec_completed_python_error_drains_admission(tmp_path, monkeypatch, code):
    import json
    import subprocess
    from pathlib import Path
    from tools import browser_use_cli as cli

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(cli, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {'BU_CDP_WS': WS})
    monkeypatch.setattr(cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args, **kwargs: None)
    registry = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    tokens = []

    def run(cmd, generated_code, env, timeout):
        with registry._db() as db:
            token = db.execute("SELECT token FROM calls WHERE state='active'").fetchone()[0]
        tokens.append(token)
        helpers = SimpleNamespace(_send=lambda req: {'result': {}})
        install_capture(helpers, registry, token)
        # Exercise ordinary Python failure before or after tracked synchronous replies.
        with pytest.raises(NameError) as error:
            exec(code, {'send': helpers._send})
        return subprocess.CompletedProcess(cmd, 1, 'partial output\n', str(error.value))

    monkeypatch.setattr(cli, '_run_cli_killing_process_group', run)
    for _ in range(2):
        result = json.loads(cli.browser_exec(code, task_id='caller'))
        assert result['success'] is False
        assert result['exit_code'] == 1
        assert result['output'] == 'partial output\n'
        assert 'undefined_fixture_name' in result['stderr']
        assert registry.call(tokens[-1])['inflight'] == 0
        assert registry.call(tokens[-1])['state'] == 'drained'
    assert len(tokens) == 2  # the same owner can retry, not OwnershipBusy


@pytest.mark.parametrize('returncode', [0, 1])
@pytest.mark.parametrize('uncertainty', ['transport', 'inflight'])
def test_browser_exec_completed_cli_keeps_uncertain_calls_fenced(tmp_path, monkeypatch, returncode, uncertainty):
    import json
    import subprocess
    from pathlib import Path
    from tools import browser_use_cli as cli

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(cli, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {'BU_CDP_WS': WS})
    monkeypatch.setattr(cli, '_route_backend', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args, **kwargs: None)
    registry = OwnershipRegistry(tmp_path / 'browser_tabs.sqlite')
    tokens = []

    def run(cmd, code, env, timeout):
        with registry._db() as db:
            token = db.execute("SELECT token FROM calls WHERE state='active'").fetchone()[0]
        tokens.append(token)
        if uncertainty == 'transport':
            def send(req):
                raise TimeoutError('fixture transport failure')
            helpers = SimpleNamespace(_send=send)
            install_capture(helpers, registry, token)
            with pytest.raises(TimeoutError):
                helpers._send({'method': 'Page.navigate'})
            # Even a late reply that clears the counter cannot clear quarantine.
            registry.request_finished(token)
            assert registry.call(token)['inflight'] == 0
        else:
            registry.request_started(token)
        return subprocess.CompletedProcess(cmd, returncode, '', '')

    monkeypatch.setattr(cli, '_run_cli_killing_process_group', run)
    result = json.loads(cli.browser_exec('pass', task_id='caller'))
    assert result['exit_code'] == returncode
    assert registry.call(tokens[0])['state'] == 'quarantined'
    retry = json.loads(cli.browser_exec('pass', task_id='caller'))
    assert 'OwnershipBusy' in retry['error']
    assert len(tokens) == 1  # no second dispatch across the fence


def test_http_discovery_pins_generation_and_does_not_rediscover(tmp_path, monkeypatch):
    import io
    from tools import browser_tab_capture as capture
    urls = []
    def discover(url, timeout):
        urls.append(url)
        return io.BytesIO(('{"webSocketDebuggerUrl": "' + WS + '"}').encode())
    monkeypatch.setattr(capture, 'urlopen', discover)
    env = {'BU_CDP_URL': 'http://127.0.0.1:9222'}
    admitted, _ = prepare_exec(env, 'pass', tmp_path, ('unknown:a', 'g'))
    assert admitted.registry.call(admitted.token)['browser'] == WS
    assert capture.pinned_websocket(env) == WS
    assert urls == ['http://127.0.0.1:9222/json/version']
    assert 'BU_CDP_URL' not in env


@pytest.mark.parametrize('matching', [True, False])
def test_daemon_identity_requires_pinned_endpoint_and_binds_start(tmp_path, monkeypatch, matching):
    from pathlib import Path
    from tools import browser_tab_capture as capture
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'daemon')
    proc = tmp_path / 'proc' / '123'
    proc.mkdir(parents=True)
    # Linux stat fields after comm: field 3 through field 22 (starttime).
    (proc / 'stat').write_text('123 (daemon with spaces) ' + ' '.join(['S'] + ['0'] * 18 + ['987']))
    endpoint = WS if matching else WS.replace('11111111-', '22222222-', 1)
    (proc / 'environ').write_bytes(b'BU_NAME=daemon\0BU_CDP_WS=' + endpoint.encode() + b'\0')
    monkeypatch.setattr(capture, 'Path', lambda value: tmp_path / 'proc' if value == '/proc' else Path(value))
    helpers = SimpleNamespace(_send=lambda req: {'pid': 123})
    if matching:
        capture._verify_daemon(helpers, registry, token)
        assert registry.call(token)['daemon_start'] == '987'
        assert registry.call(token)['daemon_pid'] == 123
    else:
        with pytest.raises(OwnershipBusy):
            capture._verify_daemon(helpers, registry, token)
        assert registry.owned(token) == []


def test_owner_and_browser_generation_get_separate_daemons(tmp_path):
    names = []
    for owner, ws in [('a', WS), ('b', WS), ('a', WS.replace('11111111-', '22222222-', 1))]:
        env = {'BU_CDP_WS': ws}
        prepare_exec(env, 'pass', tmp_path, (owner, 'g'))
        names.append(env['BU_NAME'])
    assert len(set(names)) == 3


def test_inflight_survives_registry_reopen(tmp_path):
    path = tmp_path / 'tabs.sqlite'
    registry = OwnershipRegistry(path)
    token = registry.admit('a', 'g', WS, 'daemon')
    registry.request_started(token)
    reopened = OwnershipRegistry(path)
    reopened.finish(token)
    reopened.request_finished(token)  # late response cannot silently clear quarantine
    assert reopened.call(token)['state'] == 'quarantined'
    assert not reopened.retire('a', 'g', evidence='fixture:terminal')
