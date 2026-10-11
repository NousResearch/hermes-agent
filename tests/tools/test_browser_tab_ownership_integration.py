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


@pytest.mark.parametrize('req,reply', [
    ({'method': 'Page.navigate'}, {'result': {}}),
    ({'meta': 'ping'}, {'pong': True, 'pid': 123}),
    ({'meta': 'set_session', 'session_id': 'session'}, {'session_id': 'session'}),
])
def test_confirmed_reply_drains_request(tmp_path, req, reply):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'daemon')
    helpers = SimpleNamespace(_send=lambda req: reply)
    install_capture(helpers, registry, token)
    assert helpers._send(req) == reply
    registry.finish(token)
    assert registry.call(token)['state'] == 'drained'
    assert registry.call(token)['inflight'] == 0


@pytest.mark.parametrize('req', [
    {'method': 'Page.navigate'}, {'meta': 'ping'},
    {'meta': 'set_session', 'session_id': 'session'},
])
@pytest.mark.parametrize('reply', [{}, [], None, {'result': None}, {'pong': False}])
def test_ambiguous_reply_keeps_request_inflight(tmp_path, req, reply):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'daemon')
    helpers = SimpleNamespace(_send=lambda req: reply)
    install_capture(helpers, registry, token)
    with pytest.raises(ValueError):
        helpers._send(req)
    registry.finish(token)
    assert registry.call(token)['inflight'] == 1
    assert registry.call(token)['state'] == 'quarantined'
    with pytest.raises(OwnershipBusy):
        registry.admit('owner', 'g', WS, 'daemon')


@pytest.mark.parametrize('req', [
    {'method': 'Page.navigate'}, {'meta': 'ping'},
    {'meta': 'set_session', 'session_id': 'session'},
    {'method': 'Target.createTarget'},
])
@pytest.mark.parametrize('error', [False, True])
def test_real_harness_ipc_eof_and_explicit_error(tmp_path, monkeypatch, req, error):
    import json
    import socket
    import threading
    monkeypatch.setenv('BH_RUNTIME_DIR', str(tmp_path / 'runtime'))
    monkeypatch.setenv('BH_AGENT_WORKSPACE', str(tmp_path / 'workspace'))
    helpers = pytest.importorskip('browser_harness.helpers')
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'daemon')
    client, server = socket.socketpair()
    observed = []

    def peer():
        with server:
            data = b''
            while not data.endswith(b'\n'):
                data += server.recv(4096)
            observed.append(json.loads(data))
            if error:
                server.sendall(b'{"error":"known daemon rejection"}\n')

    thread = threading.Thread(target=peer)
    thread.start()
    monkeypatch.setattr(helpers.ipc, 'connect', lambda *a, **kw: (client, None))
    # Restore both hooks after the test; install_capture wraps the real IPC boundary.
    monkeypatch.setattr(helpers, '_send', helpers._send)
    monkeypatch.setattr(helpers.ipc, 'request', helpers.ipc.request)
    install_capture(helpers, registry, token)
    try:
        with pytest.raises(RuntimeError if error else ValueError):
            helpers._send(req)
    finally:
        client.close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert observed == [req]
    registry.finish(token)
    assert registry.call(token)['inflight'] == (0 if error else 1)
    assert registry.call(token)['state'] == ('drained' if error else 'quarantined')
    if error:
        registry.admit('owner', 'g', WS, 'daemon')
    else:
        with pytest.raises(OwnershipBusy):
            registry.admit('owner', 'g', WS, 'daemon')


def test_prepare_exec_preserves_only_builder_trusted_site_dir(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys
    from tools import browser_use_cli as cli
    from tools import browser_tool

    trusted = tmp_path / 'trusted-site'
    package = trusted / 'browser_harness'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('BUNDLED_MARKER = "trusted"\n')
    untrusted = tmp_path / 'parent-site'
    untrusted.mkdir()
    monkeypatch.setenv('PYTHONPATH', str(untrusted))
    monkeypatch.setattr(browser_tool, '_build_browser_env', lambda: {
        'BU_CDP_WS': WS, 'PYTHONPATH': str(untrusted), 'PATH': os.defpath})
    monkeypatch.setattr(cli, '_harness_site_dir', lambda: str(trusted))
    env = cli._base_subprocess_env()
    assert env['PYTHONPATH'] == str(trusted)
    prepare_exec(env, 'pass', tmp_path / 'home', ('owner', 'g'))
    paths = env['PYTHONPATH'].split(os.pathsep)
    assert str(trusted) in paths
    assert str(untrusted) not in paths
    probe = subprocess.run([sys.executable, '-S', '-c',
        'import browser_harness; from tools import browser_tab_capture; '
        'import json; print(json.dumps([browser_harness.BUNDLED_MARKER, browser_tab_capture.__file__]))'],
        env={'PATH': os.defpath, 'PYTHONPATH': env['PYTHONPATH']},
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert probe.returncode == 0, probe.stderr
    marker, source = json.loads(probe.stdout)
    assert marker == 'trusted'
    assert source.endswith('tools/browser_tab_capture.py')


def test_admission_exists_before_cli_and_namespaces_profile_owner(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'bad-venv')
    env = {'BU_CDP_WS': WS, 'BU_NAME': 'named'}
    a, code = prepare_exec(env, 'print(1)', tmp_path / 'a', ('session:s', 'g'))
    assert a.registry.call(a.token)['state'] == 'active'
    assert env['BU_CDP_WS'] == WS
    assert 'bad-venv' not in env['PYTHONPATH']
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


@pytest.mark.parametrize('session', ['', 'named'])
def test_browser_exec_entry_restores_owned_target(tmp_path, monkeypatch, session):
    import json
    import sys
    import types
    from pathlib import Path
    from tools import browser_use_cli as cli, browser_tab_capture as capture

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': True})
    monkeypatch.setattr(cli, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {'BU_CDP_WS': WS})
    monkeypatch.setattr(cli, '_route_backend', lambda *args: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args: None)
    monkeypatch.setattr(capture, '_verify_daemon', lambda *args: None)
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
    from tools.registry import registry as tool_registry
    entry = tool_registry.get_entry('browser_exec')
    for _ in range(2):
        result = entry.handler({'code': "cdp('Page.navigate', url='https://fixture.invalid')", 'session': session}, task_id='caller')
        assert json.loads(result)['success']
    assert attached == ['owned-0', 'owned-0']
    assert len(set(names)) == 1
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
    monkeypatch.setattr(cli, '_route_backend', lambda *args: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: None)
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
    monkeypatch.setattr(cli, '_route_backend', lambda *args: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args: None)
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
@pytest.mark.parametrize('uncertainty', ['transport', 'inflight', 'empty_reply'])
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
    monkeypatch.setattr(cli, '_route_backend', lambda *args: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args: None)
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
        elif uncertainty == 'empty_reply':
            helpers = SimpleNamespace(_send=lambda req: {})
            install_capture(helpers, registry, token)
            try:
                helpers._send({'method': 'Page.navigate'})
            except ValueError:
                pass
            assert registry.call(token)['inflight'] == 1
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


@pytest.mark.parametrize('reply', [{}, {'result': {}}, {'result': {'targetId': ''}}])
def test_ambiguous_creation_reply_stays_fenced(tmp_path, reply):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'g', WS, 'daemon')
    helpers = SimpleNamespace(_send=lambda req: reply)
    install_capture(helpers, registry, token)
    with pytest.raises(ValueError):
        helpers._send({'method': 'Target.createTarget'})
    registry.finish(token)
    assert registry.call(token)['state'] == 'quarantined'
    assert registry.owned(token) == []


@pytest.mark.parametrize('enabled', [None, False, 'true', 1])
def test_ownership_requires_explicit_boolean_optin(tmp_path, monkeypatch, enabled):
    import json
    from pathlib import Path
    from tools import browser_use_cli as cli
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cli, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(cli, '_read_browser_cfg', lambda: {'tab_cleanup_enabled': enabled})
    monkeypatch.setattr(cli, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(cli, '_base_subprocess_env', lambda: {})
    monkeypatch.setattr(cli, '_route_backend', lambda *args: None)
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *args: None)
    monkeypatch.setattr(cli, '_find_screenshot', lambda *args: None)
    def run(cmd, code, env, timeout):
        assert env['BU_NAME'] == 'named'
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(cli, '_run_cli_killing_process_group', run)
    assert json.loads(cli.browser_exec('pass', session='named'))['success']
    assert not (tmp_path / 'browser_tabs.sqlite').exists()
