"""Real Cua transports against an explicit protocol-test child (not a desktop).

The tiny executable records every subprocess environment and RPC. It exercises
manifest -> private serve -> status -> MCP handshake -> CLI -> stop with no
host display access. Real desktop pixels/input require separate live integration.
"""
import json
from pathlib import Path
import socket
import sys
import tempfile

import pytest

from hermes_cli.session_execution import (
    ComputerUseLaunchContext, SessionExecutionContext,
    register_session_execution_context, remove_session_execution_context,
)


DRIVER = r'''
import json, os, signal, socket, sys, time
args = sys.argv[1:]
log = os.environ['TEST_CUA_LOG']
def record(kind, payload):
    with open(log, 'a') as f:
        f.write(json.dumps({'kind': kind, 'payload': payload, 'env': dict(os.environ)})+'\n')
record('spawn', args)
verb = args[0]
if verb == 'manifest':
    print(json.dumps({'binary_version': '0.23.2', 'mcp_invocation': {'command': sys.argv[0], 'args': ['mcp']},
        'subcommands': [{'name': n, 'args': [{'name': a} for a in flags]} for n, flags in {
            'mcp': ['--socket', '--grant'],
            'serve': ['--socket','--permission-mode','--capability-manifest','--approve-capability-manifest','--embedded'],
            'stop': ['--socket']}.items()]}))
elif verb == '--help':
    print('--no-overlay')
elif verb == 'serve':
    path = args[args.index('--socket')+1]
    s = socket.socket(socket.AF_UNIX); s.bind(path); s.listen()
    while True:
        conn, _ = s.accept()
        data = conn.recv(64); conn.close()
        if data == b'stop': break
    s.close()
elif verb in ('status', 'stop'):
    s = socket.socket(socket.AF_UNIX)
    try: s.connect(args[args.index('--socket')+1]); s.sendall(verb.encode())
    except OSError: sys.exit(1)
    print('{}')
elif verb == 'call':
    record('call', {'name': args[1], 'arguments': json.loads(args[2])})
    print(json.dumps({'ok': True}))
elif verb == 'mcp':
    for line in sys.stdin:
        msg=json.loads(line); method=msg.get('method'); params=msg.get('params', {})
        if 'id' not in msg: continue
        if method == 'initialize': result={'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'protocol-fixture','version':'1'}}
        elif method == 'tools/list':
            result={'tools':[{'name':n,'inputSchema':{'type':'object','properties':{'target':{},'delivery_mode':{}}}} for n in ['get_desktop_state','click','type_text','press_key','hotkey','start_session','end_session','set_config','get_config']]}
        elif method == 'tools/call':
            record('rpc',params)
            if params['name']=='click' and os.environ.get('TEST_CUA_END_ONCE'):
                os.environ.pop('TEST_CUA_END_ONCE')
                result={'isError':True, 'content':[{'type':'text','text':'session has ended; call start_session'}]}
            elif params['name']=='get_desktop_state':
                result={'content':[{'type':'image','mimeType':'image/png','data':'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=='}]}
            else: result={'content':[{'type':'text','text':'{}'}], 'structuredContent':{'ok':True}}
        else: result={}
        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)
'''


@pytest.mark.linux_only
def test_private_standard_context_reaches_all_transports_and_restricts_targets(tmp_path, monkeypatch):
    from tools.computer_use.tool import handle_computer_use, release_computer_use_session, _get_backend
    from hermes_cli.session_execution import resolve_session_execution_context
    from tools.computer_use.cua_backend import CuaDriverBackend
    from hermes_constants import get_hermes_home
    # Keep shell/driver setup entirely under the runner's temporary profile.
    assert str(get_hermes_home()).startswith('/tmp/')
    binary = tmp_path / 'protocol-driver'
    binary.write_text('#!' + sys.executable + '\n' + DRIVER)
    binary.chmod(0o700)
    wrapper = tmp_path / 'prefix.py'
    wrapper.write_text('import os,sys; os.environ["PREFIX_CHAIN"]=os.environ.get("PREFIX_CHAIN", "")+sys.argv[1]; os.execvpe(sys.argv[2],sys.argv[2:],os.environ)')
    monkeypatch.setenv('TEST_CUA_HOST_SECRET', 'must-not-inherit')
    monkeypatch.setenv('OPENAI_API_KEY', 'provider-secret')
    monkeypatch.setenv('LOGNAME', 'host-logname')
    monkeypatch.setenv('CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS', '1')
    with tempfile.TemporaryDirectory(prefix='hc-test-') as directory:
        runtime = Path(directory)
        endpoint = runtime / 'wl'
        sock = socket.socket(socket.AF_UNIX); sock.bind(str(endpoint))
        log = tmp_path / 'calls.jsonl'
        controlled = [False]
        context = SessionExecutionContext(
            command_prefix=(sys.executable, str(wrapper), 'outer,'),
            env_set={'XDG_RUNTIME_DIR': str(runtime), 'WAYLAND_DISPLAY': 'wl', 'TEST_CUA_LOG': str(log),
                     'CUA_DRIVER_RS_ENABLE_WAYLAND': '1', 'PATH': '/usr/bin'},
            env_unset={'TEST_CUA_HOST_SECRET', 'LOGNAME'},
            computer_use=ComputerUseLaunchContext(driver_command=str(binary), private_daemon=True,
                runtime_dir=str(runtime),
                command_prefix=(sys.executable, str(wrapper), 'inner'),
                no_overlay=False, session_name='private-run', theme='cua.test', desktop_only=True,
                allow_input=lambda: not controlled[0]))
        register_session_execution_context('cu-session', context)
        try:
            lease = resolve_session_execution_context(session_id='cu-session')
            backend = CuaDriverBackend(execution_context=lease)
            assert backend.permission_mode == 'standard'
            backend.stop()  # construction alone spawns nothing
            result = handle_computer_use({'action':'capture','app':'screen'}, session_id='cu-session')
            assert json.loads(result)['width'] > 0, result  # tiny fixture image is text-only by provider policy
            for action in ('list_apps', 'list_windows'):
                result = json.loads(handle_computer_use({'action':action}, session_id='cu-session'))
                assert result['count'] == 0, result
            for args in ({'action':'capture','pid':321,'window_id':123},
                         {'action':'click','app':'foreign','coordinate':[1,1]},
                         {'action':'click','pid':321,'coordinate':[1,1]}):
                result = handle_computer_use(args, session_id='cu-session')
                assert 'desktop-only' in str(result), result
                denied = json.loads(result)
                assert denied['ok'] is False
                assert denied['code'] == 'policy_denied'
                assert denied['verdict']['decision'] == 'deny'
                assert denied['verdict']['retryable'] is False
            result = handle_computer_use({'action':'capture','app':'screen'}, session_id='cu-session')
            result = handle_computer_use({'action':'click','app':'screen','coordinate':[1,1],
                                         'delivery_mode':'foreground'}, session_id='cu-session')
            assert json.loads(result)['ok'], result
            backend = _get_backend('cu-session')
            controlled[0] = True  # takeover AFTER capture/earlier input: callback must be live
            import tools.computer_use_tool  # real tool registration
            from tools.registry import registry
            refused = json.loads(registry.dispatch('computer_use',
                {'action':'type','text':'blocked','app':'screen'}, session_id='cu-session'))
            assert refused['ok'] is False
            assert 'paused by session owner' in refused['message']
            assert refused['code'] == 'policy_denied'
            assert refused['verdict']['decision'] == 'deny'
            assert refused['verdict']['retryable'] is False
            assert 'do not retry' in refused['verdict']['hint'].lower()
            assert 'recommended' not in refused['verdict']
            assert 'escalation' not in refused
            assert not any(row['kind'] in {'rpc', 'call'} and row['payload']['name'] == 'type_text'
                           for row in map(json.loads, log.read_text().splitlines()))
            with pytest.raises(RuntimeError, match='paused'):
                backend._session._call_tool_via_cli('type_text', {'target': {'kind':'desktop','display_id':'primary'},
                                                                'text':'blocked'}, 5)
            assert json.loads(handle_computer_use({'action':'capture','app':'screen'}, session_id='cu-session'))['width'] > 0
            controlled[0] = False
            backend._session._call_tool_via_cli('get_desktop_state', {'session':'private-run'}, 5)
            remove_session_execution_context('cu-session')
            assert not backend._embedded_daemon._running
            with pytest.raises(RuntimeError, match='revoked'):
                backend._session.call_tool('get_desktop_state', {'session':'private-run'})
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            spawns = [r for r in rows if r['kind']=='spawn']
            assert {'manifest','serve','status','mcp','call','stop'} <= {r['payload'][0] for r in spawns}
            for row in spawns:
                env = row['env']
                assert env.get('PREFIX_CHAIN') == 'outer,inner', row['payload']
                assert env['WAYLAND_DISPLAY'] == 'wl'
                assert env['XDG_RUNTIME_DIR'] == str(runtime)
                assert 'TEST_CUA_HOST_SECRET' not in env
                assert 'LOGNAME' not in env, row['payload']
                assert env['PATH'] == '/usr/bin', row['payload']
                assert 'OPENAI_API_KEY' not in env
                assert 'CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS' not in env
            serve = next(r['payload'] for r in spawns if r['payload'][0]=='serve')
            assert Path(serve[serve.index('--socket')+1]).parent == runtime
            assert serve[serve.index('--permission-mode')+1] == 'standard'
            assert '--no-overlay' not in serve
            assert serve[serve.index('--cursor-theme')+1] == 'cua.test'
            click = next(r['payload']['arguments'] for r in rows if r['kind']=='rpc' and r['payload']['name']=='click')
            assert click['target'] == {'kind':'desktop','display_id':'primary'}
            assert 'pid' not in click and 'window_id' not in click
            assert not any(r['kind']=='rpc' and r['payload']['name'] in {'list_apps','list_windows'} for r in rows)
        finally:
            release_computer_use_session('cu-session')
            remove_session_execution_context('cu-session')
            sock.close()


@pytest.mark.linux_only
@pytest.mark.parametrize('boundary', ['queued', 'recovery', 'revival'])
@pytest.mark.parametrize('denial', ['paused', 'invalid', 'revoked'])
def test_mcp_send_revalidates_live_policy(tmp_path, monkeypatch, boundary, denial):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.session_execution import SessionExecutionError, resolve_session_execution_context
    from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession

    binary = tmp_path / 'protocol-driver'
    binary.write_text('#!' + sys.executable + '\n' + DRIVER)
    binary.chmod(0o700)
    log = tmp_path / 'calls.jsonl'
    monkeypatch.setenv('TEST_CUA_LOG', str(log))  # direct protocol fixture, no embedded daemon
    if boundary == 'revival':
        monkeypatch.setenv('TEST_CUA_END_ONCE', '1')
    paused, invalid = threading.Event(), threading.Event()
    reached, release = threading.Event(), threading.Event()
    register_session_execution_context('send-boundary', SessionExecutionContext(
        env_set={'TEST_CUA_LOG': str(log), **({'TEST_CUA_END_ONCE': '1'} if boundary == 'revival' else {})},
        validate=lambda: not invalid.is_set(),
        computer_use=ComputerUseLaunchContext(driver_command=str(binary), private_daemon=True,
                                              allow_input=lambda: not paused.is_set())))
    bridge = _AsyncBridge()
    session = _CuaDriverSession(bridge, execution_context=resolve_session_execution_context(session_id='send-boundary'))

    async def hold():
        reached.set()
        assert await asyncio.to_thread(release.wait, 10), 'pending send was never released'

    try:
        session.start()
        session.call_tool('start_session', {'session': 'send-boundary'})
        if boundary == 'recovery':
            populate = session._populate_capabilities

            async def hold_recovery(transport):
                await populate(transport)
                await hold()

            monkeypatch.setattr(session, '_populate_capabilities', hold_recovery)
            session._timeout_suspect = True
        else:
            send = session._call_tool_async

            async def hold_send(name, args):
                if name == ('start_session' if boundary == 'revival' else 'click'):
                    await hold()
                return await send(name, args)

            monkeypatch.setattr(session, '_call_tool_async', hold_send)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(session.call_tool, 'click', {'x': 1, 'y': 1})
            try:
                assert reached.wait(10), f'{boundary} never reached the pending send'
                if denial == 'paused':
                    paused.set()
                elif denial == 'invalid':
                    invalid.set()
                else:
                    remove_session_execution_context('send-boundary')
            finally:
                release.set()
            reason = {'paused': 'paused by session owner', 'invalid': 'validation failed', 'revoked': 'revoked'}[denial]
            with pytest.raises(SessionExecutionError, match=reason):
                pending.result(timeout=15)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        clicks = [r for r in rows if r['kind'] == 'rpc' and r['payload']['name'] == 'click']
        assert len(clicks) == (1 if boundary == 'revival' else 0)  # revival's first call was explicitly rejected
        declarations = [r for r in rows if r['kind'] == 'rpc' and r['payload']['name'] == 'start_session']
        assert len(declarations) == (2 if boundary != 'queued' and denial == 'paused' else 1)
        if denial == 'paused':
            assert session.call_tool('get_config', {})['isError'] is False
        # Revoked contexts may end an existing transport, never restart it.
        assert session.call_tool('end_session', {'session': 'send-boundary'})['isError'] is False
    finally:
        release.set()
        session.stop()
        bridge.stop()
        remove_session_execution_context('send-boundary')


def test_typed_mcp_policy_denial_never_enters_transport_recovery(monkeypatch):
    from hermes_cli.session_execution import SessionExecutionError
    from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession

    bridge = _AsyncBridge()
    bridge.start()
    session = _CuaDriverSession(bridge)
    session._started = True
    denial = SessionExecutionError('daemon proxy policy denied')

    async def denied_send(name, args):
        raise denial

    monkeypatch.setattr(session, '_call_tool_async', denied_send)
    try:
        with pytest.raises(SessionExecutionError) as caught:
            session.call_tool('click', {})
        assert caught.value is denial
        assert not session._timeout_suspect
    finally:
        bridge.stop()


@pytest.mark.linux_only
def test_context_invalidated_during_mcp_startup_is_policy_denied(tmp_path, monkeypatch):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import tools.computer_use_tool
    from tools.registry import registry
    from tools.computer_use.tool import release_computer_use_session
    from tools.computer_use.cua_backend_session import _CuaDriverSession

    binary = tmp_path / 'protocol-driver'
    binary.write_text('#!' + sys.executable + '\n' + DRIVER)
    binary.chmod(0o700)
    log = tmp_path / 'calls.jsonl'
    invalid, starting, release = threading.Event(), threading.Event(), threading.Event()
    lifecycle = _CuaDriverSession._lifecycle_coro

    async def hold_startup(self):
        starting.set()
        assert await asyncio.to_thread(release.wait, 10), 'startup was never released'
        return await lifecycle(self)

    monkeypatch.setattr(_CuaDriverSession, '_lifecycle_coro', hold_startup)
    with tempfile.TemporaryDirectory(prefix='hc-start-') as runtime:
        register_session_execution_context('startup-boundary', SessionExecutionContext(
            env_set={'TEST_CUA_LOG': str(log)}, validate=lambda: not invalid.is_set(),
            computer_use=ComputerUseLaunchContext(driver_command=str(binary), private_daemon=True,
                                                  runtime_dir=runtime)))
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(registry.dispatch, 'computer_use',
                                      {'action': 'capture', 'app': 'screen'}, session_id='startup-boundary')
                try:
                    assert starting.wait(10), 'MCP startup never began'
                    invalid.set()
                finally:
                    release.set()
                result = json.loads(pending.result(timeout=15))
            assert result['code'] == 'policy_denied', result
            assert result['verdict']['decision'] == 'deny'
            assert result['verdict']['retryable'] is False
            assert 'validation failed' in result['message']
            assert 'install' not in result.get('hint', '')
            assert not any(r['kind'] == 'rpc' for r in map(json.loads, log.read_text().splitlines()))
        finally:
            release.set()
            release_computer_use_session('startup-boundary')
            remove_session_execution_context('startup-boundary')


@pytest.mark.linux_only
def test_private_runtime_stages_exact_approved_manifest(tmp_path):
    from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon
    from hermes_cli.session_execution import resolve_session_execution_context
    manifest = tmp_path / 'approved.json'
    manifest.write_bytes(b'{"version":3,"capabilities":[]}\n')
    with tempfile.TemporaryDirectory(prefix='hc-man-') as directory:
        runtime = Path(directory)
        register_session_execution_context('manifest', SessionExecutionContext(
            computer_use=ComputerUseLaunchContext(private_daemon=True, runtime_dir=directory)))
        daemon = _EmbeddedCuaDaemon(sys.executable, 'bounded', str(manifest),
                    execution_context=resolve_session_execution_context(session_id='manifest'))
        try:
            args = daemon._serve_args()
            staged = Path(args[args.index('--capability-manifest')+1])
            assert staged.parent == runtime
            assert staged.read_bytes() == manifest.read_bytes()
            assert staged.stat().st_mode & 0o777 == 0o600
        finally:
            daemon.stop()
            remove_session_execution_context('manifest')
        assert not staged.exists()


@pytest.mark.linux_only
@pytest.mark.parametrize('session_id', [None, 'rotated-owner'])
def test_registry_task_alias_uses_private_transport_without_moving_approvals(tmp_path, monkeypatch, session_id):
    import tools.computer_use_tool  # real tool registration
    from tools.registry import registry
    from tools.computer_use import tool
    from hermes_cli.session_execution import resolve_session_execution_context

    binary = tmp_path / 'protocol-driver'
    binary.write_text('#!' + sys.executable + '\n' + DRIVER)
    binary.chmod(0o700)
    log = tmp_path / 'calls.jsonl'
    approvals = []

    def approve(action, args, summary):
        approvals.append(action)
        return 'approve_session'

    monkeypatch.setattr(tool, '_approval_callback', approve)
    monkeypatch.setenv('HERMES_COMPUTER_USE_BACKEND', 'cua')
    valid = [True]
    with tempfile.TemporaryDirectory(prefix='hc-alias-') as runtime:
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(Path(runtime) / 'wl'))
        register_session_execution_context('transport-owner', SessionExecutionContext(
            env_set={'TEST_CUA_LOG': str(log), 'XDG_RUNTIME_DIR': runtime,
                     'WAYLAND_DISPLAY': 'wl', 'CUA_DRIVER_RS_ENABLE_WAYLAND': '1'},
            validate=lambda: valid[0],
            computer_use=ComputerUseLaunchContext(driver_command=str(binary), private_daemon=True,
                                                  runtime_dir=runtime, desktop_only=True)), task_ids=('transport-task',))
        try:
            result = registry.dispatch('computer_use', {'action': 'capture', 'app': 'screen'},
                                       session_id=session_id, task_id='transport-task')
            assert json.loads(result)['width'] > 0, result
            backend = tool._get_backend(session_id or '', task_id='transport-task')
            assert backend.execution_context is resolve_session_execution_context(task_id='transport-task')
            assert backend.permission_mode == 'standard'
            for _ in range(2):
                result = registry.dispatch('computer_use', {'action': 'click', 'coordinate': [1, 1]},
                                           session_id=session_id, task_id='transport-task')
                assert json.loads(result)['ok'], result
            assert approvals == ['click']
            assert ('click', 'background') in tool._always_allow[session_id or '']
            assert 'transport-owner' not in tool._always_allow
            assert 'transport-task' not in tool._always_allow
            before = log.read_text()
            valid[0] = False
            denied = json.loads(registry.dispatch('computer_use', {'action': 'list_apps'},
                                                 session_id=session_id, task_id='transport-task'))
            assert denied['code'] == 'policy_denied'
            assert denied['verdict']['retryable'] is False
            assert log.read_text() == before, 'invalid cached ownership must not send or restart'
        finally:
            tool.release_computer_use_session(session_id or '')
            remove_session_execution_context('transport-owner')
            sock.close()


@pytest.mark.parametrize('session_id', [None, 'rotated-owner', 'other-owner'])
@pytest.mark.parametrize('cached', [False, True], ids=['cold', 'cached-host'])
def test_registry_task_alias_denies_before_backend_start(monkeypatch, session_id, cached):
    import tools.computer_use_tool  # real tool registration
    from tools.registry import registry
    from tools.computer_use import tool

    # Even the fallback is inert: a broken lookup cannot touch a desktop.
    monkeypatch.setenv('HERMES_COMPUTER_USE_BACKEND', 'noop')
    starts = []
    monkeypatch.setattr(tool._NoopBackend, 'start', lambda self: starts.append(self))
    valid = [True]
    register_session_execution_context('task-owner', SessionExecutionContext(validate=lambda: valid[0]),
                                       task_ids=('actual-task',))
    try:
        if cached:
            assert json.loads(registry.dispatch('computer_use', {'action': 'list_apps'},
                                                session_id=session_id))['count'] == 0
            starts.clear()
        if session_id == 'other-owner':
            register_session_execution_context('other-owner', SessionExecutionContext())
        else:
            valid[0] = False
        result = json.loads(registry.dispatch('computer_use', {'action': 'list_apps'},
                                             session_id=session_id, task_id='actual-task'))
        assert result.get('code') == 'policy_denied', result
        assert result['ok'] is False
        assert result['verdict']['decision'] == 'deny'
        assert result['verdict']['retryable'] is False
        reason = 'conflicting' if session_id == 'other-owner' else 'validation failed'
        assert reason in result['message']
        assert not starts, 'ownership must be checked before starting even an inert fallback'
    finally:
        tool.release_computer_use_session(session_id or '')
        remove_session_execution_context('task-owner')
        remove_session_execution_context('other-owner')


@pytest.mark.linux_only
@pytest.mark.parametrize('invalid', ['driver', 'validation', 'missing-launch', 'shared-daemon', 'other-backend'])
def test_invalid_launch_context_is_non_retry_policy_denial(tmp_path, monkeypatch, invalid):
    import tools.computer_use_tool  # real tool registration
    from tools.registry import registry
    from tools.computer_use.tool import release_computer_use_session

    binary = tmp_path / 'driver'
    marker = tmp_path / 'spawned'
    binary.write_text(f'#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n')
    binary.chmod(0o700)
    valid = [True]
    register_session_execution_context('invalid-launch', SessionExecutionContext(
        computer_use=None if invalid == 'missing-launch' else ComputerUseLaunchContext(
            driver_command=str(binary), private_daemon=invalid != 'shared-daemon'),
        validate=lambda: valid[0]))
    try:
        if invalid == 'driver':
            binary.unlink()
        elif invalid == 'validation':
            valid[0] = False
        elif invalid == 'other-backend':
            monkeypatch.setenv('HERMES_COMPUTER_USE_BACKEND', 'noop')
        result = json.loads(registry.dispatch('computer_use',
            {'action': 'capture', 'app': 'screen'}, session_id='invalid-launch'))
        assert result['ok'] is False
        assert result['code'] == 'policy_denied'
        assert result['verdict']['decision'] == 'deny'
        assert result['verdict']['retryable'] is False
        assert 'do not retry' in result['verdict']['hint'].lower()
        assert 'recommended' not in result['verdict']
        assert 'escalation' not in result
        assert 'install' not in result.get('hint', '')
        reason = {'driver': 'driver executable unavailable', 'validation': 'validation failed',
                  'missing-launch': 'private_daemon', 'shared-daemon': 'private_daemon',
                  'other-backend': 'requires the cua backend'}[invalid]
        assert reason in result['message']
        assert not marker.exists()
    finally:
        release_computer_use_session('invalid-launch')
        remove_session_execution_context('invalid-launch')
