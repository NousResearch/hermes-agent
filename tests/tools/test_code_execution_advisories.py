"""Security guidance must survive execute_code without script printing (#138520)."""
import json

import pytest

from hermes_cli import plugins
import model_tools
from tools.code_kernel import shutdown_all_kernels


UNSAFE_LOOKING = 'def inert(source):\n    return ET.parse(source)\n'


@pytest.fixture
def security_guidance(monkeypatch, tmp_path):
    home = tmp_path / 'hermes-home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('plugins:\n  enabled: [security-guidance]\napprovals:\n  mode: off\n')
    monkeypatch.setattr(plugins, '_plugin_manager', plugins.PluginManager())
    plugins.discover_plugins()
    assert plugins.has_hook('transform_tool_result')
    monkeypatch.delenv('SECURITY_GUIDANCE_BLOCK', raising=False)
    monkeypatch.delenv('SECURITY_GUIDANCE_DISABLE', raising=False)
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    yield
    shutdown_all_kernels()


@pytest.mark.platforms('posix')
def test_kernel_delivers_unprinted_security_warning(security_guidance, monkeypatch, tmp_path):
    monkeypatch.setattr('tools.code_execution_tool._load_config', lambda: {'mode': 'project', 'timeout': 30})
    target = tmp_path / 'inert.py'
    code = f'import hermes_tools\nr = hermes_tools.write_file({str(target)!r}, {UNSAFE_LOOKING!r})\nassert isinstance(r, dict), r\n'
    # The real agent executor owns the outer post hook and suppresses its inner observer.
    with model_tools.suppress_post_tool_call_hook():
        result = json.loads(model_tools.handle_function_call('execute_code', {'code': code}, task_id='advisory-kernel', enabled_tools=['write_file']))
    assert result['status'] == 'success', result
    assert target.read_text() == UNSAFE_LOOKING
    assert result['output'] == ''
    assert result.get('tool_advisories'), result
    assert result['tool_advisories'][0]['tool'] == 'write_file'
    assert 'Security guidance' in result['tool_advisories'][0]['message']


def test_normal_write_preserves_warning(security_guidance, tmp_path):
    target = tmp_path / 'ordinary.py'
    result = model_tools.handle_function_call('write_file', {'path': str(target), 'content': UNSAFE_LOOKING})
    assert 'Security guidance' in result
    assert target.read_text() == UNSAFE_LOOKING


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('transport', ['file', 'socket', 'kernel-socket'])
@pytest.mark.parametrize('malformed', [False, True])
def test_real_rpc_json_and_single_write(security_guidance, monkeypatch, tmp_path, transport, malformed):
    import socket
    import subprocess
    import threading
    from pm.shell import bash
    from tools.code_execution_rpc import _rpc_poll_loop, _rpc_server_loop
    from tools.code_execution_tool import generate_hermes_tools_module
    from tools.code_kernel import CellAuthority
    from tools.registry import registry
    from tools.tool_advisories import current_advisory_sink, take_tool_advisories

    calls = []
    original_dispatch = registry.dispatch

    def dispatch(name, args, **kwargs):
        calls.append(name)
        result = original_dispatch(name, args, **kwargs)
        return 'not-json' if malformed else result

    monkeypatch.setattr(registry, 'dispatch', dispatch)
    monkeypatch.setenv('HERMES_RPC_TOKEN', 'fixture-token')
    monkeypatch.setenv('HERMES_RPC_PERSISTENT', '1')
    log, counter, stop = [], [0], threading.Event()
    server = None
    if transport == 'file':
        rpc = tmp_path / 'rpc'
        rpc.mkdir()
        monkeypatch.setenv('HERMES_RPC_DIR', str(rpc))

        class Shell:
            def execute(self, command, cwd=None, timeout=None, stdin_data=None):
                out = subprocess.run([bash(), '-c', command], cwd=cwd, timeout=timeout,
                                     input=stdin_data or '', capture_output=True, text=True, check=True)
                return {'output': out.stdout}

        worker = threading.Thread(target=_rpc_poll_loop, args=(Shell(), str(rpc), 'file-test', log, counter,
                                  5, frozenset({'write_file'}), stop, 'fixture-token'))
    else:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        monkeypatch.setenv('HERMES_RPC_SOCKET', f'tcp://127.0.0.1:{server.getsockname()[1]}')
        authority = CellAuthority('socket-test')
        worker = threading.Thread(target=_rpc_server_loop, args=(server, 'socket-test', log, counter, 5,
                                  frozenset({'write_file'}), stop, 'fixture-token'),
                                  kwargs={'dispatch': authority.dispatch} if transport == 'kernel-socket' else {})
    namespace = {}
    exec(generate_hermes_tools_module(['write_file'], transport='file' if transport == 'file' else 'socket'), namespace)
    worker.start()
    try:
        for index in range(2):
            target = tmp_path / f'inert-{index}.py'
            if malformed:
                with pytest.raises(json.JSONDecodeError):
                    namespace['write_file'](str(target), UNSAFE_LOOKING)
            else:
                result = namespace['write_file'](str(target), UNSAFE_LOOKING)
                assert type(result) is dict
                assert 'tool_advisories' not in result
            assert target.read_text() == UNSAFE_LOOKING
        assert calls == ['write_file', 'write_file']
        assert counter == [2]
        advisories = take_tool_advisories(log)
        assert len(advisories) == 2
        assert [a['call_index'] for a in advisories] == [1, 2]
        assert all(a['tool'] == 'write_file' and a['source'] == 'security-guidance' for a in advisories)
        assert take_tool_advisories(log) == []
        assert current_advisory_sink() is None
    finally:
        stop.set()
        if namespace.get('_sock'):
            namespace['_sock'].close()
        worker.join(timeout=10)
        if server:
            server.close()
        assert not worker.is_alive()


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('block', [False, True])
def test_kernel_call_isolation_and_block_mode(security_guidance, monkeypatch, tmp_path, block):
    from tools.code_kernel import _KERNELS
    monkeypatch.setattr('tools.code_execution_tool._load_config', lambda: {'mode': 'project', 'timeout': 30})
    if block:
        monkeypatch.setenv('SECURITY_GUIDANCE_BLOCK', '1')
    target = tmp_path / 'blocked-or-warned.py'

    def run(code):
        return json.loads(model_tools.handle_function_call('execute_code', {'code': code}, task_id='isolation', enabled_tools=['write_file']))

    first = run(f'import hermes_tools\nhermes_tools.write_file({str(target)!r}, {UNSAFE_LOOKING!r})\n')
    assert first['status'] == 'success', first
    assert target.exists() is not block
    if block:
        assert first.get('tool_advisories') is None
        assert 'refused this write' in first['tool_errors'][0]['error']
    else:
        assert len(first['tool_advisories']) == 1
    second = run('assert isinstance(hermes_tools.write_file(' + repr(str(tmp_path / 'safe.py')) + ', "safe = 1\\n"), dict)')
    assert second['kernel']['reused'] is True
    assert second['status'] == 'success', second
    assert 'tool_advisories' not in second
    assert all('_advisory_sink' not in e for kernel in _KERNELS.values() for e in kernel.tool_call_log)


def test_advisory_scope_bounds_concurrency_and_failure():
    from concurrent.futures import ThreadPoolExecutor
    from tools.tool_advisories import (AdvisorySink, ToolAdvisory, bind_advisory_sink,
                                      current_advisory_sink, emit_tool_advisory)

    def collect(index):
        sink = AdvisorySink(2)
        with pytest.raises(RuntimeError), bind_advisory_sink(sink):
            warning = ToolAdvisory('test', str(index), 'x' * 3000)
            emit_tool_advisory(warning)
            emit_tool_advisory(warning)
            emit_tool_advisory(ToolAdvisory('test', 'other', 'second'))
            emit_tool_advisory(ToolAdvisory('test', 'overflow', 'omitted'))
            raise RuntimeError('fixture')
        assert current_advisory_sink() is None
        result = sink.finish()
        sink.emit(warning)
        assert sink.finish() == []
        assert len(result) == 2
        assert len(result[0]['message']) == 2048
        return result[0]['code']

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(collect, range(8))) == [str(i) for i in range(8)]


def test_unrelated_transforms_are_not_advisories(monkeypatch):
    import time
    from tools.code_execution_rpc import _default_dispatch, _handle_rpc_request
    from tools.registry import registry
    from tools.tool_advisories import current_advisory_sink, take_tool_advisories
    monkeypatch.setattr(registry, 'dispatch', lambda *a, **kw: '{"value": [1, 2]}')
    monkeypatch.setattr(plugins, 'has_hook', lambda name: True)
    monkeypatch.setattr(plugins, 'invoke_hook', lambda name, **kw: ['arbitrary redaction text'])
    log = []
    result = _handle_rpc_request({'tool': 'write_file', 'args': {}}, allowed_tools=frozenset({'write_file'}),
                               tool_call_counter=[0], max_tool_calls=2, dispatch=_default_dispatch('test'),
                               tool_call_log=log, call_start=time.monotonic(), where='test')
    assert json.loads(result) == {'value': [1, 2]}
    assert take_tool_advisories(log) == []
    assert current_advisory_sink() is None
    assert model_tools.handle_function_call('write_file', {}) == 'arbitrary redaction text'


def test_remote_result_preserves_structured_advisories():
    import time
    from tools.code_execution_tool import _finish_remote_kernel_result
    item = {'tool': 'write_file', 'call_index': 1, 'source': 'security-guidance', 'code': 'test', 'message': 'fixture'}
    result = json.loads(_finish_remote_kernel_result({'status': 'success', 'stdout': '', 'tool_advisories': [item]},
                                                    timeout=30, exec_start=time.monotonic()))
    assert result['tool_advisories'] == [item]


@pytest.mark.platforms('posix')
def test_kernel_host_failure_retains_completed_write_warning(security_guidance, monkeypatch, tmp_path):
    monkeypatch.setattr('tools.code_execution_tool._load_config', lambda: {'mode': 'project', 'timeout': 30})
    def fail_result(*args, **kwargs):
        raise RuntimeError('fixture result failure')
    monkeypatch.setattr('tools.code_kernel._cell_result', fail_result)
    target = tmp_path / 'host-failure.py'
    code = f'import hermes_tools\nhermes_tools.write_file({str(target)!r}, {UNSAFE_LOOKING!r})'
    result = json.loads(model_tools.handle_function_call('execute_code', {'code': code}, task_id='host-failure', enabled_tools=['write_file']))
    assert target.read_text() == UNSAFE_LOOKING
    assert result['status'] == 'error'
    assert 'fixture result failure' in result['error']
    assert len(result['tool_advisories']) == 1


def test_remote_host_failure_retains_completed_write_warning(security_guidance, monkeypatch, tmp_path):
    import threading
    import time
    from tools import code_execution_tool as cet
    from tools.code_execution_rpc import _default_dispatch, _handle_rpc_request
    captured, ready = {}, threading.Event()
    target = tmp_path / 'remote-failure.py'

    def poll(env, rpc, task, log, counter, *args):
        captured.update(log=log, counter=counter)
        ready.set()

    class Remote:
        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, **kwargs):
            if command == 'RUN_FIXTURE':
                assert ready.wait(5)
                _handle_rpc_request({'tool': 'write_file', 'args': {'path': str(target), 'content': UNSAFE_LOOKING}},
                    allowed_tools=frozenset({'write_file'}), tool_call_counter=captured['counter'],
                    max_tool_calls=5, dispatch=_default_dispatch('remote-failure'), tool_call_log=captured['log'],
                    call_start=time.monotonic(), where='fixture')
                raise RuntimeError('fixture transport failure')
            return {'output': '', 'returncode': 0}

    monkeypatch.setattr(cet, '_rpc_poll_loop', poll)
    monkeypatch.setattr(cet, '_ship_file_to_remote', lambda *a, **kw: None)
    monkeypatch.setattr(cet, '_ship_env_file_and_launch', lambda *a, **kw: 'RUN_FIXTURE')
    result = json.loads(cet._run_remote_per_call(Remote(), 'ssh', 'unused', 'remote-failure',
                       frozenset({'write_file'}), timeout=30, max_tool_calls=5, exec_start=time.monotonic()))
    assert target.read_text() == UNSAFE_LOOKING
    assert 'fixture transport failure' in result['error']
    assert len(result['tool_advisories']) == 1
    assert all('_advisory_sink' not in e for e in captured['log'])


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('persistent', [False, True])
def test_remote_shell_unprinted_warning(security_guidance, tmp_path, persistent):
    import time
    from tests.tools._child_env_fixtures import RunToCompletionEnv
    from tools.code_execution_tool import _finish_remote_kernel_result, _run_remote_per_call
    from tools.code_kernel_remote import execute_in_remote_kernel, shutdown_all_remote_kernels
    env = RunToCompletionEnv(tmp_path)
    target = tmp_path / 'remote-inert.py'
    code = f'import hermes_tools\nr = hermes_tools.write_file({str(target)!r}, {UNSAFE_LOOKING!r})\nassert type(r) is dict'
    try:
        with model_tools.suppress_post_tool_call_hook():
            if persistent:
                raw = execute_in_remote_kernel(code, env=env, env_type='ssh', task_env_id='remote-shell',
                    sandbox_tools=frozenset({'write_file'}), timeout=30, max_tool_calls=5, reset=False)
                assert raw is not None
                result = json.loads(_finish_remote_kernel_result(raw, timeout=30, exec_start=time.monotonic()))
            else:
                result = json.loads(_run_remote_per_call(env, 'ssh', code, 'remote-shell', frozenset({'write_file'}),
                                    timeout=30, max_tool_calls=5, exec_start=time.monotonic()))
        assert result['status'] == 'success', result
        assert result['output'] == ''
        assert result['tool_calls_made'] == 1
        assert len(result['tool_advisories']) == 1
        assert target.read_text() == UNSAFE_LOOKING
        if persistent:
            second = execute_in_remote_kernel('assert type(r) is dict', env=env, env_type='ssh', task_env_id='remote-shell',
                sandbox_tools=frozenset({'write_file'}), timeout=30, max_tool_calls=5, reset=False)
            assert second['kernel']['reused'] is True
            assert 'tool_advisories' not in second
    finally:
        shutdown_all_remote_kernels()


@pytest.mark.parametrize('kernel', [False, True])
def test_inner_lifecycle_and_execution_middleware(security_guidance, monkeypatch, tmp_path, kernel):
    import time
    from hermes_cli import middleware
    from tools.code_execution_rpc import _default_dispatch, _handle_rpc_request
    from tools.code_kernel import CellAuthority
    from tools.registry import registry
    events = []
    original_hook, original_middleware, original_dispatch = plugins.invoke_hook, middleware.run_tool_execution_middleware, registry.dispatch

    def hook(name, **kw):
        if name in ('pre_tool_call', 'post_tool_call'):
            events.append(name)
        return original_hook(name, **kw)

    def execution(*args, **kwargs):
        events.append('middleware')
        return original_middleware(*args, **kwargs)

    def dispatch(*args, **kwargs):
        events.append('dispatch')
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(plugins, 'invoke_hook', hook)
    monkeypatch.setattr(middleware, 'run_tool_execution_middleware', execution)
    monkeypatch.setattr(registry, 'dispatch', dispatch)
    log = []
    with model_tools.suppress_post_tool_call_hook():
        inner = CellAuthority('lifecycle').dispatch if kernel else _default_dispatch('lifecycle')
        result = _handle_rpc_request({'tool': 'write_file', 'args': {'path': str(tmp_path / 'lifecycle.py'), 'content': UNSAFE_LOOKING}},
            allowed_tools=frozenset({'write_file'}), tool_call_counter=[0], max_tool_calls=5, dispatch=inner,
            tool_call_log=log, call_start=time.monotonic(), where='fixture')
        assert model_tools._post_tool_call_hook_suppressed.get() is True
    assert isinstance(json.loads(result), dict)
    assert events == ['pre_tool_call', 'middleware', 'dispatch', 'post_tool_call']
    from tools.tool_advisories import take_tool_advisories
    assert len(take_tool_advisories(log)) == 1


def test_execution_budget_and_dispatch_failure_cleanup():
    import time
    from tools.code_execution_rpc import _handle_rpc_request
    from tools.tool_advisories import ToolAdvisory, current_advisory_sink, emit_tool_advisory, take_tool_advisories
    log, counter = [], [0]

    def dispatch(name, args):
        emit_tool_advisory(ToolAdvisory('fixture', 'warning', 'Review construct'))
        raise RuntimeError('fixture failure')

    for _ in range(40):
        result = _handle_rpc_request({'tool': 'write_file', 'args': {}}, allowed_tools=frozenset({'write_file'}),
            tool_call_counter=counter, max_tool_calls=40, dispatch=dispatch, tool_call_log=log,
            call_start=time.monotonic(), where='fixture')
        assert json.loads(result)['error'] == 'fixture failure'
        assert current_advisory_sink() is None
    assert counter == [40]
    assert len(take_tool_advisories(log)) == 32
    assert take_tool_advisories(log) == []


def test_inflight_rpc_cannot_leak_into_next_cell():
    import threading
    import time
    from tools.code_execution_rpc import _handle_rpc_request
    from tools.tool_advisories import ToolAdvisory, emit_tool_advisory, take_tool_advisories
    started, release = threading.Event(), threading.Event()
    log = []

    def dispatch(name, args):
        started.set()
        assert release.wait(5)
        emit_tool_advisory(ToolAdvisory('fixture', 'late', 'belongs to cell A'))
        return '{}'

    worker = threading.Thread(target=_handle_rpc_request, args=({'tool': 'write_file', 'args': {}},), kwargs={
        'allowed_tools': frozenset({'write_file'}), 'tool_call_counter': [0], 'max_tool_calls': 5,
        'dispatch': dispatch, 'tool_call_log': log, 'call_start': time.monotonic(), 'where': 'fixture'})
    worker.start()
    try:
        assert started.wait(5)
        # A settles with an in-flight background tool; B starts before A's tool finishes.
        assert take_tool_advisories(log) == []
        second_cell_start = len(log)
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert take_tool_advisories(log, second_cell_start) == []
    assert take_tool_advisories(log) == []
