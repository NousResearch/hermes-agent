"""Foreground follow-up refusal preserves the original stream, including split UTF-8."""
import shlex
import sys
import threading

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.waiting import wait_until
from tools import process_registry as module
from tools.environments import base, base_output
from tools.environments.local import LocalEnvironment
from tools.interrupt import consume_yield, request_yield
from tools.process_registry_job import job_quiescent
from tools.terminal_tool_background import yield_to_background_handler

pytestmark = pytest.mark.platforms('linux')


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    monkeypatch.setattr('gateway.session_context.async_delivery_supported', lambda: False)
    try:
        yield registry
    finally:
        for session in list(registry._running.values()):
            registry.kill_process(session.id, source='session.interrupt')
        reset_hermes_home_override(token)


def handler(command, root, route='route'):
    return yield_to_background_handler(command=command, env_type='local', cwd=str(root),
                                      effective_task_id='task', task_id='task', session_key=route)


@pytest.mark.parametrize('completed', [True, False], ids=['completed-job', 'foreign-route-refused'])
def test_refused_yield_preserves_cursor_and_split_character(tmp_path, monkeypatch, registry, completed):
    gate = tmp_path / 'gate'
    payload = f'''import os,time
from pathlib import Path
os.write(1,b'unique_prefix\\n\\xe6')
end=time.monotonic()+12
while not Path({str(gate)!r}).exists() and time.monotonic()<end: time.sleep(.01)
os.write(1,b'\\x97\\xa5-unique_tail\\n')
raise SystemExit(7)
'''
    command = shlex.join([sys.executable, '-c', payload])
    actual = handler(command, tmp_path)
    refused = actual if completed else handler(command, tmp_path, route='foreign')
    entered, release = threading.Event(), threading.Event()
    def held(proc, output):
        entered.set()
        assert release.wait(5)
        return refused(proc, output)
    held.prepare_local = actual.prepare_local
    collectors, create = [], base._new_output_collector
    def capture(*args, **kwargs):
        output = create(*args, **kwargs)
        collectors.append(output)
        return output
    monkeypatch.setattr(base, '_new_output_collector', capture)
    env, result = LocalEnvironment(cwd=str(tmp_path)), {}
    worker = threading.Thread(target=lambda: result.update(env.execute(command, timeout=15, yield_handler=held)))
    worker.start()
    try:
        wait_until(lambda: any('unique_prefix' in c.render() for c in collectors), timeout=5)
        request_yield(worker.ident)
        assert entered.wait(5)
        session = next(s for s in registry._running.values() if s.command == command)
        if completed:
            gate.touch()
            assert session._completion_event.wait(5)
        release.set()
        if not completed:
            gate.touch()
        worker.join(10)
        assert not worker.is_alive()
        assert result['output'] == 'unique_prefix\n日-unique_tail\n' and result['returncode'] == 7
        assert 'yielded_session_id' not in result
        assert session._completion_event.wait(5)
        assert (session.owner_task_id, session.session_key) == ('task', 'route')
        assert session.persist_on_release and not session.notify_on_complete
        # The native result also retains the shell's cwd observation trailer;
        # LocalEnvironment removes that trailer only from its inline response.
        assert session.output_buffer.startswith(result['output']) and session.exit_code == 7
        assert session.output_buffer.count('unique_prefix') == session.output_buffer.count('日-unique_tail') == 1
        assert job_quiescent(session) and session._result_complete
        assert registry.poll(session.id)['exit_code'] == 7
        assert registry.completion_queue.empty()
    finally:
        release.set()
        gate.touch()
        consume_yield(worker.ident)
        worker.join(15)
        env.cleanup()


def test_stop_during_iterator_read_keeps_the_fetched_piece(tmp_path, monkeypatch, registry):
    command = shlex.join([sys.executable, '-c', "print('read-boundary-日')"])
    env = LocalEnvironment(cwd=str(tmp_path))
    proc = env._run_bash(command, native_session=handler(command, tmp_path).prepare_local())
    session, stop = proc._hermes_job_session, threading.Event()
    output = base_output._new_output_collector(proc, False)
    iterate = type(proc.stdout).__iter__
    def stop_after_fetch(stream):
        for piece in iterate(stream):
            if piece:
                stop.set()
            yield piece
    monkeypatch.setattr(type(proc.stdout), '__iter__', stop_after_fetch)
    try:
        assert session._completion_event.wait(5)
        base_output._drain_stdout(proc, output, stop)
        assert output.render() == 'read-boundary-日\n'
        stop.clear()
        base_output._drain_stdout(proc, output, stop)
        assert output.render() == session.output_buffer == 'read-boundary-日\n'
        assert proc.poll() == session.exit_code == 0 and job_quiescent(session)
    finally:
        proc.stdout.close()
        env.cleanup()


@pytest.mark.parametrize('mode', ['stamp', 'checkpoint', 'arm-false', 'arm-raise', 'armed'])
def test_postadoption_setup_commits_delivery_or_keeps_foreground(tmp_path, monkeypatch, registry, mode):
    from types import SimpleNamespace
    import json
    import utils
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import terminal_tool_background as background

    gate, starts = tmp_path / 'release', tmp_path / 'starts'
    code = f'''from pathlib import Path
import time
Path({str(starts)!r}).open('a').write('once\\n')
print('prefix',flush=True)
end=time.monotonic()+12
while not Path({str(gate)!r}).exists() and time.monotonic()<end: time.sleep(.01)
print('tail',flush=True)
raise SystemExit(7)
'''
    command = shlex.join([sys.executable, '-c', code])
    monkeypatch.setattr('gateway.session_context.async_delivery_supported', lambda: True)
    actual = handler(command, tmp_path)
    pending = {'session_id': 'foreign-pending'}
    registry.pending_watchers.append(pending)
    stamped, write_failed, fail_write = threading.Event(), threading.Event(), threading.Event()
    handoff_requested = threading.Event()
    stages, armed = [], []
    stamp = background._stamp_gateway_routing
    def prepare(*args):
        staged = args[0]
        stages.append(staged)
        stamp(*args)
        if not handoff_requested.is_set():
            return
        stamped.set()
        if mode == 'stamp':
            staged.watcher_chat_id = 'refused-route'
            raise RuntimeError('routing setup refused after staging')
        if mode == 'checkpoint':
            fail_write.set()
    monkeypatch.setattr(background, '_stamp_gateway_routing', prepare)
    atomic = utils.atomic_json_write
    def write(path, data, *args, **kwargs):
        if path.name == 'processes.json' and fail_write.is_set():
            write_failed.set()
            raise PermissionError('actual handoff checkpoint unavailable')
        return atomic(path, data, *args, **kwargs)
    monkeypatch.setattr(utils, 'atomic_json_write', write)
    def arm(watcher):
        own = registry.get(watcher['session_id'])
        assert registry._lock.acquire(blocking=False), 'watcher callback ran under registry lock'
        registry._lock.release()
        assert own._lock.acquire(blocking=False), 'watcher callback ran under session lock'
        own._lock.release()
        armed.append(watcher)
        if mode == 'arm-raise':
            raise RuntimeError('gateway arming unavailable')
        return mode == 'armed'
    monkeypatch.setitem(sys.modules, 'gateway.run', SimpleNamespace(
        _gateway_runner_ref=lambda: SimpleNamespace(arm_process_watcher=arm)))
    collectors, create = [], base._new_output_collector
    def capture(*args, **kwargs):
        output = create(*args, **kwargs)
        collectors.append(output)
        return output
    monkeypatch.setattr(base, '_new_output_collector', capture)
    env, result = LocalEnvironment(cwd=str(tmp_path)), {}
    def execute():
        tokens = set_session_vars(platform='slack', chat_id='own-chat', thread_id='own-thread',
                                  session_key='route', session_id='native-parent')
        try:
            result.update(env.execute(command, timeout=15, yield_handler=actual))
        finally:
            clear_session_vars(tokens)
    worker = threading.Thread(target=execute)
    worker.start()
    try:
        wait_until(lambda: any('prefix' in c.render() for c in collectors), timeout=5)
        own = next(s for s in registry._running.values() if s.command == command)
        handoff_requested.set()
        request_yield(worker.ident)
        assert stamped.wait(5)
        if mode in ('stamp', 'checkpoint'):
            if mode == 'checkpoint':
                assert write_failed.wait(5)
            with registry._lock, own._lock:
                assert own.persist_on_release and not own.notify_on_complete
                assert own.watcher_interval == 0 and own.watcher_chat_id == 'own-chat'
                assert getattr(own, 'approval_task_reservation', None) is None
                assert own.process.poll() is None and not own.exited
            fail_write.clear()
        else:
            worker.join(5)
            assert not worker.is_alive() and result['yielded_session_id'] == own.id
            assert not own.persist_on_release and own.notify_on_complete
            assert own.watcher_interval == 5
            assert len(armed) == 1
        row = next(row for row in json.loads((tmp_path / 'processes.json').read_text())
                   if row['session_id'] == own.id)
        accepted = mode not in ('stamp', 'checkpoint')
        assert row['notify_on_complete'] is accepted and row['persist_on_release'] is not accepted
        assert row['watcher_interval'] == (5 if accepted else 0)
        assert row['watcher_chat_id'] == 'own-chat' and row['parent_session_id'] == 'native-parent'
        gate.touch()
        worker.join(10)
        assert not worker.is_alive() and own._completion_event.wait(5)
        assert starts.read_text() == 'once\n' and own.exit_code == 7
        assert own.output_buffer.startswith('prefix\ntail\n')
        assert own.output_buffer.count('prefix') == own.output_buffer.count('tail') == 1
        assert (own.owner_task_id, own.session_key, own.parent_session_id) == ('task', 'route', 'native-parent')
        assert own._result_complete and job_quiescent(own)
        assert registry.pending_watchers[0] is pending
        if mode in ('stamp', 'checkpoint'):
            assert result['returncode'] == 7 and result['output'] == 'prefix\ntail\n'
            assert 'yielded_session_id' not in result
            assert not armed
            assert len(registry.pending_watchers) == 1 and registry.completion_queue.empty()
        else:
            assert result['output'] == 'prefix\n'
            assert len(registry.pending_watchers) == (1 if mode == 'armed' else 2)
            notice = registry.completion_queue.get_nowait()
            assert notice['session_id'] == own.id and notice['exit_code'] == 7
            assert registry.completion_queue.empty()
    finally:
        fail_write.clear()
        gate.touch()
        consume_yield(worker.ident)
        worker.join(15)
        env.cleanup()
