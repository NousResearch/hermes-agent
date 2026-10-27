"""Foreground commands keep native lifetime custody across follow-up and worker loss."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading

import pytest

from tests.waiting import wait_until
from tools import process_registry as module
from tools.environments.local import LocalEnvironment
from tools.interrupt import request_yield, consume_yield, set_interrupt
from tools.process_registry_job import job_quiescent
from tools.terminal_tool_background import yield_to_background_handler

pytestmark = pytest.mark.platforms('linux')


def handler(command, root, task='foreground', route='route'):
    return yield_to_background_handler(command=command, env_type='local', cwd=str(root),
                                      effective_task_id=task, task_id=task, session_key=route)


def capture_foreground_output(monkeypatch):
    """Observe real capture readiness; command output and drain readiness differ."""
    from tools.environments import base
    create, collectors = base._new_output_collector, []
    def observed(*args, **kwargs):
        collector = create(*args, **kwargs)
        collectors.append(collector)
        return collector
    monkeypatch.setattr(base, '_new_output_collector', observed)
    return collectors


@pytest.fixture
def scope_reaper():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    adopted = []
    try:
        yield adopted
    finally:
        for pid in adopted:
            os.waitpid(pid, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.fixture
def registry(monkeypatch, scope_reaper):
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    monkeypatch.setattr('gateway.session_context.async_delivery_supported', lambda: False)
    yield registry
    for session in list(registry._running.values()):
        registry.kill_process(session.id)


@pytest.mark.parametrize('reload_owner', [False, True], ids=['live', 'cold'])
def test_foreground_followup_adopts_existing_native_scope(tmp_path, registry, reload_owner, monkeypatch):
    ready, gate, effect = (tmp_path / name for name in ('ready', 'gate', 'old-effect'))
    child = f'''import os, time
from pathlib import Path
os.setsid()
Path({str(ready)!r}).write_text(str(os.getpid()))
end=time.monotonic()+15
while time.monotonic()<end and not Path({str(gate)!r}).exists(): time.sleep(.01)
if Path({str(gate)!r}).exists(): Path({str(effect)!r}).touch()
'''
    parent = f"import subprocess,sys,time; print('partial',flush=True); subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(15)"
    command = shlex.join([sys.executable, '-c', parent])
    env = LocalEnvironment(cwd=str(tmp_path))
    collectors = capture_foreground_output(monkeypatch)
    result = {}
    thread = threading.Thread(target=lambda: result.update(env.execute(
        command, timeout=20, yield_handler=handler(command, tmp_path))))
    thread.start()
    try:
        wait_until(ready.exists, what='actual detached command child')
        wait_until(lambda: any('partial' in output.render() for output in collectors),
                   what='foreground partial output actually captured')
        request_yield(thread.ident)
        thread.join(5)
        assert not thread.is_alive(), result
        session = registry.get(result['yielded_session_id'])
        assert 'partial' in result['output']
        assert session.native_job_scope, 'fresh foreground yield lost native scope'
        assert session.output_log and not session.persist_on_release
        pid = int(ready.read_text())
        if reload_owner:
            cold = module.ProcessRegistry()
            assert cold.recover_from_checkpoint() == 1
            session = cold.get(session.id)
            assert session.native_job_scope and session.output_log
            registry = cold
        killed = registry.kill_process(session.id)
        assert killed['status'] == 'killed', killed
        assert job_quiescent(session)
        assert not Path(f'/proc/{pid}').exists()
        gate.touch()
        assert not effect.exists()
        assert session.output_buffer.count('partial') == 1
    finally:
        consume_yield(thread.ident)
        if result.get('yielded_session_id'):
            registry.kill_process(result['yielded_session_id'])
        thread.join(20)
        env.cleanup()


def test_foreground_primary_exit_preserves_detached_service_and_can_stop_it(tmp_path, registry):
    ready = tmp_path / 'service'
    child = f"import os,time; from pathlib import Path; os.setsid(); Path({str(ready)!r}).touch(); time.sleep(15)"
    parent = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}]); print('shell finished'); raise SystemExit(7)"
    command = shlex.join([sys.executable, '-c', parent])
    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = env.execute(command, timeout=10, yield_handler=handler(command, tmp_path))
        assert result['returncode'] == 7 and 'shell finished' in result['output']
        wait_until(ready.exists)
        session = next(s for s in registry._running.values() if s.command == command)
        assert session.persist_on_release and not session.notify_on_complete
        assert not job_quiescent(session) and session.process.poll() is None
        assert registry.kill_all(task_id='foreground') == 0, 'ordinary retirement preserves service lifetime'
        assert registry.kill_process(session.id)['status'] == 'killed'
        assert job_quiescent(session)
    finally:
        env.cleanup()


def test_foreground_preserves_stdin_env_cwd_full_output_and_primary_status(tmp_path, registry):
    sub = tmp_path / 'sub'
    sub.mkdir()
    payload = "import os,sys; print(os.getcwd()); print(os.environ['FOREGROUND_TEST']); print(sys.stdin.read()); print('日'*75000)"
    command = 'cd ' + shlex.quote(str(sub)) + '; export FOREGROUND_TEST=kept; ' + shlex.join([sys.executable, '-c', payload])
    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = env.execute(command, stdin_data='stdin retained', timeout=15,
                             yield_handler=handler(command, tmp_path))
        assert result['returncode'] == 0
        assert result['output'].count('日') == 75000
        assert 'stdin retained' in result['output'] and 'kept' in result['output']
        assert str(sub) in result['output'] and env.cwd == str(sub)
        session = next(s for s in list(registry._finished.values()) + list(registry._running.values())
                       if s.command == command)
        assert session._completion_event.wait(5)
        assert job_quiescent(session) and session.exit_code == 0
        assert registry.completion_queue.empty()
    finally:
        env.cleanup()


def test_foreground_interruption_reaps_scope_and_preserves_unrelated_job(tmp_path, registry):
    ready = tmp_path / 'ready'
    payload = f"import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path({str(ready)!r}).touch(); time.sleep(15)"
    command = shlex.join([sys.executable, '-c', payload])
    neighbor = registry.spawn_local(shlex.join([sys.executable, '-c', 'import time; time.sleep(15)']),
                                    task_id='foreign', session_key='foreign')
    env = LocalEnvironment(cwd=str(tmp_path))
    result = {}
    thread = threading.Thread(target=lambda: result.update(env.execute(
        command, timeout=20, yield_handler=handler(command, tmp_path))))
    thread.start()
    try:
        wait_until(ready.exists)
        set_interrupt(True, thread_id=thread.ident)
        thread.join(10)
        assert not thread.is_alive() and result['hermes_interrupted']
        session = next(s for s in registry._finished.values() if s.command == command)
        assert job_quiescent(session)
        assert neighbor.process.poll() is None
    finally:
        set_interrupt(False, thread_id=thread.ident)
        thread.join(20)
        env.cleanup()


@pytest.mark.parametrize('hold_reader', [False, True], ids=['ordinary-reader', 'held-original-reader'])
def test_worker_exit_after_foreground_yield_keeps_scope_output_and_cold_cleanup(
        tmp_path, registry, scope_reaper, monkeypatch, hold_reader):
    ready = tmp_path / 'ready'
    code = '''import json,os,shlex,sys,threading,time
from pathlib import Path
from tools import process_registry as m
from tools.environments.local import LocalEnvironment
from tools.terminal_tool_background import yield_to_background_handler
from tools.interrupt import request_yield
root,ready=sys.argv[1:]
command=shlex.join([sys.executable,'-c',"import time; from pathlib import Path; print('before worker exit',flush=True); Path("+repr(ready)+").touch(); time.sleep(15)"])
env=LocalEnvironment(cwd=root)
handler=yield_to_background_handler(command=command,env_type='local',cwd=root,effective_task_id='task',task_id='task',session_key='route')
result={}
t=threading.Thread(target=lambda:result.update(env.execute(command,timeout=20,yield_handler=handler)))
t.start()
end=time.monotonic()+8
while not Path(ready).exists() and time.monotonic()<end: time.sleep(.01)
request_yield(t.ident);t.join(5)
assert not t.is_alive(),result
print(json.dumps(result),flush=True)
os._exit(0)
'''
    worker = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(ready)],
                            cwd=Path(module.__file__).parents[1], env=os.environ.copy(),
                            capture_output=True, text=True, timeout=12, check=False)
    assert worker.returncode == 0, worker.stderr
    receipt = json.loads(worker.stdout.splitlines()[-1])
    assert registry.recover_from_checkpoint() == 1
    session = registry.get(receipt['yielded_session_id'])
    scope_reaper.append(session.pid)
    assert session.native_job_scope and session.output_log
    from tools.process_registry_handoff import ownership_identity
    from tools.process_registry_job import read_job_receipt
    identity = ownership_identity(session)
    release, entered, refused = threading.Event(), threading.Event(), []
    move, refuse = registry._move_to_finished, registry._refuse_backend_stop
    def held_move(current):
        if hold_reader and current is session:
            entered.set()
            assert release.wait(10)
        return move(current)
    def observe_refusal(current, consume_output):
        if current is session:
            refused.append((job_quiescent(current), current._completion_event.is_set(),
                            current._result_complete, current._result_persisted_generation))
        return refuse(current, consume_output)
    monkeypatch.setattr(registry, '_move_to_finished', held_move)
    monkeypatch.setattr(registry, '_refuse_backend_stop', observe_refusal)
    neighbor = registry.spawn_local(shlex.join([sys.executable, '-c', 'import time; time.sleep(15)']),
                                    task_id='foreign', session_key='foreign')
    try:
        verdict = registry.kill_process(session.id, expected_ownership=identity)
        if hold_reader:
            assert entered.is_set() and verdict['error'] == 'process_stop_unknown'
        if verdict['status'] == 'error':
            # Kernel cleanup may precede the original reader's final publication.
            # A truthful refusal must retain custody until that exact reader finishes.
            assert verdict['error'] == 'process_stop_unknown' and refused
            quiescent, published, complete, persisted = refused[-1]
            assert quiescent and (not published or not complete or persisted != session.ownership_generation)
            assert job_quiescent(session) and session._scope_stop_pending
            assert session.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
            assert neighbor.process.poll() is None
            release.set()
            assert session._completion_event.wait(5)
            assert session._result_complete and session._result_persisted_generation == session.ownership_generation
            assert ownership_identity(session) == identity
            verdict = registry.kill_process(session.id, expected_ownership=identity)
            assert verdict['status'] == 'already_exited', (verdict, read_job_receipt(session), session)
        else:
            assert verdict['status'] == 'killed', (verdict, read_job_receipt(session), session)
        assert job_quiescent(session) and not session._scope_stop_pending
        assert session.output_buffer == 'before worker exit\n' and session.exit_code == -15
        assert session.id not in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        assert neighbor.process.poll() is None
    finally:
        release.set()
        registry.kill_process(neighbor.id)
        assert neighbor._completion_event.wait(5)


def test_old_foreground_handle_cannot_stop_a_handed_off_job(tmp_path, registry, monkeypatch):
    ready = tmp_path / 'handoff-ready'
    payload = f"import time; from pathlib import Path; Path({str(ready)!r}).touch(); time.sleep(15)"
    command = shlex.join([sys.executable, '-c', payload])
    env = LocalEnvironment(cwd=str(tmp_path))
    launch, captured, result = env._run_bash, [], {}
    def capture(*args, **kwargs):
        proc = launch(*args, **kwargs)
        captured.append(proc)
        return proc
    monkeypatch.setattr(env, '_run_bash', capture)
    thread = threading.Thread(target=lambda: result.update(env.execute(
        command, timeout=20, yield_handler=handler(command, tmp_path))))
    thread.start()
    try:
        wait_until(ready.exists)
        request_yield(thread.ident)
        thread.join(5)
        assert not thread.is_alive(), result
        session = registry.get(result['yielded_session_id'])
        assert registry.transfer_ownership(session.id, from_owner='foreground', to_owner='parent',
            to_task_id='parent', to_session_key='parent-route', to_parent_session_id='native-parent',
            note='continue', child_stop_event=None) is session
        refused = captured[0].kill()
        assert refused['status'] == 'ownership_changed' and session.process.poll() is None
        assert not job_quiescent(session)
        assert registry.kill_process(session.id)['status'] == 'killed'
        assert job_quiescent(session)
    finally:
        consume_yield(thread.ident)
        thread.join(20)
        env.cleanup()


def test_foreground_registration_failure_never_replays_original_command(tmp_path, registry, monkeypatch):
    starts = tmp_path / 'starts'
    payload = f"import os,time; from pathlib import Path; Path({str(starts)!r}).open('a').write(str(os.getpid())+'\\n'); time.sleep(15)"
    command = shlex.join([sys.executable, '-c', payload])
    track, captured = registry._track_started, []
    def fail_once(session, *args, **kwargs):
        captured.append(session)
        if len(captured) == 1:
            wait_until(starts.exists)
            raise RuntimeError('reader failed after native foreground start')
        return track(session, *args, **kwargs)
    monkeypatch.setattr(registry, '_track_started', fail_once)
    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        from tools.environments.base import CommandExecutionUnknown
        with pytest.raises(CommandExecutionUnknown) as refused:
            env.execute(command, timeout=20, yield_handler=handler(command, tmp_path))
        session = captured[0]
        assert refused.value.session_id == session.id
        assert 'after native foreground start' in str(refused.value.__cause__)
        assert session._completion_event.wait(10)
        assert job_quiescent(session)
        assert len(starts.read_text().splitlines()) == 1
        assert all(item is session for item in captured)
    finally:
        env.cleanup()


def test_native_execute_recovers_deleted_working_directory(tmp_path, registry):
    import shutil
    working = tmp_path / 'working'
    working.mkdir()
    env = LocalEnvironment(cwd=str(working))
    try:
        first = 'printf "initial command\\n"'
        assert env.execute(first, yield_handler=handler(first, working))['returncode'] == 0
        shutil.rmtree(working)
        command = 'printf "recovered command\\n"'
        result = env.execute(command, yield_handler=handler(command, working))
        assert result['returncode'] == 0 and 'recovered command' in result['output']
        assert env.cwd == str(tmp_path)
        session = next(p for p in [*registry._running.values(), *registry._finished.values()] if p.command == command)
        assert session.cwd == str(tmp_path) and session.native_job_scope
        assert session._completion_event.wait(5) and job_quiescent(session)
    finally:
        env.cleanup()
