"""Native terminal execution retains escaped children and holds Stop until its reader settles."""
import asyncio
import ctypes
import json
from pathlib import Path
import shlex
import sys
import threading

import psutil
import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until, wait_until

pytestmark = pytest.mark.platforms('linux')


def alive(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture
def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    yield
    assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('held_reader', [False, True], ids=['collected', 'original-reader-held'])
async def test_terminal_frontdoor_native_stop_holds_fifo_and_reaps_detached_child(
        tmp_path, monkeypatch, subreaper, held_reader):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    from tools import process_registry as module
    from tools.environments import base
    from tools.process_registry_job import job_quiescent
    from tools.terminal_tool import terminal_tool

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    stopped, entered, release_reader = threading.Event(), threading.Event(), threading.Event()
    release_turn = asyncio.Event()
    monkeypatch.setattr(base, 'is_interrupted', stopped.is_set)
    ready, release, effect, starts = [tmp_path / name for name in ('ready', 'release', 'effect', 'starts')]
    payload = f'''import os,time,threading
from pathlib import Path
Path({str(starts)!r}).open('a').write('one original command\\n')
pid=os.fork()
if pid==0:
 os.setsid()
 if os.fork(): os._exit(0)
 fd=os.open(os.devnull,os.O_RDWR)
 for n in (0,1,2): os.dup2(fd,n)
 Path({str(ready)!r}).write_text(str(os.getpid()))
 end=time.monotonic()+20
 while not Path({str(release)!r}).exists() and time.monotonic()<end: time.sleep(.01)
 if Path({str(release)!r}).exists(): Path({str(effect)!r}).touch()
 os._exit(0)
os.waitpid(pid,0)
print('original foreground output',flush=True)
threading.Event().wait(20)
'''
    command = 'exec ' + shlex.join([sys.executable, '-c', payload])
    move = registry._move_to_finished
    def hold_publication(session):
        if held_reader and session.command == command:
            entered.set()
            assert release_reader.wait(10)
        return move(session)
    monkeypatch.setattr(registry, '_move_to_finished', hold_publication)
    class Agent:
        session_id = 's'
        def hard_interrupt(self):
            stopped.set()
    agent, executed, tool_results, successor_alive = Agent(), [], [], []
    tokens = set_session_vars(session_id='foreign-parent', session_key='route')
    try:
        foreign = registry.spawn_local('exec ' + shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(30)']), cwd=str(tmp_path), task_id='s',
            owner_task_id='s', session_key='route')
    finally:
        clear_session_vars(tokens)

    async def execute(owner, ref, row):
        if row['request_id'] == 'original':
            owner.adopt_agent('s', row['generation'], agent)
            tokens = set_session_vars(session_id='s', session_key='route')
            try:
                tool_results.append(json.loads(await asyncio.to_thread(terminal_tool, command,
                    task_id='s', session_id='s', timeout=15, workdir=str(tmp_path), force=True)))
            finally:
                clear_session_vars(tokens)
            await release_turn.wait()
            return 'known original turn completed'
        successor_alive.append(alive(int(ready.read_text())))
        if not successor_alive[-1]:
            owned = next(p for p in registry._finished.values() if p.command == command)
            assert job_quiescent(owned) and owned._completion_event.is_set()
            assert owned._result_complete and owned._result_persisted_generation == owned.ownership_generation
            assert owned.output_buffer.count('original foreground output') == 1 and type(owned.exit_code) is int
        assert foreign.process.poll() is None
        executed.append(row['request_id'])
        release.touch()
        return 'one human successor'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    child = None
    try:
        await _submit(authority, 'original')
        await async_wait_until(ready.exists, timeout=10)
        child = int(ready.read_text())
        assert alive(child)
        with registry._lock:
            owned = next((p for p in registry._running.values() if p.command == command), None)
        if held_reader:
            assert owned is not None and owned.native_job_scope and owned.parent_session_id == 's'
        await _submit(authority, 'human-successor')
        generation = authority._handle(REF).execution_generation
        if held_reader:
            with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
                await authority.interrupt(ACTOR, REF, generation)
            assert entered.is_set() and process_stop_pending(db, 's') is not None
            assert not owned._completion_event.is_set() and owned._scope_stop_pending
            assert executed == [] and foreign.process.poll() is None
        else:
            await authority.interrupt(ACTOR, REF, generation)
        release_reader.set()
        release_turn.set()
        await async_wait_until(lambda: executed == ['human-successor'] and authority.sessions['s'].task.done(), timeout=10)
        if successor_alive != [False]:
            await async_wait_until(effect.exists, timeout=3)
        assert successor_alive == [False], dict(alive=successor_alive, effect=effect.exists(),
                                               stop_fence=process_stop_pending(db, 's'))
        assert owned is not None and owned.native_job_scope and owned.parent_session_id == 's'
        assert process_stop_pending(db, 's') is None and not alive(child) and not effect.exists()
        assert starts.read_text().splitlines() == ['one original command']
        assert len(tool_results) == 1
        rows = [(r['request_id'], r['status'], r['outcome']) for r in list_session_admissions(
            db, session_id='s', pending_only=False) if r['principal_id'] != 'native-process-stop']
        assert rows == [('original', 'terminal', 'completed'), ('human-successor', 'terminal', 'completed')]
    finally:
        stopped.set()
        release_reader.set()
        release_turn.set()
        release.touch()
        if child is not None:
            await asyncio.to_thread(wait_until, lambda: not alive(child), timeout=20)
            try:
                import os
                await asyncio.to_thread(os.waitpid, child, 0)
            except ChildProcessError:
                pass
        await asyncio.to_thread(registry.kill_process, foreign.id)
        if authority.sessions['s'].task is not None:
            await asyncio.wait_for(authority.sessions['s'].task, 20)
        db.close()
