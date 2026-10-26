"""Native job recovery retains output and never retries a partially launched command."""

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
from tools import process_registry_results as results
from tools.process_registry_job import job_quiescent, job_receipt_path, read_job_receipt


pytestmark = pytest.mark.platforms('linux')


@pytest.mark.parametrize('refuse_result', [False, True], ids=['durable', 'write-refused'])
def test_cold_completed_job_keeps_checkpoint_until_its_output_is_durable(tmp_path, monkeypatch, refuse_result):
    monkeypatch.setattr(module, '_systemd_run_user_scope_available', lambda: False)
    code = '''import json, os, shlex, sys, time
from tools import process_registry as m
from tools.process_registry_job import read_job_receipt
m._systemd_run_user_scope_available = lambda: False
r = m.ProcessRegistry()
s = r.spawn_local('exec ' + shlex.join([sys.executable, '-c',
    'import time; time.sleep(.3); print("offline-output", flush=True); raise SystemExit(23)']),
    task_id='task', session_key='route', persist_on_release=True)
while read_job_receipt(s) is None: time.sleep(.01)
print(json.dumps({'session_id': s.id, 'pid': s.pid, 'host_start_time': s.host_start_time,
                  'native_job_scope': s.native_job_scope}), flush=True)
os._exit(0)
'''
    worker = subprocess.run([sys.executable, '-c', code], cwd=Path(module.__file__).parents[1],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=10, check=False)
    assert worker.returncode == 0, worker.stderr
    entry = json.loads(worker.stdout.strip().splitlines()[-1])
    wait_until(lambda: job_quiescent(entry), timeout=10)
    # Ordinary output retention cannot delete the only native scope verdict
    # before cold recovery publishes the completed output/result.
    from agent import runtime_session_store
    from tools.process_registry_output_log import worker_output_log
    from tools.process_registry_results import RESULT_RETENTION_SECONDS
    import time
    receipt_path = job_receipt_path(entry['session_id'])
    old = time.time() - RESULT_RETENTION_SECONDS - 1
    os.utime(receipt_path, (old, old))
    monkeypatch.setattr(runtime_session_store, 'is_worker_process', lambda: True)
    worker_output_log('proc_next')
    assert job_quiescent(entry)
    registry = module.ProcessRegistry()
    entered, release = threading.Event(), threading.Event()
    finish = registry._move_to_finished

    def held_finish(session):
        entered.set()
        assert release.wait(10)
        return finish(session)

    monkeypatch.setattr(registry, '_move_to_finished', held_finish)
    original_write = results.atomic_json_write

    def write(path, *args, **kwargs):
        if refuse_result:
            raise PermissionError('refuse completed-result commit')
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(results, 'atomic_json_write', write)
    try:
        assert registry.recover_from_checkpoint() == 1
        assert entered.wait(5)
        session = registry.get(entry['session_id'])
        assert session.exited and 'offline-output' in session.output_buffer
        assert registry._write_checkpoint()
        assert entry['session_id'] in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
    finally:
        release.set()
    assert session._completion_event.wait(5)
    if refuse_result:
        assert entry['session_id'] in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        monkeypatch.setattr(results, 'atomic_json_write', original_write)
        assert results.save_completed_result(session)
        assert registry._write_checkpoint()
    restored = results.load_completed_results(owns_entry=lambda row: True, strict=True)[session.id]
    assert restored.exit_code == 23
    assert 'offline-output' in restored.output_buffer
    assert not json.loads(module._checkpoint_path().read_text())


def test_pty_registration_failure_keeps_original_job_without_command_replay(tmp_path, monkeypatch):
    pytest.importorskip('ptyprocess')
    monkeypatch.setattr(module, '_systemd_run_user_scope_available', lambda: False)
    registry = module.ProcessRegistry()
    track = registry._track_started
    captured = []
    effect = tmp_path / 'starts'
    payload = ('import os, time; from pathlib import Path; '
               'print("original-output", flush=True); '
               f'p=Path({str(effect)!r}); p.open("a").write(str(os.getpid())+"\\n"); '
               'time.sleep(.5)')

    def fail_first(session, *args, **kwargs):
        captured.append(session)
        if len(captured) == 1:
            wait_until(effect.exists, timeout=5)
            raise RuntimeError('reader registration failed after native command start')
        return track(session, *args, **kwargs)

    monkeypatch.setattr(registry, '_track_started', fail_first)
    with pytest.raises(RuntimeError, match='original command was not retried'):
        registry.spawn_local('exec ' + shlex.join([sys.executable, '-c', payload]),
                             use_pty=True, task_id='task', session_key='route')
    session = captured[0]
    assert all(item is session for item in captured)
    assert session._completion_event.wait(10)
    assert len(effect.read_text().splitlines()) == 1
    assert read_job_receipt(session)['pid'] == session.pid
    assert job_quiescent(session)
    assert 'original-output' in session.output_buffer
    assert registry.get(session.id) is session
