"""Original stream errors preserve native scope/output uncertainty after physical exit."""
import json
import shlex
import sys
import subprocess
import threading
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.parametrize('failure', ['pipe-read', 'pty-read', 'decoder-final', 'append'])
def test_original_stream_failure_never_becomes_a_complete_empty_result(tmp_path, monkeypatch, failure):
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    monkeypatch.setattr(module, 'worker_output_log', lambda session_id: None)
    monkeypatch.setattr(module, '_systemd_run_user_scope_available', lambda: False)
    registry = module.ProcessRegistry()
    entered, release = threading.Event(), threading.Event()
    name = '_pty_reader_loop' if failure == 'pty-read' else '_reader_loop'
    original_reader = getattr(registry, name)
    def held_reader(session):
        entered.set()
        assert release.wait(10)
        original_reader(session)
    monkeypatch.setattr(registry, name, held_reader)
    payload = 'print("original stream output",flush=True); raise SystemExit(7)'
    command = shlex.join([sys.executable, '-c', payload])
    if failure == 'pty-read':
        own = registry.spawn_local(command, cwd=str(tmp_path), use_pty=True)
    else:
        # The generic native launcher also supports real pipe readers; normal
        # background spawn chooses durable logs, so bind this pipe explicitly.
        from tools.process_registry_job import job_argv
        own = registry._new_session(command, 'reader', 'reader', 'route', str(tmp_path))
        own.process = subprocess.Popen(job_argv(own, [sys.executable, '-c', payload], 2),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)
        own.pid = own.process.pid
        own.host_start_time = registry._safe_host_start_time(own.pid)
        registry._track_started(own, held_reader, 'original-pipe-reader')
    try:
        assert entered.wait(5)
        wait_until(lambda: job_quiescent(own), timeout=5)
        if failure == 'pipe-read':
            stream = own.process.stdout
            def unreadable(*args, **kwargs):
                raise OSError('finite original pipe read failure')
            own.process.stdout = SimpleNamespace(buffer=SimpleNamespace(read1=unreadable),
                fileno=stream.fileno, close=stream.close, read=unreadable)
        elif failure == 'pty-read':
            def unreadable(*args, **kwargs):
                raise OSError('finite original PTY read failure')
            monkeypatch.setattr(own._pty, 'read', unreadable)
        elif failure == 'decoder-final':
            factory = module.codecs.getincrementaldecoder
            def decoder(*args, **kwargs):
                delegate = factory(*args, **kwargs)
                def build(*a, **kw):
                    actual = delegate(*a, **kw)
                    def decode(data, final=False):
                        if final:
                            raise UnicodeError('finite original decoder flush failure')
                        return actual.decode(data, final=final)
                    return SimpleNamespace(decode=decode)
                return build
            monkeypatch.setattr(module.codecs, 'getincrementaldecoder', decoder)
        else:
            def unreadable(*args, **kwargs):
                raise OSError('finite original output ingestion failure')
            monkeypatch.setattr(registry, '_ingest_output', unreadable)
        release.set()
        assert own._completion_event.wait(5)
        assert own.exit_code == 7 and job_quiescent(own)
        assert not own._result_complete and own._result_persisted_generation == -1
        assert own._scope_stop_pending
        result = registry.kill_process(own.id)
        assert (result['status'], result['error']) == ('error', 'process_stop_unknown')
        assert any(row['session_id'] == own.id for row in json.loads(module._checkpoint_path().read_text()))
    finally:
        release.set()
        own._completion_event.wait(5)
