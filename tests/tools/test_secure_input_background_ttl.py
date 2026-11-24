"""Real background pipe output must never outlive secure-input protection."""
import json
import os
import shlex
import shutil
import signal
import sys
from pathlib import Path

import pytest

from agent import secure_input_broker as broker


@pytest.mark.platforms('posix')
def test_background_pipe_ui_poll_and_receipt_stay_safe_after_expiry(tmp_path, monkeypatch):
    from tests.waiting import wait_until
    from tools import process_registry as processes
    from tools import terminal_tool as terminal

    broker.clear_all()
    clock = [1000.0]
    monkeypatch.setattr(broker, '_now', lambda: clock[0])
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, 'process_registry', registry)
    # Bypass user shell rc files, not the real spawn/env/capture/notification paths.
    monkeypatch.setattr(registry, '_scope_argv',
                        lambda session, command, *args: [shutil.which('bash'), '-c', command])
    streamed = []
    registry.on_output = lambda session, chunk: streamed.append(chunk)
    ready = tmp_path / 'ready'
    value = 'synthetic-background-ttl-sensitive-suffix'
    ref = broker.register_secret(value, purpose='fixture', ttl_seconds=10)['secret_ref']
    code = ("import os,signal,threading; from pathlib import Path; gate=threading.Event(); "
            "signal.signal(signal.SIGUSR1, lambda *args: gate.set()); value=os.environ['GH_TOKEN']; "
            "print(value+'x'*80,flush=True); Path(" + repr(str(ready)) + ").write_text('ready'); "
            "gate.wait(10); print(value,flush=True)")
    command = 'exec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
    try:
        started = json.loads(terminal.terminal_tool(command, background=True, workdir=str(tmp_path),
                                                    env_secret_refs={'GH_TOKEN': ref}))
        assert started.get('exit_code') == 0, started
        session = registry.get(started['session_id'])
        wait_until(lambda: ready.exists() and session.total_output_chars > 0, timeout=5)
        clock[0] += 11
        assert broker.describe_secret(ref) is None
        os.kill(session.pid, signal.SIGUSR1)
        result = registry.wait(session.id, timeout=10)
        assert value not in session.output_buffer
        assert value not in ''.join(streamed)
        assert value not in result['output']
        assert 'TTL expired' in result['output']
        for receipt in Path(os.environ['HERMES_HOME']).glob('logs/process-results/*'):
            assert value not in receipt.read_text()
    finally:
        registry.kill_all()
        broker.clear_all()


def test_file_backed_worker_rejects_secret_background_before_consumption(monkeypatch):
    from agent import runtime_session_store
    from tools.terminal_tool_secure_input import validate_secret_refs
    broker.clear_all()
    ref = broker.register_secret('synthetic-worker-sensitive-suffix', purpose='fixture')['secret_ref']
    monkeypatch.setattr(runtime_session_store, 'is_worker_process', lambda: True)
    try:
        with pytest.raises(ValueError, match='file-backed'):
            validate_secret_refs(None, {'GH_TOKEN': ref}, env_type='local', background=True)
        assert broker.describe_secret(ref)['consumed'] is False
    finally:
        broker.clear_all()
