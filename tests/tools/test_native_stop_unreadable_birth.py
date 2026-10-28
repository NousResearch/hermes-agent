"""Unreadable current PID birth permits native unit cleanup, never a host PID signal."""
import asyncio
import json
import shlex
import sys

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.asyncio
@pytest.mark.parametrize('native_restart', [False, True], ids=['direct', 'native-restart'])
@pytest.mark.parametrize('refuse_unit', [False, True], ids=['unit-stopped', 'unit-refused-then-retry'])
async def test_unreadable_birth_keeps_the_real_neighbor_safe(tmp_path, monkeypatch, native_restart, refuse_unit):
    from gateway.session_ancillary import process_stop_pending, retry_process_stop
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch
    from tools import process_registry as module
    from tools.process_registry_output_log import output_dir
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    code = 'exec ' + shlex.join([sys.executable, '-c', 'import threading; print("unit output",flush=True); threading.Event().wait(20)'])
    tokens = set_session_vars(session_id='foreign-native-parent', session_key='route')
    try:
        neighbor = original.spawn_local(code, cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    scope_argv = original._scope_argv
    def launch_unit(session, command, suffix, label):
        session.systemd_unit = 'hermes-worker-' + suffix + '.scope'
        return [module._find_shell(), '-lic', 'set +m; ' + command]
    monkeypatch.setattr(original, '_scope_argv', launch_unit)
    output_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, 'worker_output_log', lambda sid: output_dir() / (sid + '.log'))
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = original.spawn_local(code, cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    monkeypatch.setattr(original, '_scope_argv', scope_argv)
    unit, birth = own.systemd_unit, own.host_start_time
    probe = module.ProcessRegistry._safe_host_start_time
    monkeypatch.setattr(module.ProcessRegistry, '_safe_host_start_time', staticmethod(
        lambda pid: None if pid == own.pid else probe(pid)))
    attempts = []
    def stop_unit(name):
        assert name == unit, 'foreign native unit selected'
        attempts.append(name)
        if refuse_unit:
            return False
        own.process.terminate()  # This fixture's actual unit boundary holds its original child handle.
        own.process.wait(5)
        assert own._completion_event.wait(5)
        return True
    monkeypatch.setattr(module, '_stop_systemd_unit', stop_unit)
    current = original
    if native_restart:
        epoch = begin_runtime_epoch(db, instance_id='unreadable-birth-owner')
        old = authority
        authority = SessionAuthority(old.runner, profile_id='owned', instance_id='unreadable-birth-owner', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
        current = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', current)
    monkeypatch.setattr(current, '_terminate_host_pid', lambda *a, **k: pytest.fail('unproved host PID signaled'))
    try:
        assert type(birth) is int and birth > 0
        await asyncio.to_thread(wait_until, lambda: 'unit output' in own.output_buffer, timeout=5)
        if native_restart:
            if refuse_unit:
                with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
                    await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
            else:
                await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        else:
            result = current.kill_process(own.id)
            assert (result['status'] == 'error') is refuse_unit, result
        if refuse_unit:
            assert own.process.poll() is None and not neighbor.exited
            row = next(row for row in json.loads(module._checkpoint_path().read_text()) if row['session_id'] == own.id)
            assert row['host_start_time'] == birth and row['_scope_stop_pending']
            refuse_unit = False
            if native_restart:
                assert await retry_process_stop(authority, REF)
            else:
                assert current.kill_process(own.id)['status'] in {'killed', 'already_exited'}
        assert attempts and own.process.poll() is not None and not neighbor.exited
        assert own.output_buffer.count('unit output') == 1
        if native_restart:
            assert process_stop_pending(db, 's') is None
    finally:
        refuse_unit = False
        await asyncio.to_thread(original.kill_process, own.id)
        await asyncio.to_thread(original.kill_process, neighbor.id)
        db.close()


@pytest.mark.asyncio
async def test_partial_unit_launch_with_no_birth_retains_evidence_without_pid_fallback(tmp_path, monkeypatch):
    from gateway.status import get_process_start_time
    from tests.waiting import wait_until
    from tools import process_registry as module
    from tools.process_registry_output_log import output_dir
    registry = module.ProcessRegistry()
    launch, processes, signals = module.subprocess.Popen, [], []
    entered = tmp_path / 'command-started'
    output_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, 'worker_output_log', lambda sid: output_dir() / (sid + '.log'))
    def unit_launch(session, command, suffix, label):
        session.systemd_unit = 'hermes-worker-' + suffix + '.scope'
        return [module._find_shell(), '-lic', 'set +m; ' + command]
    monkeypatch.setattr(registry, '_scope_argv', unit_launch)
    def captured(*args, **kwargs):
        process = launch(*args, **kwargs)
        processes.append((process, get_process_start_time(process.pid)))
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', captured)
    monkeypatch.setattr(registry, '_safe_host_start_time', lambda _pid: None)
    def tracking_failed(*args):
        wait_until(entered.exists, timeout=5)
        raise RuntimeError('tracking failed after actual unit launch')
    monkeypatch.setattr(registry, '_track_started', tracking_failed)
    monkeypatch.setattr(registry, '_terminate_host_pid', lambda *a, **k: signals.append((a, k)))
    monkeypatch.setattr(module, '_stop_systemd_unit', lambda _unit: False)
    command = 'exec ' + shlex.join([sys.executable, '-c',
        f'from pathlib import Path; import threading; print("partial unit output",flush=True); Path({str(entered)!r}).touch(); threading.Event().wait(20)'])
    try:
        with pytest.raises(RuntimeError, match='tracking failed after actual unit launch'):
            registry.spawn_local(command, cwd=str(tmp_path))
        assert signals == []
        own = next(iter(registry._running.values()))
        assert own.host_start_time is None and own._scope_stop_pending
        row = next(row for row in json.loads(module._checkpoint_path().read_text()) if row['session_id'] == own.id)
        assert row['host_start_time'] is None and row['systemd_unit']
    finally:
        process, birth = processes[0]
        assert get_process_start_time(process.pid) == birth
        process.terminate()  # Fixture owns this original fresh child; no production identity is backfilled.
        process.wait(5)
    def stopped_unit(_unit):
        return process.poll() is not None
    monkeypatch.setattr(module, '_stop_systemd_unit', stopped_unit)
    registry._reap_untracked(own, process)
    assert own._completion_event.is_set()
    assert own.output_buffer.count('partial unit output') == 1
    assert signals == []
