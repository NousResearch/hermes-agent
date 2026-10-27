"""Physical job completion cannot discharge a Stop whose original result is not durable."""
import asyncio
import json
from pathlib import Path
import shlex
import sys
import threading

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.asyncio
@pytest.mark.parametrize(('reload_owner', 'native_control', 'hold_recovery'), [
    (False, False, False), (False, True, False), (True, False, False), (True, True, False), (True, True, True)],
    ids=['live-direct', 'live-native-owner', 'reloaded-direct', 'reloaded-native-owner', 'reloaded-held-reader'])
async def test_result_write_refusal_preserves_stop_output_and_retry(
        tmp_path, monkeypatch, native_control, reload_owner, hold_recovery):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    from tools.process_registry_results import load_completed_results
    from tools import process_registry_results as results

    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    tokens = set_session_vars(session_id='foreign-parent', session_key='route')
    try:
        neighbor = original.spawn_local('exec ' + shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(30)']), cwd=str(tmp_path),
            task_id='same-raw-owner', owner_task_id='same-raw-owner', session_key='route')
    finally:
        clear_session_vars(tokens)
    write, refusals, executed = results.atomic_json_write, [], []
    refusing = True
    reader_waiting, release_reader = threading.Event(), threading.Event()
    reader = module.ProcessRegistry._log_reader_loop
    def delayed_recovered_reader(registry, process):
        if hold_recovery and not refusing and process.id == own.id:
            reader_waiting.set()
            assert release_reader.wait(10), 'original recovered reader was never released'
        return reader(registry, process)
    monkeypatch.setattr(module.ProcessRegistry, '_log_reader_loop', delayed_recovered_reader)
    def refuse_result(path, *args, **kwargs):
        if refusing and Path(path).parent.name == 'process-results':
            refusals.append(str(path))
            raise PermissionError('finite fixture result-store refusal')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(results, 'atomic_json_write', refuse_result)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = original.spawn_local(shlex.join([sys.executable, '-c',
            'print("original durable output", flush=True); raise SystemExit(7)']), cwd=str(tmp_path),
            task_id='same-raw-owner', owner_task_id='same-raw-owner', session_key='route')
    finally:
        clear_session_vars(tokens)
    current = original
    async def execute(owner, ref, row):
        assert job_quiescent(own) and not neighbor.exited
        executed.append(row['request_id'])
        return 'human continued'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert refusals and own._result_persisted_generation == -1 and job_quiescent(own)
        assert own.output_buffer.count('original durable output') == 1
        assert Path(own.output_log).exists()
        if native_control:
            with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
                await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
            assert process_stop_pending(db, 's') is not None
            await _submit(authority, 'pending-human')
            await asyncio.wait_for(authority.sessions['s'].task, 10)
            assert executed == []
        else:
            result = current.kill_process(own.id)
            assert result['status'] == 'error' and result['error'] == 'process_stop_unknown'
        assert own._scope_stop_pending and not neighbor.exited
        assert own.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        if reload_owner:
            current = module.ProcessRegistry()
            monkeypatch.setattr(module, 'process_registry', current)
            current.recover_from_checkpoint()
            restored = current.get(own.id)
            assert await asyncio.to_thread(restored._completion_event.wait, 10)
            assert restored._result_persisted_generation == -1
            if native_control:
                epoch = begin_runtime_epoch(db, instance_id='result-retry-owner')
                old = authority
                authority = SessionAuthority(old.runner, profile_id='owned', instance_id='result-retry-owner', db=db, epoch=epoch)
                authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
        refusing = False
        if native_control:
            await _submit(authority, 'ordinary-follow-up')
            if hold_recovery:
                assert await asyncio.to_thread(reader_waiting.wait, 10)
                parked = authority.sessions['s'].task
                await asyncio.wait_for(parked, 10)
                assert executed == [] and process_stop_pending(db, 's') is not None
                with current._lock:
                    publishing = current._running.get(own.id) or current._finished.get(own.id)
                assert not publishing._completion_event.is_set() and publishing._result_persisted_generation == -1
                assert job_quiescent(publishing) and neighbor.process.poll() is None
                from hermes_state_runtime import list_session_admissions
                assert [(r['request_id'], r['status']) for r in list_session_admissions(db, session_id='s')
                        if r['principal_id'] != 'native-process-stop'] == [
                    ('pending-human', 'queued'), ('ordinary-follow-up', 'queued')]
                release_reader.set()
            from tests.waiting import async_wait_until
            # A recovered reader can finish after this drain parks; its native
            # completion observation schedules another drain of this same FIFO.
            await async_wait_until(lambda: executed == ['pending-human', 'ordinary-follow-up']
                                   and authority.sessions['s'].task.done(), timeout=10)
            assert executed == ['pending-human', 'ordinary-follow-up']
            from hermes_state_runtime import list_session_admissions
            assert [(r['request_id'], r['status'], r['outcome'])
                    for r in list_session_admissions(db, session_id='s', pending_only=False)
                    if r['principal_id'] != 'native-process-stop'] == [
                ('pending-human', 'terminal', 'completed'), ('ordinary-follow-up', 'terminal', 'completed')]
            assert process_stop_pending(db, 's') is None
        else:
            result = current.kill_process(own.id)
            assert result['status'] in {'already_exited', 'killed'}
        retained = load_completed_results(owns_entry=lambda row: row['id'] == own.id, strict=True)[own.id]
        assert retained.exit_code == 7 and retained.output_buffer == own.output_buffer == 'original durable output\n'
        assert retained._result_persisted_generation == retained.ownership_generation
        assert own.id not in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        assert not neighbor.exited and neighbor.process.poll() is None
    finally:
        release_reader.set()
        refusing = False
        await asyncio.to_thread(original.kill_process, neighbor.id)
        db.close()


@pytest.mark.asyncio
async def test_a_saved_partial_snapshot_cannot_stand_in_for_the_failed_final_result(tmp_path, monkeypatch):
    import threading
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError
    from tools import process_registry as module
    from tools import process_registry_results as results
    from tests.waiting import async_wait_until
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    entered, release = threading.Event(), threading.Event()
    ingest = registry._ingest_output
    def pause(session, text, **kwargs):
        if 'final original output' in text:
            entered.set()
            assert release.wait(20)
        ingest(session, text, **kwargs)
    monkeypatch.setattr(registry, '_ingest_output', pause)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = registry.spawn_local('exec ' + shlex.join([sys.executable, '-c',
            'print("final original output", flush=True)']), cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    executed = []
    async def execute(owner, ref, row):
        assert own.output_buffer.count('final original output') == 1
        executed.append(row['request_id'])
        return 'continued'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    write = results.atomic_json_write
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.to_thread(own.process.wait, 5)
        registry.poll(own.id)  # Actual exit reconciliation while its original reader is held.
        assert own.exited and not own._completion_event.is_set()
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, 0)
        await _submit(authority, 'pending-human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert own._result_persisted_generation == -1  # A partial write never certifies the original final result.
        assert json.loads((tmp_path / 'logs' / 'process-results' / (own.id + '.json')).read_text())['result_complete'] is False
        def refuse_final(*args, **kwargs):
            raise PermissionError('finite final original-result refusal')
        monkeypatch.setattr(results, 'atomic_json_write', refuse_final)
        release.set()
        assert await asyncio.to_thread(own._completion_event.wait, 5)
        assert own._result_persisted_generation == -1
        assert executed == [] and process_stop_pending(db, 's')
        monkeypatch.setattr(results, 'atomic_json_write', write)
        assert results.save_completed_result(own)
        await async_wait_until(lambda: executed == ['pending-human'] and authority.sessions['s'].task.done(), timeout=5)
        assert process_stop_pending(db, 's') is None
    finally:
        release.set()
        monkeypatch.setattr(results, 'atomic_json_write', write)
        await asyncio.to_thread(own._completion_event.wait, 5)
        db.close()
