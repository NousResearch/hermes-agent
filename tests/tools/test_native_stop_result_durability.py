"""Physical job completion cannot discharge a Stop whose original result is not durable."""
import asyncio
import json
from pathlib import Path
import shlex
import sys

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.asyncio
@pytest.mark.parametrize('native_control', [False, True], ids=['direct', 'native-owner'])
@pytest.mark.parametrize('reload_owner', [False, True], ids=['live', 'reloaded'])
async def test_result_write_refusal_preserves_stop_output_and_retry(tmp_path, monkeypatch, native_control, reload_owner):
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
            await asyncio.wait_for(authority.sessions['s'].task, 10)
            assert executed == ['pending-human', 'ordinary-follow-up']
            assert process_stop_pending(db, 's') is None
        else:
            result = current.kill_process(own.id)
            assert result['status'] in {'already_exited', 'killed'}
        retained = load_completed_results(owns_entry=lambda row: row['id'] == own.id, strict=True)[own.id]
        assert retained.exit_code == 7 and retained.output_buffer.count('original durable output') == 1
        assert retained._result_persisted_generation == retained.ownership_generation
        assert own.id not in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        assert not neighbor.exited
    finally:
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
