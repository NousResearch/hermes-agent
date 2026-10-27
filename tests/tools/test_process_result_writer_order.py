"""Delayed partial result snapshots cannot replace the original reader result."""
import json
import shlex
import sys
import threading
import pytest
from tests.waiting import wait_until

@pytest.mark.platforms('linux')
@pytest.mark.parametrize('refuse_final', [False, True], ids=['final-written', 'final-refused'])
@pytest.mark.parametrize('cached_writer', [False, True], ids=['original-object', 'cached-reloaded-partial'])
def test_late_partial_stop_writer_cannot_overwrite_original_final_result(tmp_path, monkeypatch, cached_writer, refuse_final):
    from tools import process_registry as module
    from tools import process_registry_results as results
    from agent import redact
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    registry=module.ProcessRegistry()
    entered, release_reader, partial_captured, release_writer = [threading.Event() for _ in range(4)]
    ingest=registry._ingest_output
    def pause_reader(session,text,**kwargs):
        if 'final owned output' in text:
            entered.set()
            assert release_reader.wait(20)
        ingest(session,text,**kwargs)
    monkeypatch.setattr(registry,'_ingest_output',pause_reader)
    redact_output=redact.redact_terminal_output
    def pause_partial(text,*args,**kwargs):
        if threading.current_thread().name=='stale-stop-writer':
            partial_captured.set()
            assert release_writer.wait(20)
        return redact_output(text,*args,**kwargs)
    monkeypatch.setattr(redact,'redact_terminal_output',pause_partial)
    own=registry.spawn_local(shlex.join([sys.executable,'-c','print("final owned output",flush=True); raise SystemExit(7)']),cwd=str(tmp_path))
    write = results.atomic_json_write
    def refuse_reader(path, *args, **kwargs):
        if refuse_final and threading.current_thread() is own._reader_thread:
            raise PermissionError('finite original final-result refusal')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(results, 'atomic_json_write', refuse_reader)
    observed={}
    writer=None
    try:
        assert entered.wait(5)
        own.process.wait(5)
        registry.poll(own.id)
        assert own.exited and not own._completion_event.is_set()
        if cached_writer:
            assert not results.save_completed_result(own)
            cached = results.load_completed_results(owns_entry=lambda row: row['id'] == own.id)[own.id]
            assert cached.output_buffer == ''
            assert not cached._result_complete
        else:
            cached = own
        def stop():
            if cached_writer:
                observed['saved'] = results.save_completed_result(cached)
            else:
                observed.update(registry.kill_process(own.id))
        writer=threading.Thread(target=stop,name='stale-stop-writer')
        writer.start()
        assert partial_captured.wait(5)
        release_reader.set()
        assert own._completion_event.wait(5)
        retained=tmp_path/'logs'/'process-results'/(own.id+'.json')
        if not refuse_final:
            assert 'final owned output' in json.loads(retained.read_text())['output']
        release_writer.set()
        writer.join(5)
        assert not writer.is_alive()
        print('LIVE_OUTPUT',own.output_buffer,'RETAINED',json.loads(retained.read_text())['output'],'STOP',observed)
        saved = json.loads(retained.read_text())
        if cached_writer and refuse_final:
            assert observed['saved'] is False and saved['result_complete'] is False and saved['output'] == ''
            assert cached._result_persisted_generation == -1 and cached._scope_stop_pending
            assert own._result_persisted_generation == -1 and own._scope_stop_pending
            registry._write_checkpoint()
            assert any(row['session_id'] == own.id for row in json.loads(module._checkpoint_path().read_text()))
            restored = results.load_completed_results(owns_entry=lambda row: row['id'] == own.id)[own.id]
            assert restored._result_persisted_generation == -1 and restored._scope_stop_pending
            assert not restored._result_complete
            monkeypatch.setattr(results, 'atomic_json_write', write)
            assert results.save_completed_result(own)
            saved = json.loads(retained.read_text())
        assert 'final owned output' in saved['output'] and saved['exit_code'] == 7
        assert saved['result_complete'] is True
        if cached_writer:
            # Only copying an actual complete retained payload may promote this cache.
            assert cached._result_complete is (not refuse_final)
            if cached._result_complete:
                assert cached.output_buffer.count('final owned output') == 1
            cached.completion_consumed = True
            assert results.save_completed_result(cached)
            saved = json.loads(retained.read_text())
            assert saved['completion_consumed'] is True and saved['result_complete'] is True
            assert saved['output'].count('final owned output') == 1 and saved['exit_code'] == 7
    finally:
        release_reader.set();release_writer.set()
        if writer: writer.join(5)
        own._completion_event.wait(5)
        monkeypatch.setattr(results, 'atomic_json_write', write)
        registry.kill_process(own.id)
