"""Delayed partial result snapshots cannot replace the original reader result."""
import json
import shlex
import sys
import threading
import pytest
from tests.waiting import wait_until

@pytest.mark.platforms('linux')
@pytest.mark.parametrize('cached_writer', [False, True], ids=['original-object', 'cached-reloaded-partial'])
def test_late_partial_stop_writer_cannot_overwrite_original_final_result(tmp_path, monkeypatch, cached_writer):
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
    observed={}
    writer=None
    try:
        assert entered.wait(5)
        own.process.wait(5)
        registry.poll(own.id)
        assert own.exited and not own._completion_event.is_set()
        if cached_writer:
            assert results.save_completed_result(own)
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
        assert 'final owned output' in json.loads(retained.read_text())['output']
        release_writer.set()
        writer.join(5)
        assert not writer.is_alive()
        print('LIVE_OUTPUT',own.output_buffer,'RETAINED',json.loads(retained.read_text())['output'],'STOP',observed)
        saved = json.loads(retained.read_text())
        assert 'final owned output' in saved['output'] and saved['exit_code'] == 7
        assert saved['result_complete'] is True
        if cached_writer:
            # A still-partial cached object never becomes a final-output authority.
            assert not cached._result_complete
            cached.completion_consumed = True
            assert results.save_completed_result(cached)
            saved = json.loads(retained.read_text())
            assert saved['completion_consumed'] is True and saved['result_complete'] is True
            assert saved['output'].count('final owned output') == 1 and saved['exit_code'] == 7
    finally:
        release_reader.set();release_writer.set()
        if writer: writer.join(5)
        own._completion_event.wait(5)
        registry.kill_process(own.id)
