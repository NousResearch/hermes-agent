"""Lifecycle receipts through native dispatch in owned runtime fixtures."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from tests.tools.test_skill_native_audit import sandbox as sandbox, call, records, create_args


pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def handler(monkeypatch):
    import model_tools
    entry = model_tools.registry._tools['skill_manage']
    def install(fn):
        monkeypatch.setattr(entry, 'handler', fn)
    return install


def test_actual_runner_scope(tmp_path):
    temp_root = Path(tempfile.gettempdir()).resolve()
    home = Path(os.environ['HERMES_HOME']).resolve()
    assert tmp_path.resolve().is_relative_to(temp_root)
    assert home == (tmp_path / 'hermes_test').resolve()
    assert home.is_relative_to(temp_root)
    assert os.environ.get('HERMES_TEST_ISOLATION')
    assert 'HERMES_STATE_DB_GUARD_BYPASS' not in os.environ
    print(json.dumps({'tmp_path':str(tmp_path), 'tempfile_root':tempfile.gettempdir()}))


def test_disabled_exception_dispatches_exactly_once(sandbox, handler):
    # No native_call_audit key: exercise actual default, not a mocked predicate.
    (sandbox / 'config.yaml').write_text('skills:\n  ledger: true\n  external_dirs: []\n')
    calls = []
    error = RuntimeError('handler-exception-canary')
    class UnrenderableError(RuntimeError):
        # Registry normally translates exceptions; this real handler error also
        # raises during translation, exposing the callback exception boundary.
        def __str__(self):
            raise error
    def raising(args, **kwargs):
        calls.append(args)
        raise UnrenderableError()
    handler(raising)
    response = call({'private-argument-canary':'private-value-canary'})
    assert 'error' in json.loads(response)  # native model layer translates escaped exceptions
    assert len(calls) == 1, 'native disabled dispatch must never retry a raising handler'
    assert not (sandbox / 'skill_native_audit.jsonl').exists()


def test_rollback_failed_preserves_rows_and_recovery_snapshot(sandbox, monkeypatch):
    from tools import skill_manager_batch as batch, skill_ledger
    original_remove = batch.shutil.rmtree
    original_temp = batch.tempfile.mkdtemp
    snapshots = []
    def fail_remove(path, *args, **kwargs):
        if Path(path) == sandbox / 'skills' / 'audit-fixture':
            raise OSError('rollback-failure-canary')
        return original_remove(path, *args, **kwargs)
    def capture_snapshot(*args, **kwargs):
        path = original_temp(*args, **kwargs)
        snapshots.append(Path(path))
        return path
    monkeypatch.setattr(batch.shutil, 'rmtree', fail_remove)
    monkeypatch.setattr(batch.tempfile, 'mkdtemp', capture_snapshot)
    ops = create_args()['operations']
    ops.append({'action':'patch', 'name':'audit-fixture', 'old_string':'absent-canary', 'new_string':'unused'})
    response = call({'operations':ops})
    assert json.loads(response)['success'] is False
    entry, done = records(sandbox)
    assert done['handler_status'] == 'failure'
    assert done['batch_rollback'] == 'rollback_failed'
    rows = skill_ledger.list_entries()
    assert rows[0]['native_skill_call'] == entry['invocation_id']
    assert done['ledger_entries'][0]['id'] == rows[0]['id']
    assert (sandbox / 'skills' / 'audit-fixture').exists()
    assert len(snapshots) == 1 and snapshots[0].is_dir()
    assert snapshots[0].resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert 'rollback-failure-canary' not in (sandbox / 'skill_native_audit.jsonl').read_text()


@pytest.mark.parametrize('seam', ['_enabled', '_key', '_append', '_digest'])
def test_unavailable_preparation_dispatches_raising_callback_once(sandbox, monkeypatch, seam, caplog):
    from tools import skill_native_audit as audit
    error = RuntimeError('private-exception-path-canary')
    calls = []
    def broken(*args):
        raise RuntimeError('audit-preparation-secret-canary')
    monkeypatch.setattr(audit, seam, broken)
    def callback():
        calls.append(True)
        raise error
    with pytest.raises(RuntimeError) as exc:
        audit.dispatch_skill_call({'secret-canary':'payload-canary'}, {}, callback)
    assert exc.value is error and len(calls) == 1
    warnings = [r for r in caplog.records if r.name == audit.__name__]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == audit._WARNING
    assert warnings[0].args == () and warnings[0].exc_info is None


def test_enabled_exception_completion_is_sanitized_and_propagates(sandbox):
    from tools import skill_native_audit as audit, skill_ledger
    error = RuntimeError('private-handler-error-canary')
    calls = []
    def callback():
        calls.append(True)
        skill_ledger.append_entry('patch', 'fixture', evidence={'private-evidence':'payload-canary'})
        raise error
    with pytest.raises(RuntimeError) as exc:
        audit.dispatch_skill_call({'private-args':'args-canary'}, dict.fromkeys(['task_id','session_id','turn_id','tool_call_id','api_request_id']), callback)
    assert exc.value is error and len(calls) == 1
    assert audit.ledger_invocation_id() is None
    entry, done = records(sandbox)
    assert done['handler_status'] == 'exception'
    assert done['result_hmac_sha256'] is None
    assert done['ledger_status'] == 'appended'
    assert done['ledger_entries'][0]['id'] == skill_ledger.list_entries()[0]['id']
    assert done['invocation_id'] == entry['invocation_id']
    assert all(v is None for v in entry['native_ids'].values())
    text = (sandbox / 'skill_native_audit.jsonl').read_text()
    for value in ['private-handler-error-canary','private-evidence','payload-canary','private-args','args-canary','RuntimeError']:
        assert value not in text


def test_non_exception_exit_leaves_entry_only_and_restores_context(sandbox, handler):
    from tools import skill_native_audit as audit
    exit_signal = KeyboardInterrupt('interrupt-secret-canary')
    calls = []
    def interrupted(args, **kwargs):
        calls.append(audit.ledger_invocation_id())
        raise exit_signal
    handler(interrupted)
    with pytest.raises(KeyboardInterrupt) as exc:
        call({'secret':'interrupt-payload-canary'})
    assert exc.value is exit_signal
    rows = records(sandbox)
    assert len(rows) == 1 and rows[0]['event'] == 'handler_entry'
    assert calls == [rows[0]['invocation_id']]
    assert audit.ledger_invocation_id() is None
    assert 'interrupt-secret-canary' not in (sandbox / 'skill_native_audit.jsonl').read_text()


@pytest.mark.parametrize('mode', ['enabled','disabled','unavailable'])
def test_nested_native_dispatch_isolates_child_ledger_and_restores_parent(sandbox, handler, monkeypatch, mode):
    from tools import skill_native_audit as audit, skill_ledger
    seen = {}
    def nested(args, **kwargs):
        name = args['level']
        seen[name] = audit.ledger_invocation_id()
        if name == 'parent':
            with monkeypatch.context() as mp:
                if mode == 'disabled':
                    mp.setattr(audit, '_enabled', lambda: False)
                elif mode == 'unavailable':
                    def broken(*args):
                        raise OSError('private-prepare-canary')
                    mp.setattr(audit, '_key', broken)
                child = call({'level':'child'})
            assert json.loads(child)['success'] is True
            assert audit.ledger_invocation_id() == seen['parent']
        skill_ledger.append_entry('patch', name, evidence={'level':name})
        return '{"success":true}'
    handler(nested)
    assert json.loads(call({'level':'parent'}))['success'] is True
    assert audit.ledger_invocation_id() is None
    rows = skill_ledger.list_entries()
    by_name = {r['skill']:r for r in rows}
    assert by_name['parent']['native_skill_call'] == seen['parent']
    if mode == 'enabled':
        assert seen['child'] and seen['child'] != seen['parent']
        assert by_name['child']['native_skill_call'] == seen['child']
        assert len(records(sandbox)) == 4
    else:
        assert seen['child'] is None, 'disabled/unavailable child must mask the parent span'
        assert 'native_skill_call' not in by_name['child']
        assert len(records(sandbox)) == 2
    completions = {r['invocation_id']:r for r in records(sandbox) if r['event']=='handler_completion'}
    assert [r['id'] for r in completions[seen['parent']]['ledger_entries']] == [by_name['parent']['id']]


@pytest.mark.parametrize('observer', ['ledger_appended', 'ledger_invocation_id'])
def test_ledger_observer_failure_preserves_written_row_result(sandbox, monkeypatch, handler, caplog, observer):
    from tools import skill_native_audit as audit, skill_ledger
    ids = []
    def broken(*args):
        raise RuntimeError('observer-exception-private-path-canary')
    monkeypatch.setattr(audit, observer, broken)
    def append(args, **kwargs):
        ids.append(skill_ledger.append_entry('patch','fixture',evidence={'secret':'evidence-canary'}))
        return '{"success":true}'
    handler(append)
    assert json.loads(call({'secret':'args-canary'}))['success'] is True
    rows = skill_ledger.list_entries()
    assert len(rows) == 1, 'optional observer must not prevent ledger writes'
    assert ids == [rows[0]['id']], 'observer failure must not turn an already appended ID into None'
    assert rows[0]['evidence'] == {'secret':'evidence-canary'}
    entry, done = records(sandbox)
    assert done['handler_status'] == 'success'
    assert done['ledger_status'] == ('missing' if observer == 'ledger_appended' else 'appended')
    if observer == 'ledger_invocation_id':
        assert 'native_skill_call' not in rows[0]
        assert done['ledger_entries'][0]['id'] == rows[0]['id']
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    assert warnings[0].getMessage() == audit._WARNING
    assert warnings[0].args == () and warnings[0].exc_info is None
    assert 'observer-exception-private-path-canary' not in caplog.text


def test_real_threads_separate_native_spans(sandbox, handler):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, get_ident
    from tools import skill_native_audit as audit, skill_ledger
    # Precreate a valid private key so this test isolates span attribution.
    (sandbox / 'skill_native_audit.key').write_bytes(b'k' * 32)
    (sandbox / 'skill_native_audit.key').chmod(0o600)
    barrier = Barrier(2)
    seen = {}
    def concurrent(args, **kwargs):
        name = args['name']
        seen[name] = (get_ident(), audit.ledger_invocation_id())
        barrier.wait(timeout=10)
        row_id = skill_ledger.append_entry('patch', name, evidence={'owner':name})
        barrier.wait(timeout=10)
        assert audit.ledger_invocation_id() == seen[name][1]
        return json.dumps({'success':True,'row':row_id})
    handler(concurrent)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name:call({'name':name}, session_id=name, tool_call_id='call-'+name), ['thread-a','thread-b']))
    assert all(json.loads(r)['success'] for r in results)
    assert len({v[0] for v in seen.values()}) == 2
    assert len({v[1] for v in seen.values()}) == 2
    assert audit.ledger_invocation_id() is None
    rows = skill_ledger.list_entries()
    journal = records(sandbox)
    assert len(journal) == 4 and len(rows) == 2
    entries = {r['invocation_id']:r for r in journal if r['event']=='handler_entry'}
    completions = {r['invocation_id']:r for r in journal if r['event']=='handler_completion'}
    for row in rows:
        invocation = seen[row['skill']][1]
        assert row['native_skill_call'] == invocation
        assert entries[invocation]['native_ids']['session_id'] == row['skill']
        assert entries[invocation]['native_ids']['tool_call_id'] == 'call-'+row['skill']
        assert [item['id'] for item in completions[invocation]['ledger_entries']] == [row['id']]


@pytest.mark.parametrize('seam', ['completion_append', 'result_digest', 'evidence_digest'])
def test_late_audit_failure_is_safe_and_never_changes_handler_result(sandbox, monkeypatch, handler, caplog, seam):
    from tools import skill_native_audit as audit, skill_ledger
    original_append, original_digest = audit._append, audit._digest
    raw = '{"success":true,"private-result":"result-payload-canary"}'
    calls, ids = [], []
    def broken_append(home, row):
        if row['event'] == 'handler_completion':
            raise OSError('private-append-path-exception-canary')
        original_append(home, row)
    def broken_digest(key, value):
        if (seam == 'result_digest' and value == raw) or (seam == 'evidence_digest' and value == {'secret':'evidence-payload-canary'}):
            raise RuntimeError('private-digest-exception-canary')
        return original_digest(key, value)
    monkeypatch.setattr(audit, '_append', broken_append if seam=='completion_append' else original_append)
    monkeypatch.setattr(audit, '_digest', broken_digest)
    def append(args, **kwargs):
        calls.append(args)
        ids.append(skill_ledger.append_entry('patch','fixture',evidence={'secret':'evidence-payload-canary'}))
        return raw
    handler(append)
    assert call({'private-args':'args-payload-canary'}) == raw
    assert len(calls) == 1
    rows = skill_ledger.list_entries()
    assert ids == [rows[0]['id']]
    assert audit.ledger_invocation_id() is None
    journal = records(sandbox)
    if seam == 'evidence_digest':
        assert len(journal) == 2
        assert journal[1]['handler_status'] == 'success'
        assert journal[1]['ledger_entries'] == [] and journal[1]['ledger_status'] == 'missing'
    else:
        assert len(journal) == 1 and journal[0]['event'] == 'handler_entry'
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    assert warnings[0].getMessage() == audit._WARNING
    assert warnings[0].args == () and warnings[0].exc_info is None
    text = (sandbox / 'skill_native_audit.jsonl').read_text() + caplog.text
    for canary in ['result-payload-canary','args-payload-canary','evidence-payload-canary','private-digest-exception-canary','private-append-path-exception-canary',str(sandbox)]:
        assert canary not in text


@pytest.mark.parametrize('mode', ['enabled','disabled','unavailable'])
def test_nested_callback_exception_restores_exact_parent_span(sandbox, monkeypatch, mode):
    from tools import skill_native_audit as audit
    error = RuntimeError('nested-exception-canary')
    seen = []
    def parent():
        parent_id = audit.ledger_invocation_id()
        with monkeypatch.context() as mp:
            if mode == 'disabled':
                mp.setattr(audit, '_enabled', lambda:False)
            elif mode == 'unavailable':
                def broken(*args):
                    raise OSError('private-preparation-canary')
                mp.setattr(audit, '_key', broken)
            def child():
                seen.append(audit.ledger_invocation_id())
                raise error
            with pytest.raises(RuntimeError) as exc:
                audit.dispatch_skill_call({}, {}, child)
            assert exc.value is error
        assert audit.ledger_invocation_id() == parent_id
        if mode == 'enabled':
            assert seen[0] and seen[0] != parent_id
        else:
            assert seen == [None]
        return '{"success":true}'
    assert audit.dispatch_skill_call({}, {}, parent) == '{"success":true}'
    assert audit.ledger_invocation_id() is None


def test_rollback_observer_failure_warns_safely_without_changing_rollback(sandbox, monkeypatch, caplog):
    from tools import skill_native_audit as audit, skill_ledger
    def broken(*args):
        raise RuntimeError('private-rollback-observer-path-canary')
    monkeypatch.setattr(audit, 'batch_rollback_observed', broken)
    ops = create_args()['operations']
    ops.append({'action':'patch','name':'audit-fixture','old_string':'absent-canary','new_string':'unused'})
    response = call({'operations':ops})
    assert json.loads(response)['success'] is False
    assert not (sandbox / 'skills' / 'audit-fixture').exists()
    entry, done = records(sandbox)
    assert done['handler_status'] == 'failure'
    assert done['batch_rollback'] == 'not_observed'
    assert done['ledger_entries'][0]['id'] == skill_ledger.list_entries()[0]['id']
    warnings = [r for r in caplog.records if r.name == audit.__name__]
    assert len(warnings) == 1, 'rollback observer failure must emit the safe audit warning'
    assert warnings[0].getMessage() == audit._WARNING
    assert warnings[0].args == () and warnings[0].exc_info is None
    assert 'private-rollback-observer-path-canary' not in caplog.text


@pytest.mark.parametrize('mode', ['enabled','unavailable'])
def test_native_escaped_handler_exception_keeps_single_dispatch_and_receipt(sandbox, handler, monkeypatch, caplog, mode):
    from tools import skill_native_audit as audit
    escaped = RuntimeError('native-private-error-canary')
    calls = []
    class UnrenderableError(RuntimeError):
        def __str__(self):
            raise escaped
    def raising(args, **kwargs):
        calls.append(args)
        raise UnrenderableError()
    handler(raising)
    if mode == 'unavailable':
        def broken(*args):
            raise OSError('native-private-path-canary')
        monkeypatch.setattr(audit, '_key', broken)
    assert 'error' in json.loads(call({'payload':'native-private-payload-canary'}))
    assert len(calls) == 1
    assert audit.ledger_invocation_id() is None
    warnings = [r for r in caplog.records if r.name == audit.__name__]
    if mode == 'enabled':
        entry, done = records(sandbox)
        assert done['handler_status'] == 'exception'
        assert done['result_hmac_sha256'] is None
        assert done['ledger_entries'] == []
        assert done['invocation_id'] == entry['invocation_id']
        assert entry['native_ids'] == dict.fromkeys(['task_id','session_id','turn_id','tool_call_id','api_request_id'])
        text = (sandbox / 'skill_native_audit.jsonl').read_text()
        for canary in ['native-private-error-canary','native-private-payload-canary','UnrenderableError','RuntimeError']:
            assert canary not in text
        assert warnings == []
    else:
        assert not (sandbox / 'skill_native_audit.jsonl').exists()
        assert len(warnings) == 1
        assert warnings[0].getMessage() == audit._WARNING
        assert warnings[0].args == () and warnings[0].exc_info is None


def test_unavailable_warning_backend_cannot_prevent_native_dispatch(sandbox, monkeypatch, handler):
    import logging
    from tools import skill_native_audit as audit
    calls = []
    def broken(*args):
        raise OSError('audit-private-failure-canary')
    class BrokenSink(logging.Handler):
        def emit(self, record):
            assert record.getMessage() == audit._WARNING
            assert record.args == () and record.exc_info is None
            raise RuntimeError('logging-private-failure-canary')
    monkeypatch.setattr(audit, '_key', broken)
    def successful(args, **kwargs):
        calls.append(args)
        return '{"success":true}'
    handler(successful)
    sink = BrokenSink()
    audit.logger.addHandler(sink)
    try:
        result = call({'private':'payload-canary'})
    finally:
        audit.logger.removeHandler(sink)
    assert len(calls) == 1, 'warning sink failure must not prevent the handler callback'
    assert json.loads(result)['success'] is True
    assert audit.ledger_invocation_id() is None
