"""Filesystem boundaries through native dispatch, in owned canonical-runner roots."""
import json
import logging
import multiprocessing
import os
from pathlib import Path
import stat
import tempfile
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tools.test_skill_native_audit import sandbox as sandbox, call, create_args, records

pytestmark = pytest.mark.platforms("linux")


def test_actual_security_runner_scope(tmp_path):
    temp_root = Path(tempfile.gettempdir()).resolve()
    home = Path(os.environ['HERMES_HOME']).resolve()
    assert tmp_path.resolve().is_relative_to(temp_root)
    assert home == (tmp_path / 'hermes_test').resolve()
    assert home.is_relative_to(temp_root)
    assert os.environ.get('HERMES_TEST_ISOLATION')
    assert 'HERMES_STATE_DB_GUARD_BYPASS' not in os.environ
    print(json.dumps({'tmp_path': str(tmp_path), 'tempfile_root': tempfile.gettempdir(), 'db_guard_bypass': False, 'isolation': True}))


@pytest.fixture
def counted(sandbox, monkeypatch):
    import model_tools
    from hermes_constants import get_hermes_home
    assert get_hermes_home() == sandbox
    entry = model_tools.registry._tools['skill_manage']
    original = entry.handler
    seen = []
    def wrapped(args, **kwargs):
        seen.append(1)
        return original(args, **kwargs)
    monkeypatch.setattr(entry, 'handler', wrapped)
    return seen


def _native_once(home, counted):
    assert json.loads(call(create_args()))['success'] is True
    assert counted == [1]
    assert (home / 'skills/audit-fixture/SKILL.md').is_file()


def test_regular_files_accept_existing_valid_key(sandbox, counted):
    key = sandbox / 'skill_native_audit.key'
    key.write_bytes(b'K' * 32)
    key.chmod(0o600)
    journal = sandbox / 'skill_native_audit.jsonl'
    journal.write_text('{"prior":true}\n')
    journal.chmod(0o600)
    _native_once(sandbox, counted)
    assert key.read_bytes() == b'K' * 32
    assert len(records(sandbox)) == 3
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (key, journal))


@pytest.mark.parametrize('name', ['skill_native_audit.key', 'skill_native_audit.jsonl'])
@pytest.mark.parametrize('unsafe', ['symlink', 'permissions', 'hardlink', 'directory', 'owner'])
def test_unsafe_files_leave_targets_unchanged(sandbox, counted, caplog, monkeypatch, name, unsafe):
    path = sandbox / name
    target = sandbox / 'target'
    before = b'K' * 32 if name.endswith('.key') else b'{"prior":true}\n'
    target.write_bytes(before)
    target.chmod(0o600)
    if unsafe == 'symlink':
        path.symlink_to(target)
    elif unsafe == 'hardlink':
        os.link(target, path)
    elif unsafe == 'directory':
        path.mkdir()
    else:
        target.rename(path)
        target = path
        if unsafe == 'owner':
            if os.geteuid() != 0:
                pytest.skip('ownership change requires root')
            os.chown(path, 65534, -1)
        else:
            path.chmod(0o640)
    attempted = []
    original_open = os.open
    def tracked(path_arg, flags, *args, **kwargs):
        if str(path_arg).endswith(name):
            attempted.append(1)
        return original_open(path_arg, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', tracked)
    _native_once(sandbox, counted)
    assert not attempted, 'unsafe file reached open before regular/owner/mode/link refusal'
    if unsafe != 'directory':
        assert target.read_bytes() == before, 'unsafe target was modified'
    assert 'Native skill-call audit unavailable; tool execution is unaffected.' in caplog.text
    if unsafe == 'permissions':
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    if unsafe == 'owner':
        assert path.stat().st_uid == 65534


def _fifo_child(home, name, connection, counted):
    from tools import skill_native_audit as audit
    original_open, original_read = os.open, os.read
    def opened(path, flags, *args, **kwargs):
        if str(path).endswith(name):
            connection.send(('ready', str(home / name), 'open'))
        return original_open(path, flags, *args, **kwargs)
    def read(fd, size):
        if stat.S_ISFIFO(os.fstat(fd).st_mode):
            connection.send(('ready', str(home / name), 'read'))
        return original_read(fd, size)
    os.open, os.read = opened, read
    class Capture(logging.Handler):
        def emit(self, record):
            pass
    audit.logger.addHandler(Capture())
    # Acknowledgment precedes the guarded operation even when no syscall is reached.
    connection.send(('admitted', str(home / name)))
    try:
        _native_once(home, counted)
        connection.send(('done', counted == [1]))
    finally:
        connection.close()


@pytest.mark.parametrize('name', ['skill_native_audit.key', 'skill_native_audit.jsonl'])
def test_writerless_fifo_naturally_rejects(sandbox, counted, name):
    os.mkfifo(sandbox / name, 0o600)
    parent, child = multiprocessing.get_context('fork').Pipe()
    process = multiprocessing.get_context('fork').Process(target=_fifo_child, args=(sandbox, name, child, counted))
    process.start()
    child.close()
    events = []
    try:
        assert parent.poll(15), 'no admission acknowledgment'
        events.append(parent.recv())
        assert events[0] == ('admitted', str(sandbox / name))
        while parent.poll(3):
            try:
                event = parent.recv()
            except EOFError:
                break
            events.append(event)
            if event[0] == 'done':
                break
        natural = any(event[0] == 'done' for event in events)
        if not natural:
            assert any(event[0] == 'ready' for event in events), 'no blocked-syscall readiness acknowledgment'
            print(json.dumps({'fifo_outcome': 'watchdog_terminated_RED', 'events': events}))
        assert natural, 'watchdog termination is not natural FIFO rejection'
        assert not any(event[0] == 'ready' for event in events), 'FIFO reached forbidden open/read'
        process.join(5)
        assert process.exitcode == 0
        assert stat.S_ISFIFO((sandbox / name).lstat().st_mode)
        print(json.dumps({'fifo_outcome': 'natural_GREEN', 'events': events}))
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()


@pytest.mark.parametrize('size', [0, 31, 33])
def test_bad_existing_key_is_never_overwritten(sandbox, counted, caplog, size):
    key = sandbox / 'skill_native_audit.key'
    key.write_bytes(b'K' * size)
    key.chmod(0o600)
    _native_once(sandbox, counted)
    assert key.read_bytes() == b'K' * size
    assert not (sandbox / 'skill_native_audit.jsonl').exists()
    assert 'Native skill-call audit unavailable' in caplog.text


def test_missing_security_primitive_declines_io(sandbox, counted, caplog, monkeypatch):
    monkeypatch.delattr(os, 'O_NOFOLLOW')
    _native_once(sandbox, counted)
    assert not (sandbox / 'skill_native_audit.key').exists()
    assert 'Native skill-call audit unavailable' in caplog.text


def test_profile_symlink_is_not_followed(sandbox, counted, caplog, monkeypatch):
    alias = sandbox.parent / 'profile-alias'
    alias.symlink_to(sandbox, target_is_directory=True)
    from tools import skill_native_audit as audit
    monkeypatch.setattr(audit, 'get_hermes_home', lambda: alias)
    _native_once(sandbox, counted)
    assert not (sandbox / 'skill_native_audit.key').exists()
    assert 'Native skill-call audit unavailable' in caplog.text


def _overlap_worker(home, index, gate, connection):
    from tools import skill_native_audit as audit
    try:
        gate.wait(15)
        with ThreadPoolExecutor(max_workers=4) as pool:
            digests = list(pool.map(lambda _: hashlib.sha256(audit._key(home)).hexdigest(), range(4)))
        for record in range(16):
            audit._append(home, {'worker': index, 'record': record, 'sequence': list(range(2048))})
        connection.send(('ok', digests))
    except Exception as error:
        connection.send(('error', type(error).__name__))
    finally:
        connection.close()


def test_overlapping_process_threads_share_durable_first_key(sandbox, monkeypatch):
    from tools import skill_native_audit as audit
    original = os.urandom
    def slow_random(size):
        time.sleep(0.05)
        return original(size)
    monkeypatch.setattr(os, 'urandom', slow_random)
    context = multiprocessing.get_context('fork')
    gate = context.Barrier(6)
    owned = []
    try:
        for index in range(6):
            parent, child = context.Pipe()
            process = context.Process(target=_overlap_worker, args=(sandbox, index, gate, child))
            process.start()
            child.close()
            owned.append((process, parent))
        replies = []
        for process, parent in owned:
            assert parent.poll(30), 'owned worker exceeded watchdog'
            replies.append(parent.recv())
            process.join(5)
            assert process.exitcode == 0
        assert all(reply[0] == 'ok' for reply in replies), 'concurrent audit IO failed'
        digests = [digest for _, group in replies for digest in group]
        assert len(set(digests)) == 1, 'workers observed different first keys'
        key = sandbox / 'skill_native_audit.key'
        assert key.stat().st_size == 32
        assert hashlib.sha256(key.read_bytes()).hexdigest() == digests[0]
        assert hashlib.sha256(audit._key(sandbox)).hexdigest() == digests[0]
        rows = records(sandbox)
        assert len(rows) == 96
        assert {(row['worker'], row['record']) for row in rows} == {(i, j) for i in range(6) for j in range(16)}
        assert all(row['sequence'] == list(range(2048)) for row in rows)
        assert all(stat.S_IMODE((sandbox / name).stat().st_mode) == 0o600 for name in ['skill_native_audit.key', 'skill_native_audit.jsonl'])
        print('overlap: 6 processes, 24 concurrent first-key thread callers, 96 complete JSONL rows')
    finally:
        for process, parent in owned:
            if process.is_alive():
                process.kill()
            process.join(5)
            parent.close()


def _audit_fd(fd):
    return Path(os.readlink('/proc/self/fd/' + str(fd))).name in ('skill_native_audit.key', 'skill_native_audit.jsonl')


def test_short_writes_persist_exact_bytes_and_sync(sandbox, monkeypatch):
    from tools import skill_native_audit as audit
    original_write, original_sync = os.write, os.fsync
    synced = []
    def short(fd, data):
        return original_write(fd, data[:7]) if _audit_fd(fd) else original_write(fd, data)
    def sync(fd):
        synced.append(Path(os.readlink('/proc/self/fd/' + str(fd))).name)
        return original_sync(fd)
    monkeypatch.setattr(os, 'urandom', lambda size: b'K' * size)
    monkeypatch.setattr(os, 'write', short)
    monkeypatch.setattr(os, 'fsync', sync)
    assert bool(audit._key(sandbox) == b'K' * 32)
    assert bool((sandbox / 'skill_native_audit.key').read_bytes() == b'K' * 32), 'incomplete key write'
    row = {'event': 'controlled_probe', 'index': 7}
    audit._append(sandbox, row)
    expected = (json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
    assert bool((sandbox / 'skill_native_audit.jsonl').read_bytes() == expected), 'incomplete journal write'
    assert 'skill_native_audit.key' in synced and 'skill_native_audit.jsonl' in synced
    assert synced.count('profile') == 2


@pytest.mark.parametrize('name', ['skill_native_audit.key', 'skill_native_audit.jsonl'])
@pytest.mark.parametrize('failure', ['zero', 'exception'])
def test_write_failure_is_sanitized_and_tool_runs_once(sandbox, counted, caplog, monkeypatch, name, failure):
    original = os.write
    if name.endswith('.jsonl'):
        key = sandbox / 'skill_native_audit.key'
        key.write_bytes(b'K' * 32)
        key.chmod(0o600)
        journal = sandbox / name
        journal.write_bytes(b'{"prior":true}\n')
        journal.chmod(0o600)
    def failed(fd, data):
        if Path(os.readlink('/proc/self/fd/' + str(fd))).name == name:
            if failure == 'zero':
                return 0
            raise OSError('secret-write-error-canary')
        return original(fd, data)
    monkeypatch.setattr(os, 'write', failed)
    _native_once(sandbox, counted)
    assert 'Native skill-call audit unavailable; tool execution is unaffected.' in caplog.text
    assert 'secret-write-error-canary' not in caplog.text
    expected = b'' if name.endswith('.key') else b'{"prior":true}\n'
    assert bool((sandbox / name).read_bytes() == expected)
    if name.endswith('.key'):
        assert not (sandbox / 'skill_native_audit.jsonl').exists()


def test_fragmented_overlapping_appends_are_complete(sandbox, monkeypatch):
    original = os.write
    def short(fd, data):
        if _audit_fd(fd):
            written = original(fd, data[:512])
            time.sleep(0.0001)
            return written
        return original(fd, data)
    monkeypatch.setattr(os, 'write', short)
    test_overlapping_process_threads_share_durable_first_key(sandbox, monkeypatch)


def test_file_swap_to_symlink_never_follows(sandbox, counted, monkeypatch, caplog):
    path = sandbox / 'skill_native_audit.key'
    path.write_bytes(b'K' * 32)
    path.chmod(0o600)
    target = sandbox / 'swap-target'
    original_open = os.open
    followed = []
    def swap(path_arg, flags, *args, **kwargs):
        if str(path_arg) == 'skill_native_audit.key':
            path.rename(target)
            path.symlink_to(target)
            try:
                fd = original_open(path_arg, flags, *args, **kwargs)
            except OSError:
                raise
            followed.append(True)
            return fd
        return original_open(path_arg, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', swap)
    _native_once(sandbox, counted)
    assert not followed, 'forbidden symlink was followed by open'
    assert bool(target.read_bytes() == b'K' * 32)
    assert 'Native skill-call audit unavailable' in caplog.text


def test_directory_substitution_declines_old_descriptor_io(sandbox, counted, monkeypatch, caplog):
    path = sandbox / 'skill_native_audit.key'
    path.write_bytes(b'K' * 32)
    path.chmod(0o600)
    old = sandbox.parent / 'displaced-profile'
    original_open = os.open
    def swap(path_arg, flags, *args, **kwargs):
        if str(path_arg) == 'skill_native_audit.key':
            sandbox.rename(old)
            sandbox.mkdir()
            replacement = sandbox / 'skill_native_audit.key'
            replacement.write_bytes(b'K' * 32)
            replacement.chmod(0o600)
        return original_open(path_arg, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', swap)
    _native_once(sandbox, counted)
    assert bool((old / 'skill_native_audit.key').read_bytes() == b'K' * 32)
    assert bool((sandbox / 'skill_native_audit.key').read_bytes() == b'K' * 32)
    assert not (sandbox / 'skill_native_audit.jsonl').exists()
    assert not (old / 'skill_native_audit.jsonl').exists()
    assert 'Native skill-call audit unavailable' in caplog.text


def test_profile_parent_traversal_declines_audit(sandbox, counted, monkeypatch, caplog):
    alias = sandbox.parent / 'profile-alias'
    alias.symlink_to(sandbox, target_is_directory=True)
    from tools import skill_native_audit as audit
    monkeypatch.setattr(audit, 'get_hermes_home', lambda: alias / '..' / 'profile')
    _native_once(sandbox, counted)
    assert not (sandbox / 'skill_native_audit.key').exists()
    assert 'Native skill-call audit unavailable' in caplog.text


def test_existing_thread_lifecycle_with_valid_private_key(sandbox, monkeypatch):
    # Reuse the exact legacy assertions; precreating 0600 fixes its envelope
    # without editing the out-of-scope legacy fixture (write_bytes preserves mode).
    from tests.tools import test_skill_native_audit_lifecycle as legacy
    from hermes_constants import get_hermes_home
    assert get_hermes_home() == sandbox
    key = sandbox / 'skill_native_audit.key'
    key.write_bytes(b'k' * 32)
    key.chmod(0o600)
    handler = legacy.handler.__wrapped__(monkeypatch)
    legacy.test_real_threads_separate_native_spans(sandbox, handler)
    assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_explicit_empty_native_ids_are_retained_verification_control(sandbox, counted):
    # Existing representation behavior, not a new ID normalization feature.
    ids = dict.fromkeys(('task_id', 'session_id', 'tool_call_id', 'turn_id', 'api_request_id'), '')
    assert json.loads(call(create_args(), **ids))['success'] is True
    assert counted == [1]
    entry, done = records(sandbox)
    assert entry['native_ids'] == ids
    assert entry['invocation_id'] == done['invocation_id']
    assert done['handler_status'] == 'success'


@pytest.mark.parametrize('mode', [
    'partial_entry', 'partial_completion', 'no_fault_empty', 'no_fault_newline', 'zero_entry',
])
def test_incomplete_tail_refuses_later_native_audit_without_repair(
    sandbox, counted, caplog, monkeypatch, mode,
):
    from tools import skill_ledger, skill_native_audit as audit
    key = sandbox / 'skill_native_audit.key'
    key.write_bytes(b'K' * 32)
    key.chmod(0o600)
    journal = sandbox / 'skill_native_audit.jsonl'
    prior = b'' if mode in ('no_fault_empty', 'zero_entry') else b'{"prior":true}\n'
    journal.write_bytes(prior)
    journal.chmod(0o600)
    original_write = os.write
    target_event = b'"event":"handler_completion"' if mode == 'partial_completion' else b'"event":"handler_entry"'
    injected = []
    fragment = []

    def fault(fd, data):
        if Path(os.readlink('/proc/self/fd/' + str(fd))).name == journal.name:
            if injected == ['partial']:
                injected.append('raised')
                raise OSError('private-partial-write-error-canary')
            if not injected and target_event in bytes(data):
                if mode == 'zero_entry':
                    injected.append('zero')
                    return 0
                if mode.startswith('partial'):
                    fragment.append(bytes(data[:19]))
                    injected.append('partial')
                    return original_write(fd, data[:19])
        return original_write(fd, data)

    def warnings():
        rows = [r for r in caplog.records if r.name == audit.logger.name]
        assert all(r.getMessage() == audit._WARNING and not r.args and not r.exc_info for r in rows)
        return len(rows)

    snapshots, warning_counts = [], []
    for index in range(3):
        caplog.clear()
        args = create_args('tail-fixture-' + str(index))
        if index == 0:
            with monkeypatch.context() as patch:
                patch.setattr(os, 'write', fault)
                result = call(args, session_id='tail-session-' + str(index))
        else:
            # The real fault is removed before either healthy native call.
            result = call(args, session_id='tail-session-' + str(index))
        assert json.loads(result)['success'] is True
        assert counted == [1] * (index + 1), 'original native handler must execute exactly once'
        assert (sandbox / 'skills' / args['operations'][0]['name'] / 'SKILL.md').is_file()
        snapshots.append(journal.read_bytes())
        warning_counts.append(warnings())

    ledger = {row['skill']: row for row in skill_ledger.list_entries()}
    assert set(ledger) == {'tail-fixture-' + str(i) for i in range(3)}
    assert snapshots[0].startswith(prior)
    print(json.dumps({'mode': mode, 'warning_counts': warning_counts,
                      'later_bytes_unchanged': [raw == snapshots[0] for raw in snapshots[1:]],
                      'native_handler_calls': len(counted), 'ledger_rows': len(ledger)}))
    if mode.startswith('partial'):
        assert injected == ['partial', 'raised'] and len(fragment[0]) == 19
        assert snapshots[0].endswith(fragment[0]) and not snapshots[0].endswith(b'\n')
        assert warning_counts == [1, 1, 1], 'healthy later calls must warn, not silently corrupt an incomplete tail'
        assert snapshots == [snapshots[0]] * 3, 'damaged and historical bytes must remain exactly unchanged'
        assert all('native_skill_call' not in ledger['tail-fixture-' + str(i)] for i in (1, 2))
        first = ledger['tail-fixture-0']
        if mode == 'partial_completion':
            complete_prefix = snapshots[0][:-19]
            entry = json.loads(complete_prefix.splitlines()[-1])
            assert entry['event'] == 'handler_entry'
            assert first['native_skill_call'] == entry['invocation_id'], 'retain original historical linkage'
        else:
            assert snapshots[0] == prior + fragment[0]
            assert 'native_skill_call' not in first
    else:
        assert injected == (['zero'] if mode == 'zero_entry' else [])
        assert warning_counts == ([1, 0, 0] if mode == 'zero_entry' else [0, 0, 0])
        if mode == 'zero_entry':
            assert snapshots[0] == b''
            assert 'native_skill_call' not in ledger['tail-fixture-0']
        rows = [row for row in records(sandbox) if row.get('event')]
        expected_indexes = (1, 2) if mode == 'zero_entry' else (0, 1, 2)
        assert len(rows) == 2 * len(expected_indexes)
        for index in expected_indexes:
            invocation = ledger['tail-fixture-' + str(index)]['native_skill_call']
            entry, done = [row for row in rows if row['invocation_id'] == invocation]
            assert entry['event'] == 'handler_entry' and entry['native_ids']['session_id'] == 'tail-session-' + str(index)
            assert done['event'] == 'handler_completion' and done['handler_status'] == 'success'
            assert done['ledger_entries'][0]['id'] == ledger['tail-fixture-' + str(index)]['id']
    assert stat.S_IMODE(key.stat().st_mode) == stat.S_IMODE(journal.stat().st_mode) == 0o600
