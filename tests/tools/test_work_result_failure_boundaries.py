"""Independent failure-boundary regressions; no real process or external I/O."""
import io
import os
import sqlite3
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import process_identity as identity
from tools import process_registry as registry
from tools import process_work_results as work
from tools.process_registry_output_log import _FINAL_DRAIN_MAX_BYTES


@pytest.mark.parametrize('kind,code', [(FileNotFoundError, 2), (PermissionError, 13)])
def test_spawn_preserves_original_os_error(kind, code):
    original = kind(code, 'literal failure', '/literal/path')
    with patch.object(identity.subprocess, 'Popen', side_effect=original), pytest.raises(kind) as caught:
        identity.spawn_owned(['not-executed'], purpose='test')
    assert caught.value is original
    assert getattr(original, '_hermes_child_not_started', False)


@pytest.mark.parametrize('seam', ['save', 'read'])
def test_secondary_receipt_failure_keeps_original_spawn_identity(seam):
    owner = registry.ProcessRegistry()
    source = registry.ProcessSession(id='proc_boundary', command='not-executed',
                                    completion_work_home=os.environ['HERMES_HOME'])
    original = FileNotFoundError(2, 'original rejection', '/missing')
    original._hermes_child_not_started = True
    secondary = sqlite3.OperationalError('secondary receipt unavailable')
    with ExitStack() as stack:
        stack.enter_context(patch.object(owner, '_new_session', return_value=source))
        stack.enter_context(patch.object(owner, '_spawn_env', return_value={}))
        stack.enter_context(patch.object(owner, '_scope_argv', return_value=['not-executed']))
        stack.enter_context(patch.object(registry, 'worker_output_log', return_value=None))
        stack.enter_context(patch.object(owner, '_spawn_job', side_effect=original))
        stack.enter_context(patch.object(work, 'mark_never_started'))
        stack.enter_context(patch.object(registry, 'save_completed_result',
                                         side_effect=secondary if seam == 'save' else None))
        stack.enter_context(patch.object(work, 'receipts', return_value=[],
                                         side_effect=secondary if seam == 'read' else None))
        with pytest.raises(FileNotFoundError) as caught:
            owner.spawn_local('not-executed')
    assert caught.value is original
    assert caught.value.process_receipt['session_id'] == source.id
    assert caught.value.process_receipt['work_result_error'] == 'spawn_receipt_storage_unresolved'


def test_final_drain_cap_retains_unknown_instead_of_claiming_missing():
    owner = registry.ProcessRegistry()
    source = registry.ProcessSession(id='proc_large', command='none',
                                    completion_work_home=os.environ['HERMES_HOME'])
    source.output_log = '/fixture/log'
    source.process = SimpleNamespace(poll=lambda: 0, wait=lambda: 0, returncode=0)
    data = b'x\n' * (_FINAL_DRAIN_MAX_BYTES // 2) + (
        work.PREFIX + '{"work_id":"build","incarnation":"1","status":"completed","evidence":"present"}\n'
    ).encode()
    with patch('builtins.open', return_value=io.BytesIO(data)), patch.object(work, '_record') as record, \
            patch.object(owner, '_finish_reader') as finish:
        owner._log_reader_loop(source)
    assert finish.called
    assert not record.called  # preserve the existing bounded final drain
    assert source.work_result_error == 'output_log_incomplete'


def test_postspawn_error_is_not_marked_never_started():
    original = PermissionError(13, 'ownership failed after spawn')
    with patch.object(identity.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)), \
            patch.object(identity, 'own_process_groups', side_effect=original), pytest.raises(PermissionError) as caught:
        identity.spawn_owned(['not-executed'], purpose='test')
    assert caught.value is original
    assert not getattr(original, '_hermes_child_not_started', False)
