"""Definite pre-spawn log failures preserve the error and durable supervision identity."""
from unittest.mock import Mock

import pytest

from tests.gateway.test_process_work_results import owner


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['mkdir', 'open', 'postspawn'])
async def test_log_failure_receipt_preserves_spawn_boundary(monkeypatch, failure):
    from gateway.session_context import scoped_current_session_id
    from tools import process_registry as processes
    from tools import process_registry_output_log as logs
    from tools import process_work_results as work
    from hermes_state import SessionDB

    runner, authority, key, sid = await owner(monkeypatch)
    registry = processes.ProcessRegistry()
    original = PermissionError(13, 'literal log failure', '/fixture/log')
    spawn = Mock(side_effect=original if failure == 'postspawn' else AssertionError('must not spawn'))
    monkeypatch.setattr(registry, '_spawn_job', spawn)
    monkeypatch.setattr(registry, '_scope_argv', lambda *args: ['not-executed'])
    if failure == 'mkdir':
        monkeypatch.setattr(logs, 'make_user_dir', Mock(side_effect=original))
    elif failure == 'open':
        monkeypatch.setattr(processes, 'open_output_log', Mock(side_effect=original))
    with scoped_current_session_id(sid), pytest.raises(PermissionError) as caught:
        registry.spawn_local('not-executed', session_key=key,
                             completion_work=[{'work_id': 'build', 'incarnation': '1'}])
    assert caught.value is original
    assert spawn.call_count == (failure == 'postspawn')
    if failure == 'postspawn':
        assert not hasattr(original, 'process_receipt')
        assert work.records(authority.db)[0]['disposition'] == 'enrolled'
        return
    receipt = original.process_receipt
    assert receipt['completion_reason'] == 'never_started'
    assert receipt['work_results'][0]['supervision'] == 'never_started'
    with SessionDB(authority.db.db_path) as reopened:
        row = work.records(reopened)[0]
    assert row['process_id'] == receipt['session_id']
    assert row['disposition'] == 'never_started'
    assert row['payload'] is None
    assert registry.poll(receipt['session_id'])['completion_reason'] == 'never_started'
    with scoped_current_session_id(sid), pytest.raises(ValueError, match='already enrolled'):
        registry.spawn_local('not-executed', session_key=key,
                             completion_work=[{'work_id': 'build', 'incarnation': '1'}])
    spawn.assert_not_called()
    await runner._drain_watch_notifications(registry.completion_queue)
    assert work.records(authority.db)[0]['disposition'] == 'never_started_admitted'
