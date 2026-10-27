"""Original readers retain their producer home after the caller changes profiles."""
import asyncio
import json
import shlex
import sys
import threading

import pytest

from tests.gateway.test_native_job_scope_stop import spawn
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit_home', [False, True], ids=['ambient-home', 'explicit-profile-context'])
async def test_original_reader_keeps_metadata_out_of_later_profile(tmp_path, monkeypatch, explicit_home):
    from gateway.session_ancillary import record_process_stop, retry_process_stop
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    from tools import process_registry as module, process_registry_job as jobs
    home_a, home_b = tmp_path / 'prior', tmp_path / 'current'
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setattr(module, '_systemd_run_user_scope_available', lambda: False)
    monkeypatch.setenv('HERMES_HOME', str(home_a))
    token = set_hermes_home_override(home_a if explicit_home else None)
    old = module.ProcessRegistry()
    entered, release, provenance = threading.Event(), threading.Event(), []
    move = old._move_to_finished
    def held_reader(process):
        entered.set()
        assert release.wait(15)
        provenance.append((get_hermes_home(), module._checkpoint_path(), jobs.job_receipt_path(process.id)))
        return move(process)
    monkeypatch.setattr(old, '_move_to_finished', held_reader)
    own_a = await spawn(old, home_a, 'exec ' + shlex.join([sys.executable, '-c',
        'print("original prior output",flush=True); raise SystemExit(7)']))
    reset_hermes_home_override(token)
    db, own_b = None, None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert jobs.job_quiescent(own_a)
        db, authority = _authority(home_b, monkeypatch)
        current = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', current)
        own_b = await spawn(current, home_b, 'exec ' + shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(15)']))
        await asyncio.to_thread(wait_until, lambda: jobs.read_job_receipt(own_b) is not None, timeout=5)
        assert current._host_pid_is_ours(own_b.pid, own_b.host_start_time)
        release.set()
        assert await asyncio.to_thread(own_a._completion_event.wait, 5)
        assert provenance == [(home_a, home_a / 'processes.json',
                               home_a / 'logs' / 'process-output' / (own_a.id + '.scope.exit'))]
        assert own_a.id not in {r['session_id'] for r in json.loads((home_b / 'processes.json').read_text())}
        assert not (home_b / 'logs' / 'process-results' / (own_a.id + '.json')).exists()
        prior = json.loads((home_a / 'logs' / 'process-results' / (own_a.id + '.json')).read_text())
        assert prior['result_complete'] and prior['exit_code'] == 7
        assert prior['output'].count('original prior output') == 1
        requests, request = [], jobs.request_job_stop
        def observed(registry, process):
            requests.append(process.id)
            return request(registry, process)
        monkeypatch.setattr(jobs, 'request_job_stop', observed)
        record_process_stop(authority, REF, 0)
        assert await retry_process_stop(authority, REF)
        assert own_b.id in requests and jobs.job_quiescent(own_b)
        assert get_hermes_home() == home_b  # The thread pin never mutates its caller's environment.
    finally:
        release.set()
        assert await asyncio.to_thread(own_a._completion_event.wait, 5)
        if own_b is not None:
            await asyncio.to_thread(jobs.request_job_stop, current, own_b)
            assert await asyncio.to_thread(own_b._completion_event.wait, 5)
        if db is not None:
            db.close()
