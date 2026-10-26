"""Foreign checkpoint writers preserve resource evidence until native reconciliation."""
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize('fate', ['gone', 'recycled', 'unreadable'])
def test_foreign_writer_leaves_host_verdict_to_recovery(tmp_path, monkeypatch, fate):
    from tools import process_registry as module
    registry = module.ProcessRegistry()
    path = module._checkpoint_path()
    neighbor = subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait(120)'])
    try:
        birth = registry._safe_host_start_time(neighbor.pid)
        assert type(birth) is int and birth > 0
        entry = dict(session_id='proc_prior', command='native host job', pid_scope='host',
                     pid=2**31 - 1 if fate == 'gone' else neighbor.pid,
                     host_start_time=birth + (1 if fate == 'recycled' else 0))
        foreign = dict(session_id='proc_foreign_backend', command='remote neighbor',
                       pid=neighbor.pid, pid_scope='unfamiliar-backend',
                       task_id='foreign-owner', parent_session_id='foreign-parent', session_key='foreign-route')
        path.write_text(json.dumps([entry, foreign]))
        other_home = tmp_path / 'other-profile'
        other_home.mkdir()
        other_checkpoint = other_home / 'processes.json'
        other_checkpoint.write_text(json.dumps([{**entry, 'task_id': 'other-profile-owner'}]))
        other_before = other_checkpoint.read_bytes()
        if fate == 'unreadable':
            monkeypatch.setattr(registry, '_safe_host_start_time', lambda pid: None)
        monkeypatch.setattr(registry, '_terminate_host_pid',
                            lambda *a, **kw: pytest.fail('recovery signaled a foreign or unproved PID'))
        assert registry._write_checkpoint()
        assert json.loads(path.read_text()) == [entry, foreign]
        assert registry.recover_from_checkpoint() == (1 if fate == 'unreadable' else 0)
        rows = json.loads(path.read_text())
        assert foreign in rows and neighbor.poll() is None
        assert ('proc_prior' in {row['session_id'] for row in rows}) == (fate == 'unreadable')
        assert other_checkpoint.read_bytes() == other_before
        if fate == 'unreadable':
            # Only the original waitable handle supplies positive cleanup here.
            neighbor.terminate()
            neighbor.wait(timeout=10)
        # An explicit native reconciliation exclusion is still authoritative;
        # the unrelated opaque neighbor survives the same atomic rewrite.
        assert module.ProcessRegistry()._write_checkpoint(exclude_ids={'proc_prior'})
        assert json.loads(path.read_text()) == [foreign]
        assert other_checkpoint.read_bytes() == other_before
    finally:
        if neighbor.poll() is None:
            neighbor.kill()
            neighbor.wait(timeout=10)
