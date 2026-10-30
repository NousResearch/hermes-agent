"""Real process/managed-worker construction uses the operator's watch budget (#139152)."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from tests.waiting import wait_until


def _profile(home, budget):
    from hermes_cli.config import atomic_config_write
    home.mkdir(parents=True, exist_ok=True)
    atomic_config_write(home / 'config.yaml', {'terminal': {'watch_lifetime_max_hits': budget}})


@pytest.mark.platforms('linux', 'macos')
def test_real_background_process_promotes_to_exit_notification_at_configured_limit(tmp_path, monkeypatch):
    from tools import process_registry as pr
    home = tmp_path / 'profile'
    _profile(home, 2)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(home))
    clock = [1000]
    monkeypatch.setattr(pr.time, 'time', lambda: clock[0])
    # Test-owned files release real stdout lines only after the watch is armed.
    source = (
        'import pathlib,sys,threading\n'
        'root=pathlib.Path(sys.argv[1])\n'
        'for index in range(3):\n'
        '    while not (root/str(index)).exists(): threading.Event().wait(.01)\n'
        '    print("READY stage", index, flush=True)\n'
    )
    registry = pr.ProcessRegistry()
    session = registry.spawn_local(
        shlex.join([sys.executable, '-u', '-c', source, str(tmp_path)]),
        cwd=str(tmp_path), task_id='fixture-owner', session_key='fixture-route',
    )
    try:
        assert session.watch_lifetime_max_hits == 2
        session.watch_patterns = ['READY']
        events = []
        for index in range(2):
            clock[0] = 1000 + index * 16
            (tmp_path / str(index)).touch()
            wait_until(lambda: session._watch_hits == index + 1, timeout=15, what='watch match from real child')
        wait_until(lambda: registry.completion_queue.qsize() == 3, timeout=15, what='two matches and limit summary')
        events.extend(registry.completion_queue.get_nowait() for _ in range(3))
        assert [event['type'] for event in events] == ['watch_match', 'watch_match', 'watch_disabled']
        assert session.notify_on_complete and 'lifetime cap of 2' in events[-1]['message']
        (tmp_path / '2').touch()
        assert session._completion_event.wait(15)
        completion = registry.completion_queue.get_nowait()
        assert completion['type'] == 'completion' and completion['exit_code'] == 0
        assert 'READY stage 2' in completion['output']
        assert registry.completion_queue.empty()  # no third pattern wake
    finally:
        if session.process is not None and session.process.poll() is None:
            registry.kill(session.id)
        if session._reader_thread is not None:
            session._reader_thread.join(timeout=15)


def test_fresh_managed_worker_uses_frozen_budget_not_changed_profile(tmp_path):
    home = tmp_path / 'profile'
    _profile(home, 1)
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'PATHEXT')
           if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(home), PYTHONPATH=str(root))
    source = (
        'from agent.safe_worker_policy import _bind_worker_config\n'
        '_bind_worker_config({"terminal":{"watch_lifetime_max_hits":24}})\n'
        'from tools.process_registry import ProcessRegistry\n'
        'session=ProcessRegistry._new_session("fixture","owner","owner","route",None)\n'
        'print(session.watch_lifetime_max_hits)\n'
    )
    result = subprocess.run([sys.executable, '-c', source], cwd=root, env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) == 24
