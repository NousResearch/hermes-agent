"""Public dispatch/restart diagnostics for an owner lost before any child finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("split,missing_writer", [(False, False), (True, False), (True, True)])
def test_unfinished_delegation_recovery_keeps_transcript_locator(tmp_path, split, missing_writer):
    repo = str(Path(__file__).resolve().parents[2])
    handle_path = tmp_path / "dispatch.json"
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo,
           "REPRO_HANDLE": str(handle_path), "REPRO_SPLIT": str(int(split)),
           "REPRO_MISSING": str(int(missing_writer))}
    (tmp_path / 'config.yaml').write_text('delegation:\n  independent_completions: true\n', encoding='utf-8')
    producer = r'''
import json, os, threading
from pathlib import Path
from unittest.mock import MagicMock
import tools.delegate_tool as dt
parent = MagicMock()
parent._delegate_depth = 0
parent.session_id = "diagnostic-parent"
parent._interrupt_requested = False
parent._active_children = []
parent._active_children_lock = None
started = threading.Event()
blocked = threading.Event()
def child(**kw):
    started.set()
    blocked.wait(60)
def build(**kw):
    c = MagicMock()
    c._delegate_role = "leaf"
    c._subagent_id = "diagnostic-child"
    return c
creds = dict(model="m", provider=None, base_url=None, api_key=None,
             api_mode=None, command=None, args=None)
dt._build_child_agent = build
dt._run_single_child = child
dt._resolve_delegation_credentials = lambda *a, **k: creds
if os.environ['REPRO_MISSING'] == '1':
    import tools.delegation_live_log as live
    original = live.create_live_transcripts
    def missing(*args, **kwargs):
        ident, writers, paths = original(*args, **kwargs)
        writers[0] = None
        return ident, writers, paths[1:]
    live.create_live_transcripts = missing
kwargs = {'tasks': [{'goal': 'first diagnostic task'}, {'goal': 'second diagnostic task'}]} if os.environ['REPRO_SPLIT'] == '1' else {'goal': 'unfinished diagnostic task'}
handle = json.loads(dt.delegate_task(**kwargs, background=True, parent_agent=parent))
assert handle["status"] == "dispatched", handle
assert started.wait(10), "child did not start"
Path(os.environ["REPRO_HANDLE"]).write_text(json.dumps(handle), encoding="utf-8")
os._exit(0)
'''
    first = subprocess.run([sys.executable, "-c", producer], cwd=repo, env=env,
                           text=True, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stdout + first.stderr
    handle = json.loads(handle_path.read_text(encoding="utf-8"))
    transcripts = handle["live_transcripts"]
    assert all(Path(path).is_file() for path in transcripts)
    consumer = r'''
import json, queue
from tools import async_delegation as ad
from tools.process_registry import format_process_notification
q = queue.Queue()
count = ad.restore_undelivered_completions(q)
events = [q.get_nowait() for _ in range(count)]
print(json.dumps([{"event": event, "message": format_process_notification(event)} for event in events]))
'''
    second = subprocess.run([sys.executable, "-c", consumer], cwd=repo, env=env,
                            text=True, capture_output=True, timeout=20)
    assert second.returncode == 0, second.stdout + second.stderr
    restored = json.loads(second.stdout.strip().splitlines()[-1])
    assert len(restored) == (2 if split else 1)
    by_index = {}
    for item in restored:
        event = item['event']
        assert event["status"] == "unknown"
        assert event["last_known_status"] == "running"
        assert "running" in item["message"]
        assert not event.get("summary")
        paths = event['task_transcripts']
        assert len(paths) <= 1
        for index, path in paths.items():
            assert path in item['message']
            assert Path(path).name == f'task-{index}.log'
            assert index not in by_index
            by_index[index] = path
        for path in transcripts:
            assert (path in item['message']) == (path in paths.values())
    assert set(by_index) == ({'1'} if missing_writer else {'0', '1'} if split else {'0'})
    assert set(by_index.values()) == set(transcripts)

