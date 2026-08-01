"""Consumption persists across processes, without turning status reads into acknowledgements."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.process_registry import ProcessRegistry, ProcessSession

REPO_ROOT = Path(__file__).resolve().parents[2]


def _finished(registry, sid):
    session = ProcessSession(id=sid, command="test", task_id="owner", owner_task_id="owner",
                             started_at=time.time(), notify_on_complete=True)
    session.append_output("first\nlast\n")
    session.mark_exited(0)
    session._completion_event.set()
    registry._finished[sid] = session
    return session


@pytest.mark.parametrize("action,consumed", [
    ("wait", True), ("log", True), ("kill", True), ("kill_running", True),
    ("kill_detached_gone", True), ("poll", False), ("log_head", False), ("kill_cleanup", False),
])
def test_only_consuming_results_are_visible_to_another_process(tmp_path, monkeypatch, action, consumed):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    sid = "proc_0123456789ab"
    writer = '''
import json, sys, time
from types import SimpleNamespace
from tools.process_registry import ProcessSession, process_registry as registry
from tools.registry import registry as tools
sid, action = sys.argv[1:]
session = ProcessSession(id=sid, command="test", task_id="owner", owner_task_id="owner",
                         started_at=time.time(), notify_on_complete=True)
session.append_output("first\\nlast\\n")
if action in {"kill_running", "kill_detached_gone"}:
    registry._running[sid] = session
    if action == "kill_running":
        session.env_ref = SimpleNamespace(execute=lambda *a, **kw: {"returncode": 0})
        session.pid, session.pid_scope = 123, "sandbox"
        registry._post_kill_survivors = lambda s: []
    else:
        session.detached, session.pid, session.pid_scope = True, 123, "host"
        fates = iter(("running", "gone"))  # The PID disappears after get(), before the kill.
        registry._detached_host_fate = lambda *args: next(fates)
    result = registry.kill_process(sid)
else:
    session.mark_exited(0)
    session._completion_event.set()
    registry._finished[sid] = session
    if action == "kill_cleanup":
        result = registry.kill_process(sid, consume_output=False)
    else:
        request = {"action": "log" if action == "log_head" else action, "session_id": sid[:9]}
        if action == "log_head":
            request.update(offset=0, limit=1)
        result = json.loads(tools.get_entry("process_manage").handler(request))
assert result["status"] in {"exited", "killed", "already_exited"}, result
print(json.dumps(result))
'''
    completed = subprocess.run([sys.executable, "-c", writer, sid, action], cwd=REPO_ROOT,
                               env={**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)},
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert "first" in result.get("output", result.get("output_preview", ""))
    reader = ProcessRegistry()
    assert sid not in reader._completion_consumed
    assert reader.is_completion_consumed(sid) is consumed
    assert (home / "processes-consumed" / f"{sid}.consumed").is_file() is consumed
    assert reader._drain_should_skip(sid, skip_poll_observed=False) is consumed
    _finished(reader, sid)
    assert bool(reader.unread_completions_owned_by("owner")) is not consumed
    reader.completion_queue.put({"type": "completion", "session_id": sid, "exit_code": 0,
                                 "command": "test", "output": "first\nlast\n", "owner_task_id": "owner"})
    assert bool(reader.drain_notifications(skip_poll_observed=False)) is not consumed


def test_markers_follow_profile_scope_expire_and_degrade_to_memory(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    producer = ProcessRegistry()
    sid = "proc_safe_A-1.2"
    _finished(producer, sid)
    assert producer.wait(sid)["status"] == "exited"
    marker = home_a / "processes-consumed" / f"{sid}.consumed"
    assert marker.is_file()  # Behavioural RED on main, before testing the new helper.

    consumer = ProcessRegistry()
    for home, expected in ((home_a, True), (home_b, False), (home_a, True)):
        token = set_hermes_home_override(str(home))
        try:
            assert consumer.is_completion_consumed(sid) is expected
        finally:
            reset_hermes_home_override(token)
    assert not (home_b / "processes-consumed").exists()

    before = set((home_a / "processes-consumed").iterdir())
    for unsafe in ("../escape", "/absolute", "a/b", "a\\b", "a\x00b", ".", "..", "x" * 300, "é"):
        _finished(producer, unsafe)
        assert producer.wait(unsafe)["status"] == "exited"
        assert producer.is_completion_consumed(unsafe)  # The in-memory path still works.
        assert not consumer.is_completion_consumed(unsafe)
    assert set((home_a / "processes-consumed").iterdir()) == before

    stale = marker.with_name("proc_stale.consumed")
    stale.touch()
    old = time.time() - 8 * 86400
    os.utime(stale, (old, old))
    unrelated = marker.with_name("keep.txt")
    unrelated.touch()
    os.utime(unrelated, (old, old))
    # Ordinary session pruning must not erase recent consumption.
    recent = time.time() - 6 * 86400
    os.utime(marker, (recent, recent))
    producer._finished[sid].started_at = old
    with producer._lock:
        producer._prune_if_needed()
    assert sid not in producer._finished
    assert consumer.is_completion_consumed(sid)
    assert not stale.exists()
    assert unrelated.is_file()
    os.utime(marker, (old, old))
    consumer.recover_from_checkpoint()  # Startup sweeps even without a checkpoint.
    assert not marker.exists()
    assert not consumer.is_completion_consumed(sid)

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(blocked))
    _finished(producer, "proc_unwritable")
    assert producer.wait("proc_unwritable")["status"] == "exited"
    assert producer.is_completion_consumed("proc_unwritable")
    assert not ProcessRegistry().is_completion_consumed("proc_unwritable")
    assert blocked.read_text(encoding="utf-8") == "occupied"

    missing_profile = tmp_path / "root" / "profiles" / "deleted"
    monkeypatch.setenv("HERMES_HOME", str(missing_profile))
    _finished(producer, "proc_deleted_profile")
    assert producer.wait("proc_deleted_profile")["status"] == "exited"
    assert producer.is_completion_consumed("proc_deleted_profile")
    assert not missing_profile.exists()  # Never resurrect a deleted/missing named profile.
