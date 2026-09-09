"""Maintenance cannot prune objects another checkpoint process has not published."""
from __future__ import annotations

import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Thread

from tools import checkpoint_manager as checkpoints


def test_active_snapshot_owns_store_until_its_ref_is_published(tmp_path, monkeypatch):
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    project = tmp_path / "project"
    project.mkdir()
    (project / "initial.txt").write_text("initial", encoding="utf-8")
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    assert manager.ensure_checkpoint(str(project), "initial")
    (project / "concurrent.txt").write_text("new unpublished bytes", encoding="utf-8")
    program = '''
import sys
from pathlib import Path
from tools import checkpoint_manager as c
c.CHECKPOINT_BASE = Path(sys.argv[1])
c._managed_git = lambda: None
run = c._run_git
def paused(args, *rest, **kwargs):
    result = run(args, *rest, **kwargs)
    if args[0] == "write-tree" and result[0]:
        print("tree-written", flush=True)
        sys.stdin.readline()
    return result
c._run_git = paused
assert c.CheckpointManager(enabled=True, max_total_size_mb=0).ensure_checkpoint(sys.argv[2], "concurrent")
'''
    env = {**os.environ, "PYTHONPATH": str(Path(checkpoints.__file__).resolve().parents[1])}
    child = subprocess.Popen([sys.executable, "-u", "-c", program, str(base), str(project)],
                             cwd=tmp_path, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, encoding="utf-8")
    ready = Queue()
    reader = Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True)
    reader.start()
    try:
        assert ready.get(timeout=20).strip() == "tree-written"
        result = checkpoints.prune_checkpoints(retention_days=0, checkpoint_base=base)
        assert result["errors"] > 0, "maintenance must refuse an active store writer"
        assert not checkpoints.clear_all(base)["deleted"]
    finally:
        output, error = child.communicate(input="publish\n", timeout=30)
        reader.join(timeout=5)
    assert child.returncode == 0, output + error
    assert [row["reason"] for row in manager.list_checkpoints(str(project))] == ["concurrent", "initial"]
