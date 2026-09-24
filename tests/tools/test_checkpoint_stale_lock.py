"""Regression: an abandoned ``<index>.lock`` permanently wedges a working directory.

A git process killed mid-``add`` leaves ``<GIT_INDEX_FILE>.lock`` behind. Git never clears a lock
it did not create, so every later checkpoint of that directory fails with

    fatal: Unable to create '.../indexes/<hash>.lock': File exists.

and the snapshot is silently dropped — checkpoints are best-effort, so nothing surfaces to the
user. The lock has no live owner, so it is safe to remove once it is unambiguously stale.
"""

import os
import time
from pathlib import Path

from tools.checkpoint_manager import CheckpointManager, _project_refs


def _manager() -> CheckpointManager:
    return CheckpointManager(enabled=True, max_snapshots=50, max_total_size_mb=500,
                             max_file_size_mb=10)


def _work_dir(tmp_path: Path) -> Path:
    work = tmp_path / "project"
    work.mkdir()
    (work / "main.py").write_text("print('hello')\n")
    return work


def _index_lock(work: Path) -> Path:
    return Path(str(_project_refs(str(work)).index_file) + ".lock")


def _touch_work(work: Path, text: str) -> None:
    # The content must differ, or _take short-circuits before ever running `git add`.
    (work / "main.py").write_text(text)


def test_checkpoint_recovers_from_a_stale_index_lock(tmp_path):
    work = _work_dir(tmp_path)
    assert _manager().ensure_checkpoint(str(work), "first") is True

    lock = _index_lock(work)
    lock.write_text("")
    old = time.time() - 86_400
    os.utime(lock, (old, old))
    _touch_work(work, "print('hello')\nprint('again')\n")

    assert _manager().ensure_checkpoint(str(work), "after stale lock") is True
    assert not lock.exists()


def test_a_live_index_lock_is_not_broken(tmp_path):
    work = _work_dir(tmp_path)
    assert _manager().ensure_checkpoint(str(work), "first") is True

    # A fresh lock may belong to a concurrent `git add` — it must be left alone.
    lock = _index_lock(work)
    lock.write_text("")
    _touch_work(work, "print('hello')\nprint('concurrent')\n")

    assert _manager().ensure_checkpoint(str(work), "concurrent") is False
    assert lock.exists()
