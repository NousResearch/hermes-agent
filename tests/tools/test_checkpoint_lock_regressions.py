"""Regression coverage for two silent checkpoint failures (chantier homelab-checkpoints).

  1. store contention dropped a snapshot (``store_lock`` was non-blocking);
  2. a failed take never reached the user (``ensure_checkpoint``'s return ignored).
Each test fails on the pre-fix code and passes after it.

The orphaned per-project ``index.lock`` symptom is deliberately NOT covered
here: open PRs #13232 / #52887 and issue #107149 already own it.
"""

import contextlib
import threading

import pytest

from tools import checkpoint_manager as cm
from tools.checkpoint_pruning import PruneError, store_lock


@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "main.py").write_text("print('hello')\n")
    return d


@pytest.fixture()
def checkpoint_base(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture()
def mgr(work_dir, checkpoint_base, monkeypatch):
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", checkpoint_base)
    return cm.CheckpointManager(enabled=True, max_snapshots=50)


# ---------------------------------------------------------------------------
# 1. bounded wait: a concurrent holder delays the take instead of dropping it
# ---------------------------------------------------------------------------

def test_store_lock_wait_acquires_after_a_concurrent_holder(tmp_path):
    """``wait=False`` refuses at once; ``wait=True`` acquires once released."""
    base = tmp_path / "checkpoints"
    base.mkdir()
    release = threading.Event()
    held = threading.Event()

    def holder():
        with store_lock(base):
            held.set()
            release.wait(2.0)

    t = threading.Thread(target=holder)
    t.start()
    try:
        assert held.wait(2.0), "holder never took the lock"
        with pytest.raises(PruneError):
            with store_lock(base):  # non-blocking: refused while held
                pass
        releaser = threading.Timer(0.3, release.set)
        releaser.start()
        with store_lock(base, wait=True, timeout=5.0):  # waits, then acquires
            pass
    finally:
        release.set()
        t.join(2.0)


def test_ensure_checkpoint_contention_is_reported_then_retry_succeeds(mgr, work_dir, monkeypatch):
    """A busy store costs neither the snapshot (retried) nor the user's awareness."""
    import tools.checkpoint_pruning as cp

    state = {"n": 0}
    real_lock = cp.store_lock

    @contextlib.contextmanager
    def busy_once(base, *, wait=False, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise PruneError(f"checkpoint store is busy: {base}")
        with real_lock(base):
            yield

    monkeypatch.setattr(cp, "store_lock", busy_once)

    assert mgr.ensure_checkpoint(str(work_dir), "contended") is False
    assert mgr.consume_checkpoint_notice() is not None  # silence broken
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work_dir), "retried") is True


# ---------------------------------------------------------------------------
# 2. failure reaches the user, once, and only for real failures
# ---------------------------------------------------------------------------

def test_failed_take_queues_one_notice_naming_the_directory(mgr, work_dir, monkeypatch):
    import tools.checkpoint_pruning as pruner

    @contextlib.contextmanager
    def always_busy(base, *, wait=False, timeout=None):
        raise PruneError(f"checkpoint store is busy: {base}")

    monkeypatch.setattr(pruner, "store_lock", always_busy)

    assert mgr.ensure_checkpoint(str(work_dir), "blocked") is False
    notice = mgr.consume_checkpoint_notice()
    assert notice and str(work_dir) in notice
    # Drained: one warning, not a per-tool spam.
    assert mgr.consume_checkpoint_notice() is None


def test_successful_take_never_queues_a_notice(mgr, work_dir):
    assert mgr.ensure_checkpoint(str(work_dir), "ok") is True
    assert mgr.consume_checkpoint_notice() is None
