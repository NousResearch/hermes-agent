"""Tests for the checkpoint store's dangling-ref recovery and honest size-cap reporting.

Background: a loose ref that shadows a valid packed-ref and points at a *missing* object
(typically left behind by a ``git gc`` whose reachability snapshot predates a concurrent
checkpoint's ``update-ref``) makes **every** later ``git gc`` fail with "does not point to
a valid object!".  Object reclamation stops, the store grows past its size cap, and the old
``_shrink_store_to_cap`` returned ``True`` regardless — so a 27 GB store over a 500 MB cap
reported successful maintenance on every single write tool call (hot self-spin).

Every test runs against real git in an isolated tmp store.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from types import SimpleNamespace

import pytest

from tools.checkpoint_manager import (
    CheckpointManager,
    _cap_retry_cooling_down,
    _candidate_refs,
    _dir_size_bytes,
    _gc_store,
    _missing_tips,
    _project_hash,
    _ref_name,
    _repair_unresolvable_refs,
    _run_git,
    _shrink_store_to_cap,
    _store_lock,
    _store_path,
    repair_store,
    store_status,
)

GIT_GC = ["gc", "--prune=now", "--quiet"]


def _incompressible(size: int = 800_000) -> str:
    """Random base64 payload: git's loose-object zlib cannot shrink it away, so sizes and
    reclaiming behave like real data instead of collapsing to a few KB."""
    return base64.b64encode(os.urandom(size // 2)).decode("ascii")


@pytest.fixture()
def env(tmp_path, monkeypatch) -> SimpleNamespace:
    """A real, isolated checkpoint store with one project and one snapshot."""
    base = tmp_path / "checkpoints"
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    (work_dir / "a.txt").write_text("hello\n", encoding="utf-8")
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
    mgr = CheckpointManager(enabled=True, max_snapshots=50, max_total_size_mb=500)
    assert mgr.ensure_checkpoint(str(work_dir), "first") is True
    store = _store_path(base)
    return SimpleNamespace(base=base, work_dir=work_dir, store=store, mgr=mgr,
                           ref=_ref_name(_project_hash(str(work_dir))))


def _seed_dangling_shadow(env: SimpleNamespace) -> tuple[str, str]:
    """Leave ``env.ref`` as a loose ref pointing at a pruned object, shadowing a valid packed-ref.

    Mirrors the production corruption: the ref is packed; a writer commits a new object and
    moves the ref to it; a gc whose reachability snapshot predates that update prunes the
    object.  Returns ``(valid_packed_tip, pruned_sha)``.
    """
    store, wd, ref = env.store, str(env.work_dir), env.ref
    assert _run_git(["pack-refs", "--all"], store, wd)[0]
    packed_tip = _run_git(["rev-parse", ref], store, wd)[1]
    tree = _run_git(["rev-parse", f"{packed_tip}^{{tree}}"], store, wd)[1]
    pruned = _run_git(["commit-tree", tree, "-p", packed_tip, "-m", "raced", "--no-gpg-sign"], store, wd)[1]
    assert _run_git(["update-ref", ref, pruned], store, wd)[0]

    stashed = env.base / "shadow.stash"
    (store / ref).rename(stashed)          # gc's snapshot: the ref does not exist yet
    assert _run_git(GIT_GC, store, wd)[0]
    stashed.rename(store / ref)            # ... then the writer's update-ref lands
    return packed_tip, pruned


def _seed_dead_ref(env: SimpleNamespace, ref: str) -> None:
    """A loose ref pointing at a well-formed sha that was never written."""
    path = env.store / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("d" * 40 + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The pathology itself
# ---------------------------------------------------------------------------

class TestDanglingRefPathology:
    def test_dangling_shadow_ref_makes_gc_fail(self, env):
        _seed_dangling_shadow(env)
        ok, _, err = _run_git(GIT_GC, env.store, str(env.work_dir))
        assert ok is False, "baseline: a dangling loose ref must break git gc"
        assert "does not point to a valid object" in err

    def test_cleanup_paths_cannot_converge_on_a_dangling_ref(self, env):
        """The old cleanup loop could not make progress, yet reported success."""
        _, pruned = _seed_dangling_shadow(env)
        assert _missing_tips(env.store, str(env.work_dir), [env.ref]) == [(env.ref, pruned)]
        # Nothing droppable: the ref's tip does not resolve, so its commit count is 0.
        ok, _, _ = _run_git(["rev-list", "--count", env.ref], env.store, str(env.work_dir))
        assert ok is False

    def test_store_status_surfaces_unresolvable_refs(self, env):
        _seed_dangling_shadow(env)
        info = store_status(env.base)
        assert info["unresolvable_refs"] == [env.ref]


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

class TestRepair:
    def test_shadowed_ref_is_repaired_and_history_kept(self, env):
        packed_tip, pruned = _seed_dangling_shadow(env)

        result = repair_store(env.base)

        assert result["repaired"] == 1 and result["deleted"] == 0 and result["errors"] == 0
        assert not (env.store / env.ref).exists(), "the unresolvable loose shadow must be gone"
        assert _run_git(["rev-parse", env.ref], env.store, str(env.work_dir))[1] == packed_tip, \
            "the valid packed-ref value (real history) must take over"
        assert _run_git(GIT_GC, env.store, str(env.work_dir))[0] is True, "gc must work again"
        assert store_status(env.base)["unresolvable_refs"] == []

    def test_ref_without_a_valid_twin_is_dropped(self, env):
        dead = "refs/hermes/deadbeefcafe0001"
        _seed_dead_ref(env, dead)

        result = repair_store(env.base)

        assert result["deleted"] == 1 and result["repaired"] == 0
        assert not (env.store / dead).exists()
        assert dead not in _candidate_refs(env.store)
        assert _run_git(GIT_GC, env.store, str(env.work_dir))[0] is True

    def test_packed_only_ref_is_removed_from_packed_refs(self, env):
        """A dead ref that exists only in packed-refs must be dropped there too."""
        packed_only = "refs/hermes/feedface00000001"
        assert _run_git(["pack-refs", "--all"], env.store, str(env.work_dir))[0]
        packed_file = env.store / "packed-refs"
        packed_file.write_text(packed_file.read_text(encoding="utf-8") + f"{'e' * 40} {packed_only}\n",
                              encoding="utf-8")

        result = repair_store(env.base)

        assert result["deleted"] == 1
        assert packed_only not in packed_file.read_text(encoding="utf-8")
        assert packed_only not in _candidate_refs(env.store)

    def test_repair_is_a_noop_on_a_healthy_store(self, env):
        result = repair_store(env.base)
        assert result == {"scanned": result["scanned"], "repaired": 0, "deleted": 0, "errors": 0}
        assert result["scanned"] >= 1

    def test_repair_store_on_a_missing_base_is_inert(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", tmp_path / "nope")
        assert repair_store() == {"scanned": 0, "repaired": 0, "deleted": 0, "errors": 0}


# ---------------------------------------------------------------------------
# gc-side recovery
# ---------------------------------------------------------------------------

class TestGcStore:
    def test_gc_store_heals_a_dangling_ref_and_succeeds(self, env, caplog):
        packed_tip, _ = _seed_dangling_shadow(env)

        with caplog.at_level(logging.WARNING, logger="tools.checkpoint_manager"):
            assert _gc_store(env.store, str(env.work_dir)) is True

        assert any("gc failed" in rec.getMessage() for rec in caplog.records), \
            "a gc failure must be visible below error level"
        assert _run_git(GIT_GC, env.store, str(env.work_dir))[0] is True
        assert _run_git(["rev-parse", env.ref], env.store, str(env.work_dir))[1] == packed_tip

    def test_gc_store_skips_while_a_writer_holds_the_lock(self, env, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager._GC_LOCK_WAIT_S", 0.2)
        with _store_lock(env.store, exclusive=False, blocking=True, timeout=0.0):
            assert _gc_store(env.store, str(env.work_dir)) is False, \
                "a checkpoint in flight means 'skip maintenance', never 'race it'"


# ---------------------------------------------------------------------------
# Honest size-cap reporting
# ---------------------------------------------------------------------------

class TestSizeCapReporting:
    def test_shrink_reports_failure_instead_of_silent_success(self, env):
        """The old code returned True while the store stayed over cap — the hot-loop bug."""
        assert _shrink_store_to_cap(env.store, str(env.work_dir), 1) is False

    def test_shrink_reports_success_when_already_under_cap(self, env):
        cap = _dir_size_bytes(env.store) + 1
        assert _shrink_store_to_cap(env.store, str(env.work_dir), cap) is True

    def test_shrink_converges_by_dropping_and_reclaiming(self, tmp_path, monkeypatch):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        mgr = CheckpointManager(enabled=True, max_snapshots=50, max_total_size_mb=0)
        payload = _incompressible()
        for name in ("p1", "p2"):
            wd = tmp_path / name
            wd.mkdir()
            for i in range(4):
                (wd / "blob.bin").write_text(f"{name}-{i}:{payload}\n", encoding="utf-8")
                mgr.new_turn()
                assert mgr.ensure_checkpoint(str(wd), f"snap{i}") is True
        store = _store_path(base)
        size_before = _dir_size_bytes(store)
        cap = size_before // 2

        assert _shrink_store_to_cap(store, str(base), cap) is True
        assert _dir_size_bytes(store) <= cap
        assert _run_git(GIT_GC, store, str(base))[0] is True
        for name in ("p1", "p2"):
            ref = _ref_name(_project_hash(str(tmp_path / name)))
            remaining = int(_run_git(["rev-list", "--count", ref], store, str(base))[1])
            assert remaining >= 1, "a project must never be shrunk below one snapshot"

    def test_failed_cap_pass_backs_off_instead_of_spinning(self, tmp_path, monkeypatch, caplog):
        base = tmp_path / "checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
        wd = tmp_path / "project"
        wd.mkdir()
        (wd / "big.bin").write_text(_incompressible(4_000_000), encoding="utf-8")
        mgr = CheckpointManager(enabled=True, max_snapshots=50, max_total_size_mb=1)
        mgr.ensure_checkpoint(str(wd), "first")
        store = _store_path(base)
        assert _dir_size_bytes(store) > 1 * 1024 * 1024

        mgr.new_turn()
        (wd / "b.txt").write_text("change\n", encoding="utf-8")
        with caplog.at_level(logging.INFO, logger="tools.checkpoint_manager"):
            mgr.ensure_checkpoint(str(wd), "second")
        assert _cap_retry_cooling_down(store) is True, "an unconvergeable store must arm the backoff"
        assert any(r.levelno >= logging.WARNING and "still over its" in r.getMessage() for r in caplog.records)

        caplog.clear()
        mgr.new_turn()
        (wd / "c.txt").write_text("change\n", encoding="utf-8")
        with caplog.at_level(logging.INFO, logger="tools.checkpoint_manager"):
            mgr.ensure_checkpoint(str(wd), "third")
        assert not any("exceeded" in r.getMessage() for r in caplog.records), \
            "the second pass must not re-run maintenance inside the backoff window"


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

class TestWritePathSafety:
    def test_tip_validation_rolls_the_ref_back_instead_of_leaving_it_dangling(self, env, monkeypatch):
        before = _run_git(["rev-parse", env.ref], env.store, str(env.work_dir))[1]
        (env.work_dir / "a.txt").write_text("changed\n", encoding="utf-8")
        # Only the freshly committed tip is "missing" — a foreign gc pruned it in flight.
        monkeypatch.setattr("tools.checkpoint_manager._object_present",
                            lambda _store, _wd, sha: sha == before)

        env.mgr.new_turn()
        assert env.mgr.ensure_checkpoint(str(env.work_dir), "vanishing object") is False

        after = _run_git(["rev-parse", env.ref], env.store, str(env.work_dir))
        assert after[0] is True and after[1] == before, "the ref must not be left on a missing object"
        assert _missing_tips(env.store, str(env.work_dir), [env.ref]) == []

    def test_normal_checkpoint_still_moves_the_ref(self, env):
        before = _run_git(["rev-parse", env.ref], env.store, str(env.work_dir))[1]
        (env.work_dir / "a.txt").write_text("changed\n", encoding="utf-8")
        env.mgr.new_turn()
        assert env.mgr.ensure_checkpoint(str(env.work_dir), "second") is True
        after = _run_git(["rev-parse", env.ref], env.store, str(env.work_dir))[1]
        assert after != before
        assert _run_git(["cat-file", "-e", after], env.store, str(env.work_dir))[0] is True
        assert _run_git(GIT_GC, env.store, str(env.work_dir))[0] is True

    def test_repair_helper_is_idempotent(self, env):
        _seed_dangling_shadow(env)
        first = _repair_unresolvable_refs(env.store, str(env.work_dir))
        second = _repair_unresolvable_refs(env.store, str(env.work_dir))
        assert first["repaired"] == 1
        assert second["repaired"] == 0 and second["deleted"] == 0 and second["errors"] == 0


class TestStoreLock:
    def test_lock_wait_is_bounded_even_when_blocking(self, env):
        """flock() without LOCK_NB blocks inside the syscall and ignores the deadline, which
        would stall maintenance (or a checkpoint) forever; the wait must give up."""
        with _store_lock(env.store, exclusive=False, blocking=True, timeout=0.0):
            start = time.monotonic()
            with _store_lock(env.store, exclusive=True, blocking=True, timeout=0.2) as gc_lock:
                assert gc_lock is False, "an incompatible lock must not be granted"
            assert time.monotonic() - start < 5.0, "the bounded wait must respect its deadline"

    def test_exclusive_maintenance_lock_is_refused_while_a_writer_holds_it(self, env):
        with _store_lock(env.store, exclusive=False, blocking=True, timeout=0.0) as writer:
            assert writer is True
            with _store_lock(env.store, exclusive=True, blocking=False) as gc_lock:
                assert gc_lock is False
        with _store_lock(env.store, exclusive=True, blocking=False) as free:
            assert free is True

    def test_lock_file_lives_at_the_store_root_and_is_inert(self, env):
        with _store_lock(env.store, exclusive=True, blocking=False):
            pass
        assert (env.store / "maintenance.lock").exists()
        ok, out, _ = _run_git(["count-objects", "-v"], env.store, str(env.work_dir))
        assert ok is True and "count:" in out  # still a healthy git repo
        env.mgr.new_turn()
        (env.work_dir / "a.txt").write_text("after locking\n", encoding="utf-8")
        assert env.mgr.ensure_checkpoint(str(env.work_dir), "post-lock") is True
