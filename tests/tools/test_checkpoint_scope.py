"""CheckpointManager snapshot scope: turn (default) vs task (#68877).

scope="turn" resets the per-directory dedup on every agent iteration, so each
turn can take one snapshot (historical behavior). scope="task" makes new_turn()
a no-op — the dedup persists across the task's iterations so only the first
file mutation snapshots the pre-task baseline — while new_task() re-arms it at
the next task boundary. These tests exercise the dedup bookkeeping directly
(no git needed) by inspecting ``_checkpointed_dirs``.
"""

from tools.checkpoint_manager import CheckpointManager


class TestScopeNormalization:
    def test_default_scope_is_turn(self):
        assert CheckpointManager().scope == "turn"

    def test_task_scope_honored(self):
        assert CheckpointManager(scope="task").scope == "task"

    def test_scope_case_insensitive_and_trimmed(self):
        assert CheckpointManager(scope="  TASK ").scope == "task"

    def test_unknown_scope_degrades_to_turn(self):
        assert CheckpointManager(scope="bogus").scope == "turn"
        assert CheckpointManager(scope="").scope == "turn"


class TestTurnScopeDedup:
    def test_new_turn_clears_dedup_each_iteration(self):
        mgr = CheckpointManager(enabled=True, scope="turn")
        # Simulate a snapshot having been taken this turn.
        mgr._checkpointed_dirs.add("/work")
        mgr.new_turn()
        # Cleared → the next iteration is free to snapshot again.
        assert "/work" not in mgr._checkpointed_dirs


class TestTaskScopeDedup:
    def test_new_turn_is_noop_in_task_scope(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        mgr._checkpointed_dirs.add("/work")
        mgr.new_turn()
        # Persisted → later turns in the same task will skip re-snapshotting.
        assert "/work" in mgr._checkpointed_dirs

    def test_new_task_clears_even_in_task_scope(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        mgr._checkpointed_dirs.add("/work")
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs

    def test_new_task_clears_in_turn_scope_too(self):
        mgr = CheckpointManager(enabled=True, scope="turn")
        mgr._checkpointed_dirs.add("/work")
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs


class TestEndToEndDedupSemantics:
    """Model the loop's calls (new_task once, new_turn per iteration) and
    assert how many times a directory would be eligible for a snapshot."""

    def _eligible_count(self, scope, iterations):
        """Count how many iterations would take a snapshot: a dir is eligible
        when it is NOT already in the dedup set. Mirrors ensure_checkpoint's
        ``if abs_dir in self._checkpointed_dirs: return False`` gate."""
        mgr = CheckpointManager(enabled=True, scope=scope)
        mgr.new_task()  # task boundary, before the loop
        took = 0
        for _ in range(iterations):
            mgr.new_turn()  # start of each agent iteration
            if "/work" not in mgr._checkpointed_dirs:
                took += 1
                mgr._checkpointed_dirs.add("/work")  # ensure_checkpoint records it
        return took

    def test_turn_scope_snapshots_every_iteration(self):
        assert self._eligible_count("turn", iterations=5) == 5

    def test_task_scope_snapshots_once_per_task(self):
        assert self._eligible_count("task", iterations=5) == 1

    def test_task_scope_rearms_on_next_task(self):
        mgr = CheckpointManager(enabled=True, scope="task")
        # Task 1
        mgr.new_task()
        mgr.new_turn()
        assert "/work" not in mgr._checkpointed_dirs
        mgr._checkpointed_dirs.add("/work")
        mgr.new_turn()
        assert "/work" in mgr._checkpointed_dirs  # no second snapshot in task 1
        # Task 2 — fresh baseline
        mgr.new_task()
        assert "/work" not in mgr._checkpointed_dirs
