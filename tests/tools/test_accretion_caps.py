"""Accretion caps for _read_tracker (file_tools) and _completion_consumed
(process_registry).

Both structures are process-lifetime singletons that previously grew
unbounded in long-running CLI / gateway sessions:

  file_tools_read_tracking._read_tracker[task_id]
    ├─ read_history (set)      — one entry per unique (path, offset, limit)
    ├─ dedup (dict)            — one entry per unique (path, offset, limit)
    └─ read_timestamps (dict)  — one entry per unique resolved path
  process_registry._completion_consumed (set) — one entry per session_id
    ever polled / waited / logged

None of these were ever trimmed.  A 10k-read CLI session accumulated
roughly 1.5MB of tracker state; a gateway with high background-process
churn accumulated ~20B per session_id until the process exited.

These tests pin the new caps + prune hooks.
"""


class TestReadTrackerCaps:
    def setup_method(self):
        from tools import file_tools_read_tracking as rt

        # Clean slate per test.
        with rt._read_tracker_lock:
            rt._read_tracker.clear()



    def test_live_cap_applied_after_read_add(self, tmp_path, monkeypatch):
        """Live read_file path enforces caps."""
        from tools import file_tools as ft
        from tools import file_tools_read_tracking as rt

        monkeypatch.setattr(rt, "_READ_HISTORY_CAP", 3)
        monkeypatch.setattr(rt, "_DEDUP_CAP", 3)
        monkeypatch.setattr(rt, "_READ_TIMESTAMPS_CAP", 3)

        # Create 10 distinct files and read each once.
        for i in range(10):
            p = tmp_path / f"file_{i}.txt"
            p.write_text(f"content {i}\n" * 10)
            ft.read_file_tool(path=str(p), task_id="long-session")

        with rt._read_tracker_lock:
            td = rt._read_tracker["long-session"]
            assert len(td["read_history"]) <= 3
            assert len(td["dedup"]) <= 3
            # read_timestamps is populated lazily (via setdefault) only
            # when os.path.getmtime() succeeds. On some CI filesystems
            # that stat can race with file creation — skip rather than
            # hard-error if the dict hasn't been created yet.
            assert len(td.get("read_timestamps", {})) <= 3


class TestCompletionConsumedPrune:
    def test_prune_drops_completion_entry_with_expired_session(self):
        """When a finished session is pruned, _completion_consumed is
        cleared for the same session_id."""
        from tools.process_registry import ProcessRegistry, ProcessSession, FINISHED_TTL_SECONDS
        import time

        reg = ProcessRegistry()
        # Use the real inert record so pruning exercises the handle-cleanup contract.
        reg._finished["stale-1"] = ProcessSession(
            id="stale-1",
            command="finished-test-command",
            started_at=time.time() - (FINISHED_TTL_SECONDS + 100),
            exited=True,
        )
        reg._completion_consumed.add("stale-1")

        with reg._lock:
            reg._prune_if_needed()

        assert "stale-1" not in reg._finished
        assert "stale-1" not in reg._completion_consumed


    def test_prune_clears_dangling_completion_entries(self):
        """Stale entries in _completion_consumed without a backing session
        record are cleared out (belt-and-suspenders invariant)."""
        from tools.process_registry import ProcessRegistry

        reg = ProcessRegistry()
        # Add a dangling entry that was never in _running or _finished.
        reg._completion_consumed.add("dangling-never-tracked")

        with reg._lock:
            reg._prune_if_needed()

        assert "dangling-never-tracked" not in reg._completion_consumed
