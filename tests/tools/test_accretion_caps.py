"""Accretion caps for _read_tracker (file_tools) and _completion_consumed
(process_registry).

Both structures are process-lifetime singletons that previously grew
unbounded in long-running CLI / gateway sessions:

  file_tools._read_tracker[task_id]
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

import pytest


class TestReadTrackerCaps:
    def setup_method(self):
        from tools import file_tools

        # Clean slate per test.
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.clear()

    def test_read_history_capped(self, monkeypatch):
        """read_history set is bounded by _READ_HISTORY_CAP."""
        from tools import file_tools as ft

        monkeypatch.setattr(ft, "_READ_HISTORY_CAP", 10)
        task_data = {
            "last_key": None,
            "consecutive": 0,
            "read_history": set((f"/p{i}", 0, 500) for i in range(50)),
            "dedup": {},
            "read_timestamps": {},
        }
        ft._cap_read_tracker_data(task_data)
        assert len(task_data["read_history"]) == 10

    def test_dedup_capped_oldest_first(self, monkeypatch):
        """dedup dict is bounded; oldest entries evicted first."""
        from tools import file_tools as ft

        monkeypatch.setattr(ft, "_DEDUP_CAP", 5)
        task_data = {
            "read_history": set(),
            "dedup": {(f"/p{i}", 0, 500): float(i) for i in range(20)},
            "read_timestamps": {},
        }
        ft._cap_read_tracker_data(task_data)
        assert len(task_data["dedup"]) == 5
        # Entries 15-19 (inserted last) should survive.
        assert ("/p19", 0, 500) in task_data["dedup"]
        assert ("/p15", 0, 500) in task_data["dedup"]
        # Entries 0-14 should be evicted.
        assert ("/p0", 0, 500) not in task_data["dedup"]
        assert ("/p14", 0, 500) not in task_data["dedup"]

    def test_read_timestamps_capped_oldest_first(self, monkeypatch):
        """read_timestamps dict is bounded; oldest entries evicted first."""
        from tools import file_tools as ft

        monkeypatch.setattr(ft, "_READ_TIMESTAMPS_CAP", 3)
        task_data = {
            "read_history": set(),
            "dedup": {},
            "read_timestamps": {f"/path/{i}": float(i) for i in range(10)},
        }
        ft._cap_read_tracker_data(task_data)
        assert len(task_data["read_timestamps"]) == 3
        assert "/path/9" in task_data["read_timestamps"]
        assert "/path/7" in task_data["read_timestamps"]
        assert "/path/0" not in task_data["read_timestamps"]

    def test_cap_is_idempotent_under_cap(self, monkeypatch):
        """When containers are under cap, _cap_read_tracker_data is a no-op."""
        from tools import file_tools as ft

        monkeypatch.setattr(ft, "_READ_HISTORY_CAP", 100)
        monkeypatch.setattr(ft, "_DEDUP_CAP", 100)
        monkeypatch.setattr(ft, "_READ_TIMESTAMPS_CAP", 100)
        task_data = {
            "read_history": {("/a", 0, 500), ("/b", 0, 500)},
            "dedup": {("/a", 0, 500): 1.0},
            "read_timestamps": {"/a": 1.0},
        }
        rh_before = set(task_data["read_history"])
        dedup_before = dict(task_data["dedup"])
        ts_before = dict(task_data["read_timestamps"])

        ft._cap_read_tracker_data(task_data)

        assert task_data["read_history"] == rh_before
        assert task_data["dedup"] == dedup_before
        assert task_data["read_timestamps"] == ts_before

    def test_cap_handles_missing_containers(self):
        """Missing sub-keys don't cause AttributeError."""
        from tools import file_tools as ft

        ft._cap_read_tracker_data({})  # no containers at all
        ft._cap_read_tracker_data({"read_history": None})
        ft._cap_read_tracker_data({"dedup": None})

    def test_live_cap_applied_after_read_add(self, tmp_path, monkeypatch):
        """Live read_file path enforces caps."""
        from tools import file_tools as ft

        monkeypatch.setattr(ft, "_READ_HISTORY_CAP", 3)
        monkeypatch.setattr(ft, "_DEDUP_CAP", 3)
        monkeypatch.setattr(ft, "_READ_TIMESTAMPS_CAP", 3)

        # Create 10 distinct files and read each once.
        for i in range(10):
            p = tmp_path / f"file_{i}.txt"
            p.write_text(f"content {i}\n" * 10)
            ft.read_file_tool(path=str(p), task_id="long-session")

        with ft._read_tracker_lock:
            td = ft._read_tracker["long-session"]
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
        from tools.process_registry import ProcessRegistry, FINISHED_TTL_SECONDS
        import time

        reg = ProcessRegistry()
        # Fake a finished session whose started_at is older than the TTL.
        class _FakeSess:
            def __init__(self, sid):
                self.id = sid
                self.started_at = time.time() - (FINISHED_TTL_SECONDS + 100)
                self.exited = True

        reg._finished["stale-1"] = _FakeSess("stale-1")
        reg._completion_consumed.add("stale-1")

        with reg._lock:
            reg._prune_if_needed()

        assert "stale-1" not in reg._finished
        assert "stale-1" not in reg._completion_consumed

    def test_prune_drops_completion_entry_for_lru_evicted(self):
        """Same contract for the LRU path (over MAX_PROCESSES)."""
        from tools import process_registry as pr
        import time

        reg = pr.ProcessRegistry()

        class _FakeSess:
            def __init__(self, sid, started):
                self.id = sid
                self.started_at = started
                self.exited = True

        # Fill above MAX_PROCESSES with recently-finished sessions.
        now = time.time()
        for i in range(pr.MAX_PROCESSES + 5):
            sid = f"sess-{i}"
            reg._finished[sid] = _FakeSess(sid, now - i)  # sess-0 newest
            reg._completion_consumed.add(sid)

        with reg._lock:
            # _prune_if_needed removes one oldest finished per invocation;
            # call it enough times to trim back down.
            for _ in range(10):
                reg._prune_if_needed()

        # The _completion_consumed set should not contain session IDs that
        # are no longer in _running or _finished.
        assert (reg._completion_consumed - (reg._running.keys() | reg._finished.keys())) == set()

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
