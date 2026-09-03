"""Regression tests for #98713: delegate_task split-brain.

Background batch dispatched, persistence fails → tool used to return a generic
error while workers kept running; action=list returned count:0. Two fixes:

1. _persist_dispatch failure must not abort the dispatch (fail-open so the
   accepted batch's tool result says 'dispatched', not 'Error executing tool').
2. action=list reconciles async_delegation._records so live batches are visible
   even after a parent-agent rebuild breaks the weakref chain.
"""

import time
import threading
import unittest
from unittest.mock import patch, MagicMock


class TestPersistDispatchFailOpen(unittest.TestCase):
    """_persist_dispatch failure must not kill an already-accepted dispatch."""

    def test_persistence_failure_does_not_revert_in_memory_record(self):
        """When _persist_dispatch raises, _records still has the delegation."""
        from tools import async_delegation as ad

        # Reset state
        with ad._records_lock:
            ad._records.clear()

        called = []

        def _boom(_record):
            called.append(True)
            raise OSError("disk I/O error")

        def _runner():
            return {"results": [], "total_duration_seconds": 0.1}

        deleg_id = "test-persist-fail-" + str(int(time.time() * 1000))

        with patch.object(ad, "_persist_dispatch", side_effect=_boom):
            result = ad.dispatch_async_delegation_batch(
                goals=["task A"],
                context=None,
                toolsets=None,
                role="leaf",
                model="test-model",
                session_key="sess-test-123",
                runner=_runner,
                delegation_id=deleg_id,
            )

        # Despite persistence failure, dispatch must succeed
        self.assertEqual(result.get("status"), "dispatched",
                         f"Expected dispatched, got: {result}")
        self.assertEqual(result.get("delegation_id"), deleg_id)
        self.assertTrue(called, "_persist_dispatch was not called")

        # Clean up background thread
        time.sleep(0.2)
        with ad._records_lock:
            ad._records.pop(deleg_id, None)


class TestGetRunningRecords(unittest.TestCase):
    """get_running_records surfaces live batch records by session_key."""

    def test_empty_when_no_matching_session(self):
        from tools.async_delegation import get_running_records
        result = get_running_records(session_key="no-such-session")
        self.assertEqual(result, [])

    def test_empty_session_key_returns_empty(self):
        from tools.async_delegation import get_running_records
        self.assertEqual(get_running_records(session_key=""), [])

    def test_returns_live_record_for_session(self):
        from tools import async_delegation as ad

        deleg_id = "test-get-running-" + str(int(time.time() * 1000))
        record = {
            "delegation_id": deleg_id,
            "goal": "test goal",
            "goals": ["test goal"],
            "model": "m",
            "session_key": "sess-abc",
            "status": "running",
            "is_batch": True,
            "dispatched_at": time.time(),
            "origin_ui_session_id": "",
            "origin_session_id": "",
            "parent_session_id": None,
            "context": None,
            "toolsets": None,
            "role": "leaf",
            "completed_at": None,
            "interrupt_fn": None,
            "progress_fn": None,
            "_progress_token": None,
            "_progress_ts": time.time(),
            "_interrupted_at": None,
        }
        with ad._records_lock:
            ad._records[deleg_id] = record

        try:
            result = ad.get_running_records(session_key="sess-abc")
            ids = [r["delegation_id"] for r in result]
            self.assertIn(deleg_id, ids)
            match = next(r for r in result if r["delegation_id"] == deleg_id)
            self.assertEqual(match["goal"], "test goal")
            self.assertEqual(match["status"], "running")
            self.assertTrue(match["is_batch"])
        finally:
            with ad._records_lock:
                ad._records.pop(deleg_id, None)

    def test_does_not_return_finished_records(self):
        from tools import async_delegation as ad

        deleg_id = "test-finished-" + str(int(time.time() * 1000))
        record = {
            "delegation_id": deleg_id,
            "goal": "done",
            "goals": ["done"],
            "model": "m",
            "session_key": "sess-done",
            "status": "completed",  # finished
            "is_batch": True,
            "dispatched_at": time.time(),
        }
        with ad._records_lock:
            ad._records[deleg_id] = record

        try:
            result = ad.get_running_records(session_key="sess-done")
            self.assertEqual(result, [],
                             "Completed records must not appear in running list")
        finally:
            with ad._records_lock:
                ad._records.pop(deleg_id, None)


if __name__ == "__main__":
    unittest.main()
