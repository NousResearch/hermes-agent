import json
import os
import tempfile
import unittest
from pathlib import Path


class FrontierSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HERMES_HOME"] = str(Path(self.tmp.name) / "companyintel")
        os.environ["HERMES_PROFILE"] = "companyintel"

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, payload):
        from tools.companyintel_graph_tool import companyintel_graph
        return json.loads(companyintel_graph(payload))

    def test_init_seeds_durable_frontier_tasks_and_claim_is_atomic(self):
        init = self._call({"action": "init_run", "run_id": "run_frontier_001", "target_url": "https://example.com"})
        self.assertTrue(init["ok"], init)

        first = self._call({
            "action": "claim_frontier",
            "run_id": "run_frontier_001",
            "worker_id": "worker-a",
            "lease_seconds": 60,
        })
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["task"]["state"], "CLAIMED")
        self.assertEqual(first["task"]["worker_id"], "worker-a")
        self.assertEqual(first["task"]["attempt"], 1)

        second = self._call({
            "action": "claim_frontier",
            "run_id": "run_frontier_001",
            "worker_id": "worker-b",
            "lease_seconds": 60,
        })
        self.assertTrue(second["ok"], second)
        self.assertIsNotNone(second["task"])
        self.assertNotEqual(second["task"]["task_id"], first["task"]["task_id"])
        self.assertEqual(second["task"]["attempt"], 1)

    def test_complete_requires_current_lease_and_is_durable(self):
        self._call({"action": "init_run", "run_id": "run_frontier_002", "target_url": "https://example.com"})
        claimed = self._call({"action": "claim_frontier", "run_id": "run_frontier_002", "worker_id": "worker-a"})
        task = claimed["task"]

        wrong = self._call({
            "action": "complete_frontier", "run_id": "run_frontier_002",
            "task_id": task["task_id"], "worker_id": "worker-b", "lease_token": task["lease_token"],
        })
        self.assertFalse(wrong["ok"])

        done = self._call({
            "action": "complete_frontier", "run_id": "run_frontier_002",
            "task_id": task["task_id"], "worker_id": "worker-a", "lease_token": task["lease_token"],
        })
        self.assertTrue(done["ok"], done)
        status = self._call({"action": "frontier_status", "run_id": "run_frontier_002"})
        self.assertEqual(status["counts"]["COMPLETED"], 1)
        self.assertEqual(status["counts"]["OPEN"], 2)

    def test_failed_task_enters_retry_wait_then_reaches_failed_bound(self):
        self._call({"action": "init_run", "run_id": "run_frontier_003", "target_url": "https://example.com"})
        claimed = self._call({"action": "claim_frontier", "run_id": "run_frontier_003", "worker_id": "worker-a", "max_attempts": 1})
        task = claimed["task"]
        failed = self._call({
            "action": "fail_frontier", "run_id": "run_frontier_003", "task_id": task["task_id"],
            "worker_id": "worker-a", "lease_token": task["lease_token"], "error": "pivot unavailable",
            "retry_after_seconds": 0,
        })
        self.assertTrue(failed["ok"], failed)
        self.assertEqual(failed["state"], "FAILED")
        self.assertEqual(failed["attempt"], 1)

    def test_run_transitions_running_retry_wait_resumable_and_resume_is_durable(self):
        self._call({"action": "init_run", "run_id": "run_lifecycle_001", "target_url": "https://example.com"})
        claimed = self._call({"action": "claim_frontier", "run_id": "run_lifecycle_001", "worker_id": "worker-a", "max_attempts": 3})
        task = claimed["task"]
        failed = self._call({
            "action": "fail_frontier", "run_id": "run_lifecycle_001", "task_id": task["task_id"],
            "worker_id": "worker-a", "lease_token": task["lease_token"], "error": "upstream timeout",
            "retry_after_seconds": 0,
        })
        self.assertEqual(failed["state"], "RETRY_WAIT")
        waiting = self._call({"action": "summary", "run_id": "run_lifecycle_001"})
        self.assertEqual(waiting["status"], "RETRY_WAIT")
        self.assertEqual(waiting["retry_reason"], "upstream timeout")
        self.assertIsNotNone(waiting["retry_at"])

        ready = self._call({"action": "claim_frontier", "run_id": "run_lifecycle_001", "worker_id": "worker-b"})
        self.assertIsNone(ready["task"])
        self.assertEqual(ready["run_status"], "RESUMABLE")

        resumed = self._call({"action": "resume_frontier", "run_id": "run_lifecycle_001", "worker_id": "worker-b", "max_tasks": 1})
        self.assertTrue(resumed["resumed"], resumed)
        after = self._call({"action": "summary", "run_id": "run_lifecycle_001"})
        self.assertIn(after["status"], {"RUNNING", "RETRY_WAIT", "RESUMABLE", "PARTIAL"})
        self.assertGreaterEqual(after["resume_count"], 1)

    def test_checkpoint_resume_worker_survives_process_boundary_and_reclaims_same_task(self):
        self._call({"action": "init_run", "run_id": "run_checkpoint_001", "target_url": "https://example.com"})
        claimed = self._call({"action": "claim_frontier", "run_id": "run_checkpoint_001", "worker_id": "worker-a", "lease_seconds": 60})
        task = claimed["task"]
        checkpointed = self._call({
            "action": "checkpoint_worker", "run_id": "run_checkpoint_001", "task_id": task["task_id"],
            "worker_id": "worker-a", "lease_token": task["lease_token"], "phase": "fetching",
            "cursor": {"page": 2, "result_offset": 10},
        })
        self.assertTrue(checkpointed["ok"], checkpointed)
        self.assertEqual(checkpointed["cursor"]["page"], 2)

        import sqlite3
        db = Path(os.environ["HERMES_HOME"]) / "companyintel" / "runs" / "run_checkpoint_001" / "graph.sqlite3"
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE frontier_tasks SET lease_expires_at=0 WHERE task_id=?", (task["task_id"],))
            conn.commit()

        resumed = self._call({
            "action": "resume_worker", "run_id": "run_checkpoint_001", "worker_id": "worker-b",
            "task_id": task["task_id"], "lease_seconds": 60,
        })
        self.assertTrue(resumed["ok"], resumed)
        self.assertTrue(resumed["resumed_from_checkpoint"], resumed)
        self.assertEqual(resumed["task"]["task_id"], task["task_id"])
        self.assertEqual(resumed["task"]["attempt"], 2)
        self.assertEqual(resumed["checkpoint"]["cursor"]["result_offset"], 10)
        self.assertNotIn("lease_token", resumed["checkpoint"])

    def test_expired_lease_is_recovered_by_next_claim(self):
        self._call({"action": "init_run", "run_id": "run_frontier_004", "target_url": "https://example.com"})

        first = self._call({"action": "claim_frontier", "run_id": "run_frontier_004", "worker_id": "worker-a", "lease_seconds": 0})
        second = self._call({"action": "claim_frontier", "run_id": "run_frontier_004", "worker_id": "worker-b", "lease_seconds": 60})
        self.assertTrue(second["ok"], second)
        self.assertIsNotNone(second["task"])
        self.assertEqual(second["task"]["task_id"], first["task"]["task_id"])
        self.assertEqual(second["task"]["attempt"], 2)
        self.assertEqual(second["task"]["worker_id"], "worker-b")


if __name__ == "__main__":
    unittest.main()
