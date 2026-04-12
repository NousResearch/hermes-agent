import json

from tools.execution_work_orders import (
    cancel_execution_work_order,
    claim_next_due_execution_work_order,
    enqueue_execution_work_order,
    execution_work_order_counts,
    finish_execution_work_order,
    get_execution_work_order,
    query_execution_work_orders,
    reclaim_stale_execution_work_orders,
    resume_execution_work_order,
)


class TestExecutionWorkOrders:
    def test_enqueue_and_query(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)

        row = enqueue_execution_work_order(
            goal="Hash delegate_tool.py",
            command="cd /workspace && sha256sum tools/delegate_tool.py | awk '{print $1}'",
            now=100.0,
        )

        assert row["status"] == "queued"
        assert row["execution_path"] == "direct_terminal_work_order"
        assert row["schedule_display"] == "immediate"

        rows = query_execution_work_orders(limit=10)
        assert len(rows) == 1
        assert rows[0]["work_order_id"] == row["work_order_id"]
        assert rows[0]["goal"] == "Hash delegate_tool.py"

        counts = execution_work_order_counts(now=100.0)
        assert counts["due_count"] == 1
        assert counts["status_counts"]["queued"] == 1

        file_payload = json.loads((tmp_path / "artifacts" / "execution-work-orders" / f"{row['work_order_id']}.json").read_text(encoding="utf-8"))
        assert file_payload["command"].startswith("cd /workspace")

    def test_claim_and_finish_completed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        row = enqueue_execution_work_order(
            goal="Compile H007 files",
            command="cd /workspace && python -m py_compile tools/delegate_tool.py",
            now=10.0,
        )

        claimed = claim_next_due_execution_work_order(claim_owner="runner-a", claim_ttl_seconds=30, now=11.0)
        assert claimed is not None
        assert claimed["work_order_id"] == row["work_order_id"]
        assert claimed["status"] == "running"
        assert claimed["attempt_count"] == 1
        assert claimed["claim_owner"] == "runner-a"

        finished = finish_execution_work_order(
            row["work_order_id"],
            claim_token=claimed["claim_token"],
            now=12.5,
            result={
                "status": "completed",
                "summary": "OK",
                "exit_reason": "completed",
                "duration_seconds": 0.2,
                "execution_receipt_path": "/tmp/receipt-123.json",
                "worker_mode": "warm",
                "worker_task_id": "delegate-lease-1",
                "worker_runtime_id": "container-1",
                "worker_runtime_kind": "DockerEnvironment",
                "worker_runtime_reused": True,
            },
        )

        assert finished["status"] == "completed"
        current = get_execution_work_order(row["work_order_id"])
        assert current is not None
        assert current["status"] == "completed"
        assert current["last_receipt_id"] == "receipt-123"
        assert current["worker_runtime_reused"] is True
        assert current["claim_token"] is None

    def test_failed_attempt_auto_retries_then_fails_terminally(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        row = enqueue_execution_work_order(
            goal="Retryable task",
            command="false",
            max_attempts=2,
            retry_delay_seconds=0,
            now=20.0,
        )

        claimed1 = claim_next_due_execution_work_order(claim_owner="runner-a", claim_ttl_seconds=30, now=21.0)
        after_first = finish_execution_work_order(
            row["work_order_id"],
            claim_token=claimed1["claim_token"],
            now=21.5,
            result={
                "status": "failed",
                "error": "boom",
                "fallback_reason": "terminal_exit_code_1",
                "exit_reason": "terminal_exit_code_1",
                "duration_seconds": 0.1,
            },
        )
        assert after_first["status"] == "retry_scheduled"

        claimed2 = claim_next_due_execution_work_order(claim_owner="runner-b", claim_ttl_seconds=30, now=22.0)
        assert claimed2 is not None
        assert claimed2["attempt_count"] == 2

        after_second = finish_execution_work_order(
            row["work_order_id"],
            claim_token=claimed2["claim_token"],
            now=22.5,
            result={
                "status": "failed",
                "error": "still-boom",
                "fallback_reason": "terminal_exit_code_1",
                "exit_reason": "terminal_exit_code_1",
                "duration_seconds": 0.1,
            },
        )
        assert after_second["status"] == "failed"
        current = get_execution_work_order(row["work_order_id"])
        assert current["attempt_count"] == 2
        assert current["last_error"] == "still-boom"

    def test_reclaim_resume_and_cancel(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        row = enqueue_execution_work_order(
            goal="Eventually resume",
            command="echo hi",
            now=30.0,
        )

        claimed = claim_next_due_execution_work_order(claim_owner="runner-a", claim_ttl_seconds=5, now=31.0)
        assert claimed is not None

        reclaimed = reclaim_stale_execution_work_orders(now=40.0, limit=10)
        assert reclaimed["reclaimed_count"] == 1
        current = get_execution_work_order(row["work_order_id"])
        assert current["status"] == "queued"
        assert current["reclaimed_count"] == 1

        finished = claim_next_due_execution_work_order(claim_owner="runner-b", claim_ttl_seconds=5, now=41.0)
        finish_execution_work_order(
            row["work_order_id"],
            claim_token=finished["claim_token"],
            now=41.5,
            result={
                "status": "failed",
                "error": "fail-once",
                "fallback_reason": "terminal_exit_code_1",
                "exit_reason": "terminal_exit_code_1",
                "duration_seconds": 0.1,
            },
        )

        resumed = resume_execution_work_order(row["work_order_id"], delay_seconds=15, now=50.0)
        assert resumed["status"] == "queued"
        assert resumed["scheduled_for"] == 65.0

        cancelled = cancel_execution_work_order(row["work_order_id"], now=51.0)
        assert cancelled["status"] == "cancelled"
