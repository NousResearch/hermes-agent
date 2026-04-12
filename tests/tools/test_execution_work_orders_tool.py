import json
from types import SimpleNamespace

from cron.jobs import create_job, get_job
from tools.execution_work_orders import enqueue_execution_work_order, get_execution_work_order
from tools.execution_work_orders_tool import (
    EXECUTION_WORK_ORDERS_SCHEMA,
    RUNNER_JOB_NAME,
    RUNNER_PROMPT_MARKER,
    execution_work_orders_tool,
)


class TestExecutionWorkOrdersTool:
    def test_schema_lists_actions(self):
        action = EXECUTION_WORK_ORDERS_SCHEMA["parameters"]["properties"]["action"]
        assert action["enum"] == [
            "list",
            "query",
            "enqueue",
            "run_due",
            "reclaim_stale",
            "retry",
            "resume",
            "cancel",
            "runner_status",
            "install_runner",
            "remove_runner",
        ]

    def test_enqueue_and_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)

        payload = json.loads(
            execution_work_orders_tool(
                action="enqueue",
                goal="Count lines",
                command="cd /workspace && wc -l tools/delegate_tool.py | awk '{print $1}'",
                parent_agent=SimpleNamespace(session_id="parent-1", enabled_toolsets=["delegation", "terminal"]),
            )
        )
        assert payload["work_order"]["goal"] == "Count lines"
        work_order_id = payload["work_order"]["work_order_id"]

        listed = json.loads(execution_work_orders_tool(action="list", limit=5))
        assert listed["count"] == 1
        assert listed["work_orders"][0]["work_order_id"] == work_order_id
        assert listed["due_count"] == 1

    def test_run_due_executes_and_updates_work_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        row = enqueue_execution_work_order(goal="Compile", command="echo OK", now=1.0)

        monkeypatch.setattr(
            "tools.execution_work_orders_tool.delegate_task",
            lambda **kwargs: json.dumps(
                {
                    "results": [
                        {
                            "status": "completed",
                            "summary": "OK",
                            "exit_reason": "completed",
                            "duration_seconds": 0.2,
                            "execution_receipt_path": "/tmp/work-order-receipt.json",
                            "worker_mode": "warm",
                            "worker_task_id": "delegate-lease-xyz",
                            "worker_runtime_id": "container-xyz",
                            "worker_runtime_kind": "DockerEnvironment",
                            "worker_runtime_reused": True,
                        }
                    ]
                }
            ),
        )
        parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["delegation", "terminal"])
        payload = json.loads(execution_work_orders_tool(action="run_due", limit=5, parent_agent=parent))
        assert payload["executed_count"] == 1
        assert payload["statuses"] == {"completed": 1}

        current = get_execution_work_order(row["work_order_id"])
        assert current is not None
        assert current["status"] == "completed"
        assert current["last_receipt_id"] == "work-order-receipt"
        assert current["worker_runtime_reused"] is True

    def test_install_runner_creates_cron_job(self, tmp_path, monkeypatch):
        cron_dir = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)

        payload = json.loads(
            execution_work_orders_tool(
                action="install_runner",
                schedule="every 2h",
                limit=7,
                reclaim_limit=9,
                claim_ttl_seconds=600,
                model="gpt-5.4",
                provider="openai-codex",
            )
        )

        assert payload["installed"] is True
        assert payload["created"] is True
        job = payload["job"]
        assert job["name"] == RUNNER_JOB_NAME
        assert job["schedule"] == "every 120m"
        assert job["config"]["run_limit"] == 7
        assert job["config"]["reclaim_limit"] == 9
        assert job["config"]["claim_ttl_seconds"] == 600.0

        status = json.loads(execution_work_orders_tool(action="runner_status"))
        assert status["installed"] is True
        assert status["installed_count"] == 1
        assert status["jobs"][0]["job_id"] == job["job_id"]

        jobs_payload = json.loads((cron_dir / "jobs.json").read_text(encoding="utf-8"))
        assert RUNNER_PROMPT_MARKER in jobs_payload["jobs"][0]["prompt"]

    def test_same_name_unrelated_job_survives_runner_surface(self, tmp_path, monkeypatch):
        cron_dir = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)

        unrelated = create_job(
            prompt="plain unrelated cron job",
            schedule="every 4h",
            name=RUNNER_JOB_NAME,
            deliver="local",
        )

        status_before = json.loads(execution_work_orders_tool(action="runner_status"))
        assert status_before["installed"] is False

        removed_before = json.loads(execution_work_orders_tool(action="remove_runner"))
        assert removed_before["deleted_count"] == 0
        assert get_job(unrelated["id"]) is not None
