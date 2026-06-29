import json
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job_immediate
from tools.cronjob_tools import cronjob


_JOB = {
    "id": "job-run-1",
    "name": "manual run",
    "prompt": "hi",
    "enabled": False,
    "state": "paused",
    "paused_at": "2026-01-01T00:00:00+00:00",
    "paused_reason": "test pause",
    "next_run_at": "2026-01-02T00:00:00+00:00",
}


class _ImmediateExecutor:
    def submit(self, fn):
        fn()
        return MagicMock()


class TestCronjobRunImmediate:
    def test_run_action_dispatches_immediately(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", side_effect=[dict(_JOB), dict(_JOB)]),
            patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB)) as m_trigger,
            patch("cron.scheduler._resolve_live_delivery_context", return_value=(None, None)),
            patch("cron.scheduler._job_has_required_live_delivery_context", return_value=True),
            patch("cron.scheduler._job_requires_live_delivery_context", return_value=False),
            patch("cron.scheduler.run_job_immediate", return_value=(True, None)) as m_run_now,
        ):
            out = json.loads(cronjob(action="run", job_id=_JOB["id"]))

        assert out["success"] is True
        assert out["dispatched"] is True
        m_trigger.assert_called_once_with(_JOB["id"])
        m_run_now.assert_called_once()
        assert m_run_now.call_args.kwargs["schedule_snapshot"] == {
            "enabled": False,
            "state": "paused",
            "paused_at": "2026-01-01T00:00:00+00:00",
            "paused_reason": "test pause",
            "next_run_at": "2026-01-02T00:00:00+00:00",
        }

    def test_run_action_queues_for_next_tick_when_live_context_is_missing(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", side_effect=[dict(_JOB), dict(_JOB)]),
            patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB)),
            patch("cron.scheduler._resolve_live_delivery_context", return_value=(None, None)),
            patch("cron.scheduler._job_has_required_live_delivery_context", return_value=False),
            patch("cron.scheduler._job_requires_live_delivery_context", return_value=True),
            patch("cron.scheduler._queue_manual_run_for_tick") as m_queue,
            patch("cron.scheduler.run_job_immediate") as m_run_now,
        ):
            out = json.loads(cronjob(action="run", job_id=_JOB["id"]))

        assert out["success"] is True
        assert out["dispatched"] is False
        assert "queued for the next gateway tick" in out["note"]
        m_queue.assert_called_once()
        m_run_now.assert_not_called()

    def test_run_action_restores_schedule_after_dispatch_exception(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", side_effect=[dict(_JOB), dict(_JOB)]),
            patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB)),
            patch("cron.scheduler._resolve_live_delivery_context", return_value=(None, None)),
            patch("cron.scheduler._job_has_required_live_delivery_context", return_value=True),
            patch("cron.scheduler._job_requires_live_delivery_context", return_value=False),
            patch("cron.scheduler.run_job_immediate", side_effect=RuntimeError("submit failed")),
            patch("tools.cronjob_tools.update_job") as m_update,
        ):
            out = json.loads(cronjob(action="run", job_id=_JOB["id"]))

        assert out["success"] is True
        assert out["dispatched"] is False
        assert out["note"] == "submit failed"
        m_update.assert_called_once_with(
            _JOB["id"],
            {
                "manual_run_schedule_snapshot": None,
                "manual_run_gateway_only": None,
                "enabled": False,
                "state": "paused",
                "paused_at": "2026-01-01T00:00:00+00:00",
                "paused_reason": "test pause",
                "next_run_at": "2026-01-02T00:00:00+00:00",
            },
        )

    def test_scheduler_dispatch_restores_snapshot_after_success(self):
        with (
            patch("cron.jobs.resolve_job_ref", return_value=dict(_JOB)),
            patch("cron.scheduler._claim_running_job", return_value=object()),
            patch("cron.scheduler._job_requires_sequential_pool", return_value=False),
            patch("cron.scheduler._get_parallel_pool", return_value=_ImmediateExecutor()),
            patch("cron.scheduler._resolve_max_parallel_workers", return_value=None),
            patch("cron.scheduler._resolve_live_delivery_context", return_value=(None, None)),
            patch("cron.scheduler.run_job", return_value=(True, "output", "response", None)),
            patch("cron.jobs.save_job_output"),
            patch("cron.scheduler._deliver_result", return_value=None),
            patch("cron.scheduler.mark_job_run") as m_mark,
            patch("cron.scheduler._restore_manual_run_schedule") as m_restore,
            patch("cron.scheduler._release_running_job"),
            patch("cron.scheduler._sweep_mcp_orphans"),
        ):
            dispatched, error = run_job_immediate(
                _JOB["id"],
                schedule_snapshot={"enabled": False, "next_run_at": "2026-01-02T00:00:00+00:00"},
            )

        assert dispatched is True
        assert error is None
        m_mark.assert_called_once()
        m_restore.assert_called_once_with(
            _JOB["id"],
            {"enabled": False, "next_run_at": "2026-01-02T00:00:00+00:00"},
        )

    def test_scheduler_dispatch_returns_false_when_job_is_already_running(self):
        with (
            patch("cron.jobs.resolve_job_ref", return_value=dict(_JOB)),
            patch("cron.scheduler._claim_running_job", return_value=None),
            patch("cron.scheduler._restore_manual_run_schedule") as m_restore,
        ):
            dispatched, error = run_job_immediate(
                _JOB["id"],
                schedule_snapshot={"enabled": False},
            )

        assert dispatched is False
        assert "already running" in error
        m_restore.assert_called_once_with(_JOB["id"], {"enabled": False})
