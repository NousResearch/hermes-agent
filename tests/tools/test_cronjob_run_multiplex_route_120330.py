"""CLI `cron run` for a multiplex-routed profile must not burn an agent turn
in-process and then record `delivery_failed` (#120330).

A satellite profile with no `platforms:` block of its own delivers through the
primary gateway's `profile_routes`. The CLI process holds no live adapter, so an
in-process manual run can never deliver there. The run must divert BEFORE the
agent turn: mark the job due for the multiplexer's ticker when one serves this
profile, or fail fast when none does. Either way `run_one_job` must not fire
and `last_status` must not be downgraded.
"""
import json
from unittest.mock import patch

from tools.cronjob_tools import cronjob


_JOB = {"id": "job-routed-1", "name": "keeper digest", "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "deliver": "telegram:123456:789"}


def _patched_common(routed=True):
    """Patches shared by all tests: job resolves, nothing relay-fronted,
    telegram routed from the primary gateway, own config connects nothing."""
    from gateway.config import Platform
    fake_config = type("FakeConfig", (), {
        "get_connected_platforms": lambda self: set(),
    })()
    return (
        patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
        patch("tools.cronjob_tools._relay_fronted_delivery_platforms", return_value=set()),
        patch("cron.scheduler_preflight._delivery_platform_routed_from_primary_gateway",
              side_effect=lambda name: bool(routed) and str(name).lower() == "telegram"),
        patch("gateway.config.load_gateway_config", return_value=fake_config),
        patch("tools.cronjob_tools._notify_provider_jobs_changed_safe"),
    )


class TestMultiplexRoutedManualRun:
    def test_served_profile_marks_due_without_running(self):
        """Multiplexer serves this profile: mark due, never run in-process."""
        patches = _patched_common(routed=True)
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("hermes_cli.gateway.named_profile_served_by_running_multiplexer",
                   return_value=True), \
             patch("cron.jobs.trigger_job",
                   return_value={**_JOB, "next_run_at": "now"}) as m_trigger, \
             patch("tools.cronjob_tools.get_job",
                   return_value={**_JOB, "next_run_at": "now"}), \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.claim_job_for_fire",
                   return_value=claimed) as m_claim:
            out = json.loads(cronjob(action="run", job_id="job-routed-1"))

        assert out["success"] is True
        assert "next scheduler tick" in json.dumps(out).lower()
        m_trigger.assert_called_once()
        m_run.assert_not_called()
        m_claim.assert_not_called()

    def test_unserved_profile_fails_fast_without_running(self):
        """No multiplexer serves this profile: refuse up front, touch nothing."""
        patches = _patched_common(routed=True)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("hermes_cli.gateway.named_profile_served_by_running_multiplexer",
                   return_value=False), \
             patch("cron.jobs.trigger_job") as m_trigger, \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.claim_job_for_fire") as m_claim:
            out = json.loads(cronjob(action="run", job_id="job-routed-1"))

        assert out["success"] is False
        assert "telegram" in (out.get("error") or "")
        m_trigger.assert_not_called()
        m_run.assert_not_called()
        m_claim.assert_not_called()

    def test_non_routed_job_still_runs_inline(self):
        """No profile route: existing inline behavior is unchanged."""
        patches = _patched_common(routed=False)
        ran = {"id": "job-routed-1", "last_status": "ok", "last_error": None}
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("tools.cronjob_tools.claim_job_for_fire",
                   return_value=claimed), \
             patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=ran), \
             patch("cron.jobs.trigger_job") as m_trigger:
            out = json.loads(cronjob(action="run", job_id="job-routed-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        m_run.assert_called_once()
        m_trigger.assert_not_called()
