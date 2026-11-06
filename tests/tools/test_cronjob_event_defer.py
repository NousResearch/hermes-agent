"""Event-driven cron fires must defer and replay on claim overlap, not drop the event.

A ``cron_job`` webhook route fires ONE existing job per incoming event, and each fire
holds the at-most-once ``fire_claim`` for the whole agent run (minutes). A burst — one
GitHub PR open fans out to pull_request + check_suite + issue_comment within seconds —
therefore collides with the claim: every event landing in the in-flight window used to be
refused with "Job is already being fired by the scheduler; not run again" and DISCARDED
(the webhook adapter logged a warning and the payload was lost; a 2026-10 live gateway
dropped PR-open events exactly this way). Event fires are queueable by nature — the job
re-reads state each run — so the refused context is deferred, and one merged follow-up
run replays the whole batch once the in-flight run releases the claim.
"""
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.waiting import wait_until
from tools import cronjob_tools
from tools.cronjob_event_defer import (
    MAX_MERGED_CHARS,
    MAX_PENDING_PER_JOB,
    clear as clear_deferred,
    defer_event,
    merge_contexts,
    pending_count,
    take_pending,
)

_JOB = {"id": "evjob-1", "name": "merge gate", "prompt": "p",
        "schedule": {"kind": "cron", "expr": "7 * * * *"}}
_OVERLAP = {"claimed": False, "success": False, "reason_code": "overlap",
            "error": "Job is already being fired by the scheduler; not run again."}
_OK = {"claimed": True, "success": True, "error": None}


@pytest.fixture(autouse=True)
def _clean_defer_queue():
    clear_deferred()
    yield
    clear_deferred()


class TestEventFireDefersAndReplays:
    def test_overlap_defers_then_replays_all_pending_contexts_merged(self, caplog):
        """The invariant: an overlap-refused event answers with an immediate deferred
        result (never a drop), and the follow-up run carries EVERY queued event context
        merged into one prompt, with a replay confirmation logged."""
        calls = []
        replay_may_proceed = threading.Event()

        def fake_exec(job, extra_prompt=None):
            calls.append(extra_prompt or "")
            if len(calls) == 1:
                return dict(_OVERLAP)  # event 1: an in-flight run holds the claim
            return dict(_OK)           # the replay attempt claims and runs

        def gated_sleep(seconds):
            # Hold the replayer in its first poll so the test queues a second event
            # into the SAME batch deterministically (no sleep-race flake).
            assert replay_may_proceed.wait(timeout=5.0), "replayer never reached its poll"

        fake_time = SimpleNamespace(monotonic=time.monotonic, sleep=gated_sleep)

        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._execute_job_now", side_effect=fake_exec), \
             patch("cron.scheduler.is_job_running", return_value=False), \
             patch("tools.cronjob_tools.time", fake_time), \
             caplog.at_level(logging.INFO, logger="tools.cronjob_tools"):
            first = cronjob_tools.execute_job_for_event("evjob-1", "ctx-1")
            assert first.get("deferred") is True
            assert first.get("success") is True

            defer_event("evjob-1", "ctx-2", event_type="check_suite")
            replay_may_proceed.set()
            assert wait_until(lambda: len(calls) >= 2), calls

        assert pending_count("evjob-1") == 0
        assert "ctx-1" in calls[1] and "ctx-2" in calls[1]  # merged; nothing dropped
        assert any("replay confirmation" in r.message for r in caplog.records)

    def test_persistent_overlap_gives_up_but_keeps_events_queued(self, caplog):
        """If the claim never frees within the wait budget the replayer stops with a loud
        error — and the batch stays queued (deferred, not discarded), so a later event can
        still elect a new replayer and drain it."""
        calls = []

        def fake_exec(job, extra_prompt=None):
            calls.append(extra_prompt or "")
            return dict(_OVERLAP)

        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._execute_job_now", side_effect=fake_exec), \
             patch("cron.scheduler.is_job_running", return_value=False), \
             patch("tools.cronjob_tools._EVENT_REPLAY_POLL_SECONDS", 0.01), \
             caplog.at_level(logging.WARNING, logger="tools.cronjob_tools"):
            result, thread = cronjob_tools._defer_event_fire_and_replay(
                dict(_JOB), "ctx-1", max_wait_seconds=0.05)
            assert result.get("deferred") is True
            assert thread is not None
            assert wait_until(lambda: not thread.is_alive()), "replayer never gave up"

        assert pending_count("evjob-1") == 1  # kept queued for a future replay owner
        assert any("gave up" in r.message.lower() for r in caplog.records)

    def test_non_overlap_refusals_pass_through_undefended(self):
        """Missing/paused refusals are configuration states, not contention: they must be
        returned to the caller unchanged and must NOT queue a replay (a paused job would
        then fire the moment someone resumes it)."""
        for code in ("missing", "paused"):
            refusal = {"claimed": False, "success": False, "reason_code": code,
                       "error": f"refused: {code}"}
            with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
                 patch("tools.cronjob_tools._execute_job_now", return_value=refusal):
                out = cronjob_tools.execute_job_for_event("evjob-1", "ctx-x")
            assert out == refusal
            assert pending_count("evjob-1") == 0


class TestClaimRefusalReasonCodes:
    def test_reason_code_labels_each_claim_refusal(self):
        """The deferral decision keys off a structured code, not the human message."""
        paused_job = {**_JOB, "enabled": False, "paused_at": "2026-10-17T00:00:00+00:00"}
        for refreshed, code in ((None, "missing"), (paused_job, "paused"),
                                (dict(_JOB), "overlap")):
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
                 patch("tools.cronjob_tools.get_job", return_value=refreshed):
                _claimed, err = cronjob_tools._claim_for_manual_run("evjob-1", "test")
            assert err["reason_code"] == code


class TestDeferQueueBounds:
    def test_first_deferrer_is_the_replay_owner(self):
        first = defer_event("evjob-3", "ctx-a")
        second = defer_event("evjob-3", "ctx-b")
        assert first == (1, True)
        assert second == (2, False)

    def test_queue_is_bounded_drop_oldest_and_merge_is_capped(self):
        total = MAX_PENDING_PER_JOB + 7
        for i in range(total):
            defer_event("evjob-4", f"ctx-{i}", event_type=f"ev{i}")
        assert pending_count("evjob-4") == MAX_PENDING_PER_JOB
        events = take_pending("evjob-4")
        assert len(events) == MAX_PENDING_PER_JOB
        assert events[0].context == "ctx-7"  # the oldest overflowed entries were dropped
        merged = merge_contexts(events)
        assert len(merged) <= MAX_MERGED_CHARS
        # Under budget pressure the NEWEST event keeps its body; every event keeps a header.
        assert f"ctx-{total - 1}" in merged
        assert merged.count("--- deferred event") == MAX_PENDING_PER_JOB
