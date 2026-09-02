"""Honesty of the manual-run delivery notice (issue #83993).

A manual ``cronjob(action='run')`` finishes with a completion summary line

    Delivery target: <target> (output was delivered there by the job itself)

that was appended UNCONDITIONALLY for non-local targets — even when
``run_one_job`` had just written ``last_delivery_error`` onto the refreshed
job record because the post-run delivery (telegram/discord/…) failed. The
calling agent then relayed "all good" over a failed delivery.

The note must follow the refreshed job record: a set ``last_delivery_error``
means delivery FAILED with the error text surfaced; an empty/missing error
keeps the legacy wording byte-for-byte (zero regression), and local jobs
always say saved-locally.
"""

import contextlib
import time
from unittest.mock import patch

import pytest

from tools.cronjob_tools import _manual_run_delivery_note


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset the shared async-delegation world around each test.

    The dispatch tests below submit real workers onto the process-wide
    daemon executor in ``tools.async_delegation``. A finished worker parks
    idle holding an ``_idle_semaphore`` token, so the NEXT dispatch in this
    process REUSES that thread instead of spawning a fresh one — and only
    the fresh-spawn path keeps upstream's dispatch-and-return test winning
    its patch-visibility race: ``Thread.start()`` blocks the dispatching
    thread until the worker has bootstrapped, so the worker performs
    ``_run_claimed_job``'s lazy ``from cron.scheduler import run_one_job``
    while the test's patches are still active. On the idle-reuse path
    ``submit`` returns with the GIL still held, the patch block unwinds
    first, and the worker binds the REAL ``run_one_job`` — which then runs
    the fake job for real ("no model configured") and the mock never fires.
    Without this reset, test_cronjob_run_background.py's
    ``test_dispatches_and_returns_handle_immediately`` fails
    deterministically whenever this file runs before it. Mirrors
    ``tests/tools/test_async_delegation.py::_clean_state``.
    """
    from tools import async_delegation as ad
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    # Give just-drained workers a beat to finalize BEFORE resetting, so
    # their completion events land now instead of leaking into the next
    # test's queue (mirrors test_async_delegation.py).
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _job(job_id, deliver):
    """Per-test job dict with a UNIQUE id.

    Background workers outlive their test (daemon executor) and hold the id
    in the scheduler's shared running set until the run finishes; reusing an
    id across tests trips the in-flight dedupe guard on a straggler.
    """
    return {
        "id": job_id,
        "name": f"dn run {job_id}",
        "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "deliver": deliver,
    }


@contextlib.contextmanager
def _bound_session_key(key):
    """Bind the approval session key contextvar (background dispatch gate)."""
    from tools.approval import _approval_session_key

    token = _approval_session_key.set(key)
    try:
        yield
    finally:
        _approval_session_key.reset(token)


def _dispatch_diag(res) -> str:
    """Failure renderer for the wiring tests' dispatch asserts: the result
    dict plus the scheduler running set, so a broken assert names the return
    path that was actually taken instead of a bare KeyError."""
    try:
        from cron.scheduler import get_running_job_ids

        running = sorted(get_running_job_ids())
    except Exception as e:  # pragma: no cover - diagnostic only
        running = f"<unavailable: {e}>"
    return f"dispatch result: {res!r}; running: {running}"


def _drain_completion_event(delegation_id):
    """Wait (bounded) for this delegation's completion event; requeue others.

    The runner executes on a daemon thread, so this must be called while the
    test's patches are still active.
    """
    from tools.process_registry import process_registry

    for _ in range(100):
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            time.sleep(0.05)
            continue
        if evt.get("delegation_id") == delegation_id:
            return evt
        process_registry.completion_queue.put(evt)
        time.sleep(0.05)
    return None


class TestDeliveryNote:
    """``_manual_run_delivery_note`` — the summary-line wording contract."""

    def test_local_always_saved_locally_only(self):
        expected = " (output saved locally only)"
        assert _manual_run_delivery_note("local", {}) == expected
        # Local jobs never deliver — a stale delivery error must not leak in.
        assert (
            _manual_run_delivery_note("local", {"last_delivery_error": "telegram 400"})
            == expected
        )

    def test_remote_without_error_keeps_legacy_wording(self):
        expected = " (output was delivered there by the job itself)"
        assert _manual_run_delivery_note("telegram", {}) == expected
        assert (
            _manual_run_delivery_note("telegram", {"last_delivery_error": None})
            == expected
        )
        assert (
            _manual_run_delivery_note("discord:#ops", {"last_delivery_error": "   "})
            == expected
        )

    def test_empty_or_missing_deliver_reads_saved_locally(self):
        """Falsy deliver = no target, and the fire-time path treats it as
        "local" (no delivery, no delivery error) — the note must not claim
        "delivered there" for a target that doesn't exist (#83993 class)."""
        expected = " (output saved locally only)"
        assert _manual_run_delivery_note("", {}) == expected
        assert _manual_run_delivery_note(None, {}) == expected
        # Falsy deliver never attempts delivery — a stale error (e.g. from an
        # earlier deliver config) must not flip the wording either.
        assert _manual_run_delivery_note("", {"last_delivery_error": "old"}) == expected

    def test_whitespace_deliver_defers_to_error_record(self):
        """Whitespace-only deliver is NOT folded into local: fire time lets it
        through as a target that fails to resolve, so the recorded error must
        stay visible rather than being masked by a saved-locally wording."""
        note = _manual_run_delivery_note(" ", {"last_delivery_error": "no target"})
        assert "delivery FAILED" in note
        assert "no target" in note

    def test_remote_with_error_says_delivery_failed(self):
        note = _manual_run_delivery_note(
            "telegram", {"last_delivery_error": "send failed: 400 Bad Request"}
        )
        assert "delivery FAILED" in note
        assert "send failed: 400 Bad Request" in note

    def test_remote_error_text_truncated_to_200_chars(self):
        note = _manual_run_delivery_note("telegram", {"last_delivery_error": "E" * 500})
        assert "E" * 200 in note
        assert "E" * 201 not in note


class TestRunnerSummaryWiring:
    """The completion event the calling agent actually sees must follow the
    refreshed job record — both directions of issue #83993."""

    def test_delivery_failure_surfaces_in_completion_summary(self):
        from tools.cronjob_tools import _try_dispatch_background_run

        job = _job("job-dn-01", "telegram")
        with _bound_session_key("agent:main:telegram:dm:83993"):
            with (
                patch(
                    "tools.cronjob_tools.claim_job_for_fire",
                    return_value=job,  # claimed snapshot (return_job=True API)
                ),
                patch("cron.scheduler.run_one_job", return_value=True),
                patch(
                    "tools.cronjob_tools.get_job",
                    return_value={
                        # Post-#83993 record shape: mark_job_run writes
                        # delivery_failed (not ok) when only delivery failed.
                        "last_status": "delivery_failed",
                        "last_error": None,
                        "last_delivery_error": "telegram send failed: 400",
                    },
                ),
            ):
                res = _try_dispatch_background_run(job)
                assert res.get("dispatched") is True, _dispatch_diag(res)
                evt = _drain_completion_event(res["delegation_id"])
        assert evt is not None, "completion event never reached the queue"
        summary = evt.get("summary") or ""
        assert "Delivery target: telegram" in summary
        assert "delivery FAILED" in summary
        assert "telegram send failed: 400" in summary
        assert "delivered there by the job itself" not in summary
        # The headline must not read "Result: ok" over an undelivered run.
        assert "Result: FAILED" in summary
        assert "Result: ok" not in summary

    def test_empty_deliver_summary_states_local_not_phantom_target(self):
        """End-to-end: an empty stored deliver must render as the local target
        it behaves as at fire time — never a bare "Delivery target: " followed
        by a delivered-there claim."""
        from tools.cronjob_tools import _try_dispatch_background_run

        job = _job("job-dn-03", "")
        with _bound_session_key("agent:main:telegram:dm:86622"):
            with (
                patch(
                    "tools.cronjob_tools.claim_job_for_fire",
                    return_value=job,  # claimed snapshot (return_job=True API)
                ),
                patch("cron.scheduler.run_one_job", return_value=True),
                patch(
                    "tools.cronjob_tools.get_job",
                    return_value={"last_status": "ok", "last_error": None},
                ),
            ):
                res = _try_dispatch_background_run(job)
                assert res.get("dispatched") is True, _dispatch_diag(res)
                evt = _drain_completion_event(res["delegation_id"])
        assert evt is not None, "completion event never reached the queue"
        summary = evt.get("summary") or ""
        assert "Delivery target: local (output saved locally only)" in summary
        assert "delivered there by the job itself" not in summary

    def test_delivery_success_wording_unchanged_in_completion_summary(self):
        from tools.cronjob_tools import _try_dispatch_background_run

        job = _job("job-dn-02", "telegram")
        with _bound_session_key("agent:main:telegram:dm:83994"):
            with (
                patch(
                    "tools.cronjob_tools.claim_job_for_fire",
                    return_value=job,  # claimed snapshot (return_job=True API)
                ),
                patch("cron.scheduler.run_one_job", return_value=True),
                patch(
                    "tools.cronjob_tools.get_job",
                    return_value={"last_status": "ok", "last_error": None},
                ),
            ):
                res = _try_dispatch_background_run(job)
                assert res.get("dispatched") is True, _dispatch_diag(res)
                evt = _drain_completion_event(res["delegation_id"])
        assert evt is not None, "completion event never reached the queue"
        summary = evt.get("summary") or ""
        assert (
            "Delivery target: telegram (output was delivered there by the job itself)"
        ) in summary
        assert "delivery FAILED" not in summary
