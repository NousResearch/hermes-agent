"""Tests for the per-job ``allow_silent`` flag (#53230).

When ``allow_silent=False``:
1. ``_build_job_prompt`` must NOT inject ``[SILENT]`` suppression guidance.
2. The delivery path must NOT suppress a ``[SILENT]`` response.

When ``allow_silent=True`` (default, including back-compat for jobs created
before the field existed):
1. ``_build_job_prompt`` still injects the suppression guidance.
2. The delivery path still suppresses ``[SILENT]`` responses.
"""

import pytest

from cron.jobs import create_job
from cron.scheduler import _build_job_prompt, _is_cron_silence_response, SILENT_MARKER


class TestAllowSilentPromptInjection:
    """The ``allow_silent`` flag controls ``[SILENT]`` guidance injection."""

    def test_default_job_injects_silent_guidance(self):
        """Jobs created without specifying allow_silent get suppression guidance."""
        job = create_job(
            prompt="Daily report",
            schedule="0 9 * * *",
        )
        prompt = _build_job_prompt(job)
        assert SILENT_MARKER in prompt, (
            "Default jobs (allow_silent=True) must have [SILENT] guidance"
        )

    def test_allow_silent_true_injects_guidance(self):
        """Explicit allow_silent=True still injects suppression guidance."""
        job = create_job(
            prompt="Daily report",
            schedule="0 9 * * *",
            allow_silent=True,
        )
        prompt = _build_job_prompt(job)
        assert SILENT_MARKER in prompt

    def test_allow_silent_false_omits_silent_guidance(self):
        """allow_silent=False must NOT inject [SILENT] suppression guidance."""
        job = create_job(
            prompt="Daily report — always send an all-clear",
            schedule="0 9 * * *",
            allow_silent=False,
        )
        prompt = _build_job_prompt(job)
        assert SILENT_MARKER not in prompt, (
            "allow_silent=False jobs must not have [SILENT] guidance injected"
        )
        # But the cron_hint (delivery instructions) should still be present
        assert "scheduled cron job" in prompt

    def test_back_compat_missing_key_defaults_to_true(self):
        """Jobs created before allow_silent existed (no key in dict) default to True."""
        job = create_job(
            prompt="Legacy job",
            schedule="0 9 * * *",
        )
        # Simulate a legacy job by deleting the key
        job.pop("allow_silent", None)
        prompt = _build_job_prompt(job)
        assert SILENT_MARKER in prompt, (
            "Legacy jobs without allow_silent key must default to True"
        )

    def test_cron_hint_delivery_instructions_always_present(self):
        """Regardless of allow_silent, the DELIVERY instruction is always present."""
        for allow_silent in (True, False):
            job = create_job(
                prompt="Test",
                schedule="0 9 * * *",
                allow_silent=allow_silent,
            )
            prompt = _build_job_prompt(job)
            assert "DELIVERY:" in prompt
            assert "scheduled cron job" in prompt


class TestAllowSilentJobField:
    """The ``allow_silent`` field is properly stored in the job dict."""

    def test_field_defaults_to_true(self):
        job = create_job(prompt="Test", schedule="0 9 * * *")
        assert job.get("allow_silent") is True

    def test_field_explicitly_true(self):
        job = create_job(prompt="Test", schedule="0 9 * * *", allow_silent=True)
        assert job.get("allow_silent") is True

    def test_field_explicitly_false(self):
        job = create_job(prompt="Test", schedule="0 9 * * *", allow_silent=False)
        assert job.get("allow_silent") is False

    def test_field_is_bool(self):
        """Ensure the stored value is a proper bool, not truthy/falsy."""
        job = create_job(prompt="Test", schedule="0 9 * * *", allow_silent=False)
        assert isinstance(job.get("allow_silent"), bool)


class TestSilenceDetectionStillWorks:
    """_is_cron_silence_response is not affected by allow_silent — it's a
    pure string check. The allow_silent flag gates whether the scheduler
    *consults* it during delivery, not whether it can detect the marker."""

    def test_detects_silent_marker(self):
        assert _is_cron_silence_response("[SILENT]")

    def test_detects_silent_with_newline(self):
        assert _is_cron_silence_response("[SILENT]\n")

    def test_does_not_detect_normal_text(self):
        assert not _is_cron_silence_response("All systems normal")
        assert not _is_cron_silence_response("Daily report: nothing changed.")
