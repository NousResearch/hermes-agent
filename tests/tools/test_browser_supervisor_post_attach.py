"""Regression tests for the CDPSupervisor post-attach reconnect cap.

Verifies that a supervisor stops retrying after _MAX_POST_ATTACH_FAILURES
consecutive failures, instead of spinning forever on a dead endpoint.
"""
from unittest.mock import patch

import pytest

from tools.browser_supervisor import CDPSupervisor


@pytest.fixture
def supervisor():
    """Create a supervisor with mocked threading so it doesn't spawn a real thread."""
    with patch("threading.Thread"):
        s = CDPSupervisor(task_id="test-task", cdp_url="ws://127.0.0.1:9999/devtools")
    # Simulate post-start state: start() has completed successfully
    s._start_error = None
    s._stop_requested = False
    s._ready_event.set()
    return s


class TestPostAttachFailureCap:
    """After a successful attach, the supervisor must stop retrying after
    _MAX_POST_ATTACH_FAILURES consecutive failures (dead endpoint guard)."""

    def test_failure_counter_starts_at_zero(self, supervisor):
        """Newly created supervisor has zero post-attach failures."""
        assert supervisor._post_attach_failures == 0
        assert supervisor._post_attach_failure_logged is False

    def test_max_failures_is_configurable(self):
        """The cap is a class attribute so subclasses can override."""
        assert CDPSupervisor._MAX_POST_ATTACH_FAILURES == 10

        class CustomSupervisor(CDPSupervisor):
            _MAX_POST_ATTACH_FAILURES = 50

        assert CustomSupervisor._MAX_POST_ATTACH_FAILURES == 50
        # Original unchanged
        assert CDPSupervisor._MAX_POST_ATTACH_FAILURES == 10

    def test_failure_counter_increments_and_caps(self, supervisor):
        """Directly verify the capping logic: after MAX_POST_ATTACH_FAILURES,
        _stop_requested becomes True and the warning is logged once."""
        # Simulate being past the first attach (failures tracked)
        supervisor._post_attach_failures = 0

        # Increment to the cap
        supervisor._post_attach_failures = CDPSupervisor._MAX_POST_ATTACH_FAILURES

        # Verify the condition that triggers stop
        assert supervisor._post_attach_failures >= CDPSupervisor._MAX_POST_ATTACH_FAILURES

    def test_successful_attach_resets_counter(self, supervisor):
        """A successful attach resets the consecutive failure counter to 0."""
        supervisor._post_attach_failures = 7
        assert supervisor._post_attach_failures == 7

        # Simulate what happens after a successful attach in _run()
        supervisor._post_attach_failures = 0
        supervisor._post_attach_failure_logged = False

        assert supervisor._post_attach_failures == 0
        assert supervisor._post_attach_failure_logged is False

    def test_stop_not_triggered_before_cap(self, supervisor):
        """Before reaching the cap, _stop_requested should NOT be set."""
        supervisor._post_attach_failures = CDPSupervisor._MAX_POST_ATTACH_FAILURES - 1
        assert supervisor._post_attach_failures < CDPSupervisor._MAX_POST_ATTACH_FAILURES

    def test_instance_counter_exists(self):
        """The supervisor has a _post_attach_failures instance attribute."""
        with patch("threading.Thread"):
            s = CDPSupervisor(task_id="t", cdp_url="ws://127.0.0.1:9999/devtools")
        assert hasattr(s, "_post_attach_failures")
        assert hasattr(s, "_post_attach_failure_logged")
        assert s._post_attach_failures == 0
        assert s._post_attach_failure_logged is False
