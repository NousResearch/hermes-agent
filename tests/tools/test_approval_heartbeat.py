"""Tests for the activity-heartbeat behavior of the blocking gateway approval wait.

Regression test for false gateway inactivity timeouts firing while the agent
is legitimately blocked waiting for a user to respond to a dangerous-command
approval prompt.  Before the fix, ``entry.event.wait(timeout=...)`` blocked
silently — no ``_touch_activity()`` calls — and the gateway's inactivity
watchdog (``agent.gateway_timeout``, default 1800s) would kill the agent
while the user was still choosing whether to approve.

The fix polls the event in short slices and fires ``touch_activity_if_due``
between slices, mirroring ``_wait_for_process`` in ``tools/environments/base.py``.
"""

import os
import threading
import time
from unittest.mock import patch


def _clear_approval_state():
    """Reset all module-level approval state between tests."""
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


class TestApprovalHeartbeat:
    """The blocking gateway approval wait must fire activity heartbeats.

    Without heartbeats, the gateway's inactivity watchdog kills the agent
    thread while it's legitimately waiting for a slow user to respond to
    an approval prompt (observed in real user logs: MRB, April 2026).
    """

    SESSION_KEY = "heartbeat-test-session"

    def setup_method(self):
        _clear_approval_state()
        self._saved_env = {
            k: os.environ.get(k)
            for k in ("HERMES_GATEWAY_SESSION", "HERMES_YOLO_MODE",
                      "HERMES_SESSION_KEY")
        }
        os.environ.pop("HERMES_YOLO_MODE", None)
        os.environ["HERMES_GATEWAY_SESSION"] = "1"
        # The blocking wait path reads the session key via contextvar OR
        # os.environ fallback.  Contextvars don't propagate across threads
        # by default, so env var is the portable way to drive this in tests.
        os.environ["HERMES_SESSION_KEY"] = self.SESSION_KEY

    def teardown_method(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _clear_approval_state()

    def test_gateway_timeout_zero_waits_until_explicit_resolution(self):
        """gateway_timeout=0 means no approval timeout, not immediate deny."""
        from tools import approval as mod

        notified = threading.Event()
        result_box = {}

        def notify(_approval_data):
            notified.set()

        def worker():
            result_box["result"] = mod.check_all_command_guards(
                "git reset --hard",
                "local",
            )

        mod.register_gateway_notify(self.SESSION_KEY, notify)
        with patch.object(
            mod,
            "_get_approval_config",
            return_value={"mode": "manual", "gateway_timeout": 0},
        ):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                assert notified.wait(2), "approval prompt was not registered"
                # Old behavior treated 0 as an already-expired deadline and
                # returned BLOCKED immediately.  Give that regression a chance
                # to happen before resolving.
                time.sleep(0.05)
                assert thread.is_alive()
                assert mod.resolve_gateway_approval(self.SESSION_KEY, "once") == 1
                thread.join(2)
            finally:
                if thread.is_alive():
                    mod.resolve_gateway_approval(self.SESSION_KEY, "deny", resolve_all=True)
                    thread.join(2)

        assert not thread.is_alive()
        assert result_box["result"]["approved"] is True


