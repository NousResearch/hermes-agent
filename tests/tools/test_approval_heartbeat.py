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

    def test_heartbeat_fires_while_waiting_for_approval(self):
        """touch_activity_if_due is called repeatedly during the wait."""
        from tools.approval import (
            check_all_command_guards,
            register_gateway_notify,
            resolve_gateway_approval,
        )

        register_gateway_notify(self.SESSION_KEY, lambda _payload: None)

        # Use an Event to signal from _fake_touch back to the main thread
        # so we can resolve as soon as the first heartbeat fires — avoids
        # flakiness from fixed sleeps racing against thread startup.
        first_heartbeat = threading.Event()
        heartbeat_calls: list[str] = []

        def _fake_touch(state, label):
            # Bypass the 10s throttle so the heartbeat fires every loop
            # iteration; we're measuring whether the call happens at all.
            heartbeat_calls.append(label)
            state["last_touch"] = 0.0
            first_heartbeat.set()

        result_holder: dict = {}

        def _run_check():
            try:
                with patch(
                    "tools.environments.base.touch_activity_if_due",
                    side_effect=_fake_touch,
                ):
                    result_holder["result"] = check_all_command_guards(
                        "rm -rf /tmp/nonexistent-heartbeat-target", "local"
                    )
            except Exception as exc:  # pragma: no cover
                result_holder["exc"] = exc

        thread = threading.Thread(target=_run_check, daemon=True)
        thread.start()

        # Wait for at least one heartbeat to fire — bounded at 10s to catch
        # a genuinely hung worker thread without making a green run slow.
        assert first_heartbeat.wait(timeout=10.0), (
            "no heartbeat fired within 10s — the approval wait is blocking "
            "without firing activity pings, which is the exact bug this "
            "test exists to catch"
        )

        # Resolve the approval so the thread exits cleanly.
        resolve_gateway_approval(self.SESSION_KEY, "once")
        thread.join(timeout=5)

        assert not thread.is_alive(), "approval wait did not exit after resolve"
        assert "exc" not in result_holder, (
            f"check_all_command_guards raised: {result_holder.get('exc')!r}"
        )

        # The fix: heartbeats fire while waiting.  Before the fix this list
        # was empty because event.wait() blocked for the full timeout with
        # no activity pings.
        assert heartbeat_calls, "expected at least one heartbeat"
        assert all(
            call == "waiting for user approval" for call in heartbeat_calls
        ), f"unexpected heartbeat labels: {set(heartbeat_calls)}"

        # Sanity: the approval was resolved with "once" → command approved.
        assert result_holder["result"]["approved"] is True

    def test_wait_returns_immediately_on_user_response(self):
        """Polling slices don't delay responsiveness — resolve is near-instant."""
        from tools.approval import (
            check_all_command_guards,
            register_gateway_notify,
            resolve_gateway_approval,
        )

        register_gateway_notify(self.SESSION_KEY, lambda _payload: None)

        start_time = time.monotonic()
        result_holder: dict = {}

        def _run_check():
            result_holder["result"] = check_all_command_guards(
                "rm -rf /tmp/nonexistent-fast-target", "local"
            )

        thread = threading.Thread(target=_run_check, daemon=True)
        thread.start()

        # Resolve almost immediately — the wait loop should return within
        # its current 1s poll slice.
        time.sleep(0.1)
        resolve_gateway_approval(self.SESSION_KEY, "once")
        thread.join(timeout=5)
        elapsed = time.monotonic() - start_time

        assert not thread.is_alive()
        assert result_holder["result"]["approved"] is True
        # Generous bound to tolerate CI load; the previous single-wait
        # impl returned in <10ms, the polling impl is bounded by the 1s
        # slice length.
        assert elapsed < 3.0, f"resolution took {elapsed:.2f}s, expected <3s"

    def test_heartbeat_import_failure_does_not_break_wait(self):
        """If tools.environments.base can't be imported, the wait still works."""
        from tools.approval import (
            check_all_command_guards,
            register_gateway_notify,
            resolve_gateway_approval,
        )

        register_gateway_notify(self.SESSION_KEY, lambda _payload: None)

        result_holder: dict = {}
        import builtins
        real_import = builtins.__import__

        def _fail_environments_base(name, *args, **kwargs):
            if name == "tools.environments.base":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        def _run_check():
            with patch.object(builtins, "__import__",
                              side_effect=_fail_environments_base):
                result_holder["result"] = check_all_command_guards(
                    "rm -rf /tmp/nonexistent-import-fail-target", "local"
                )

        thread = threading.Thread(target=_run_check, daemon=True)
        thread.start()

        time.sleep(0.2)
        resolve_gateway_approval(self.SESSION_KEY, "once")
        thread.join(timeout=5)

        assert not thread.is_alive()
        # Even when heartbeat import fails, the approval flow completes.
        assert result_holder["result"]["approved"] is True
