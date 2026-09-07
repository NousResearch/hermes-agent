"""Gateway approval coalescing is requester-aware: a follower whose identical command comes from a
different sender (a lifecycle child of A's turn still pending while B's turn runs the same command)
gets its own prompt naming B as requester, instead of silently riding along on A's prompt — which
names only A, so the ``forbid_self_approval`` guard could not see B behind it. Same-sender (and
gateway-less, both ``""``) parallel calls still coalesce as before.
"""

import threading
import time

from gateway.session_context import set_session_vars
from tools import approval_context


class TestCoalesceAcrossRequesters:
    SESSION_KEY = "test-coalesce-requester"

    def setup_method(self):
        from tools import approval as mod

        mod._gateway_queues.clear()
        mod._gateway_notify_cbs.clear()
        mod._session_approved.clear()
        mod._permanent_approved.clear()

    teardown_method = setup_method

    @staticmethod
    def _data():
        return {"command": "rm -rf .git", "description": "desc",
                "pattern_key": "dangerous", "pattern_keys": ["dangerous"]}

    def _run_as(self, user_id, notify, results, idx):
        from tools import approval as mod

        def _target():
            set_session_vars(platform="discord", user_id=user_id, session_key=self.SESSION_KEY)
            results[idx] = mod._await_gateway_decision(self.SESSION_KEY, notify, self._data())

        thread = threading.Thread(target=_target)
        thread.start()
        return thread

    @staticmethod
    def _wait_until(predicate, what):
        for _ in range(600):
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError(f"timed out waiting for {what}")

    def _manual_mode(self, monkeypatch):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "timeout": 30})

    def test_different_sender_gets_own_prompt_with_own_requester(self, monkeypatch):
        from tools import approval as mod

        self._manual_mode(monkeypatch)
        notified_a, notified_b, results = [], [], [None, None]

        thread_a = self._run_as("A", notified_a.append, results, 0)
        self._wait_until(lambda: mod._gateway_queues.get(self.SESSION_KEY) and notified_a, "leader prompt")
        thread_b = self._run_as("B", notified_b.append, results, 1)
        self._wait_until(lambda: notified_b, "follower prompt")

        assert notified_a[0]["requester_user_id"] == "A"
        assert notified_b[0]["requester_user_id"] == "B"
        assert len(mod.list_gateway_approvals(self.SESSION_KEY)) == 2

        # A's prompt (oldest) decided by B, B's prompt decided by A: each is a legitimate four-eyes decision.
        assert mod.resolve_gateway_approval(self.SESSION_KEY, "once", decided_by="discord:B") == 1
        assert mod.resolve_gateway_approval(self.SESSION_KEY, "deny", decided_by="discord:A") == 1
        thread_a.join(5)
        thread_b.join(5)
        assert results[0]["choice"] == "once" and results[0]["decided_by"] == "discord:B"
        assert results[1]["choice"] == "deny" and results[1]["decided_by"] == "discord:A"
        assert not results[1].get("coalesced")

    def test_same_sender_still_coalesces(self, monkeypatch):
        from tools import approval as mod

        self._manual_mode(monkeypatch)
        hooks = []
        fire = approval_context._fire_approval_hook
        monkeypatch.setattr(approval_context, "_fire_approval_hook",
                            lambda name, **kw: hooks.append((name, kw)) or fire(name, **kw))
        notified, results = [], [None, None]

        thread_1 = self._run_as("A", notified.append, results, 0)
        self._wait_until(lambda: mod._gateway_queues.get(self.SESSION_KEY) and notified, "leader prompt")
        thread_2 = self._run_as("A", notified.append, results, 1)
        self._wait_until(lambda: any(n == "pre_approval_request" and kw.get("coalesced") for n, kw in hooks),
                         "follower attach")

        assert len(notified) == 1
        assert len(mod.list_gateway_approvals(self.SESSION_KEY)) == 1
        assert mod.resolve_gateway_approval(self.SESSION_KEY, "session", decided_by="discord:B") == 1
        thread_1.join(5)
        thread_2.join(5)
        assert results[0]["choice"] == "session"
        assert results[1]["choice"] == "session" and results[1]["coalesced"]
