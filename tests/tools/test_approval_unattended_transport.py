"""Transport-scoped unattended approval exclusion (#98728).

A registered per-session approval transport (``/v1/runs`` run-scoped callback,
streaming chat bridge) makes an api_server session resolvable; listener-less
routes and notifier-blind webhook lanes keep the fail-closed unattended deny.
"""

import os
import threading
import time

import tools.approval as approval_module
from tools import approval_context


class TestUnattendedApprovalTransport:
    """A registered per-session approval transport makes an unattended
    platform session resolvable (#98728).

    ``/v1/runs`` binds a run-scoped session key and registers a gateway
    notify callback resolvable via ``POST /v1/runs/{run_id}/approval``, so
    its guarded actions must go through the interactive gateway approval
    flow instead of the instant unattended deny/approve. Listener-less
    sessions (no registered callback) keep the fail-closed behavior from
    #87509. webhook/msgraph_webhook stay unattended even with a registered
    callback: their generic lane registers one for every turn that no
    surface can answer.
    """

    SESSION_KEY = "test-runs-approval-transport"

    def setup_method(self):
        self._saved_env = {
            k: os.environ.get(k)
            for k in (
                "HERMES_CRON_SESSION",
                "HERMES_GATEWAY_SESSION",
                "HERMES_INTERACTIVE",
                "HERMES_EXEC_ASK",
                "HERMES_SESSION_PLATFORM",
                "HERMES_SESSION_KEY",
            )
        }
        os.environ.pop("HERMES_CRON_SESSION", None)
        os.environ.pop("HERMES_GATEWAY_SESSION", None)
        os.environ.pop("HERMES_INTERACTIVE", None)
        os.environ.pop("HERMES_EXEC_ASK", None)
        os.environ["HERMES_SESSION_PLATFORM"] = "api_server"
        os.environ["HERMES_SESSION_KEY"] = self.SESSION_KEY

    def teardown_method(self):
        from tools import approval as mod
        mod._gateway_queues.clear()
        mod._gateway_notify_cbs.clear()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _isolate(self, monkeypatch):
        """Manual mode, no frozen yolo, short approval timeout for guards."""
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(
            approval_context,
            "_get_approval_config",
            lambda: {"mode": "manual", "timeout": 2},
        )

    def test_registered_transport_is_interactive(self, monkeypatch):
        """A registered notify callback flips both context checks; unregistering restores."""
        from tools.approval import (
            _is_gateway_approval_context,
            _is_unattended_platform_approval_context,
        )

        assert _is_unattended_platform_approval_context() is True
        assert _is_gateway_approval_context() is False

        approval_module.register_gateway_notify(
            self.SESSION_KEY, lambda data: None
        )
        try:
            assert _is_unattended_platform_approval_context() is False
            assert _is_gateway_approval_context() is True
        finally:
            approval_module.unregister_gateway_notify(self.SESSION_KEY)

        assert _is_unattended_platform_approval_context() is True
        assert _is_gateway_approval_context() is False

    def test_notifier_blind_platform_stays_unattended(self, monkeypatch):
        """webhook/msgraph_webhook register a notifier for every inbound turn, so a
        registered callback there must NOT flip them interactive: notifier presence
        is not resolver ownership on those fire-and-forget surfaces."""
        from tools.approval import (
            _is_gateway_approval_context,
            _is_unattended_platform_approval_context,
        )

        for platform in ("webhook", "msgraph_webhook"):
            os.environ["HERMES_SESSION_PLATFORM"] = platform
            approval_module.register_gateway_notify(
                self.SESSION_KEY, lambda data: None
            )
            try:
                assert _is_unattended_platform_approval_context() is True, platform
                assert _is_gateway_approval_context() is False, platform
            finally:
                approval_module.unregister_gateway_notify(self.SESSION_KEY)

    def test_listenerless_execute_code_still_denies(self, monkeypatch):
        """No registered callback: instant deny, no pending approval (#87509 kept)."""
        from tools.approval import check_execute_code_guard

        self._isolate(monkeypatch)
        result = check_execute_code_guard("import os", "local")
        assert result["approved"] is False
        assert "approvals.unattended_mode" in result["message"]
        assert not approval_module._gateway_queues.get(self.SESSION_KEY)

    def test_transport_execute_code_resolves_once(self, monkeypatch):
        """With a run-scoped callback, execute_code emits approval.request and
        continues on an exact ``once`` resolution."""
        from tools.approval import check_execute_code_guard

        self._isolate(monkeypatch)
        notified = []
        approval_module.register_gateway_notify(
            self.SESSION_KEY, lambda data: notified.append(dict(data))
        )

        result_holder = {}

        def _check():
            result_holder["r"] = check_execute_code_guard("import os", "local")

        t = threading.Thread(target=_check)
        t.start()
        try:
            for _ in range(400):
                if approval_module._gateway_queues.get(self.SESSION_KEY):
                    break
                time.sleep(0.005)
            assert approval_module._gateway_queues.get(self.SESSION_KEY), (
                "approval.request never became pending for the transport session"
            )
            assert approval_module.resolve_gateway_approval(
                self.SESSION_KEY, "once"
            ) == 1
        finally:
            t.join(timeout=5)

        assert "r" in result_holder, "approval wait did not return after resolve"
        r = result_holder["r"]
        assert r["approved"] is True
        assert r.get("user_approved") is True
        assert len(notified) == 1, "notify callback should have fired exactly once"

    def test_transport_execute_code_honors_deny(self, monkeypatch):
        """An exact ``deny`` resolution through the transport blocks the action."""
        from tools.approval import check_execute_code_guard

        self._isolate(monkeypatch)
        approval_module.register_gateway_notify(
            self.SESSION_KEY, lambda data: None
        )

        result_holder = {}

        def _check():
            result_holder["r"] = check_execute_code_guard("import os", "local")

        t = threading.Thread(target=_check)
        t.start()
        try:
            for _ in range(400):
                if approval_module._gateway_queues.get(self.SESSION_KEY):
                    break
                time.sleep(0.005)
            assert approval_module.resolve_gateway_approval(
                self.SESSION_KEY, "deny"
            ) == 1
        finally:
            t.join(timeout=5)

        assert "r" in result_holder, "approval wait did not return after deny"
        r = result_holder["r"]
        assert r["approved"] is False
        assert r.get("user_consent") is False
        assert r.get("outcome") == "denied"

    def test_transport_dangerous_command_resolves_once(self, monkeypatch):
        """Dangerous commands on a transport session emit approval.request and
        resolve through the gateway wait instead of the instant unattended
        deny."""
        from tools.approval import check_all_command_guards

        self._isolate(monkeypatch)
        notified = []
        approval_module.register_gateway_notify(
            self.SESSION_KEY, lambda data: notified.append(dict(data))
        )

        result_holder = {}

        def _check():
            result_holder["r"] = check_all_command_guards(
                "sudo systemctl restart nginx", "local"
            )

        t = threading.Thread(target=_check)
        t.start()
        try:
            for _ in range(400):
                if approval_module._gateway_queues.get(self.SESSION_KEY):
                    break
                time.sleep(0.005)
            assert approval_module._gateway_queues.get(self.SESSION_KEY), (
                "approval.request never became pending for the transport session"
            )
            assert approval_module.resolve_gateway_approval(
                self.SESSION_KEY, "once"
            ) == 1
        finally:
            t.join(timeout=5)

        assert "r" in result_holder, "approval wait did not return after resolve"
        r = result_holder["r"]
        assert r["approved"] is True
        assert r.get("user_approved") is True
        assert len(notified) == 1
        assert "unattended platform" not in (r.get("message") or "")
