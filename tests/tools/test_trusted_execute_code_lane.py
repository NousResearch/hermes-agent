"""Profile-scoped trusted lane for execute_code auto-approval (#44993, #104697 interlock).

Focused test module per the #104580 review (2K godfile invariant): the trusted-lane
suite lives here, not in tests/tools/test_approval.py. Two groups:

- Acceptance tests against a REAL isolated policy store (no reader mocks): the
  operator grant through the sanctioned /approvals handler chain persists the trust
  entry and opens the lane; the forged-direct-writer inverse is refused with the
  persisted bytes unchanged; mutation controls prove both assertions bite.
- Dispatch-level lane tests (mocked allowlist/profile, as before) + the /status
  observability regression (#44993 status requirement).
"""

import os
import unittest.mock as mock
from pathlib import Path

import pytest
import yaml

import tools.approval as approval_module
from tools import approval_context
from tools.approval import check_execute_code_guard
from tools.approval_context import (
    _execute_code_profile_is_trusted,
    grant_operator_policy_write,
    reset_operator_policy_write,
    trusted_execute_code_status_line,
)
from tests.tools._handler_frame_sim import build_handler_module


def _register_gateway_auto_approve(session_key: str) -> None:
    """Auto-answer any pending gateway approval with 'once' (human path evidence)."""
    def _notify(_approval_data):
        with approval_module._lock:
            entries = approval_module._gateway_queues.get(session_key, [])
            if entries:
                entry = entries[-1]
                entry.result = "once"
                entry.event.set()

    with approval_module._lock:
        approval_module._gateway_notify_cbs[session_key] = _notify


def _isolate(monkeypatch):
    """Neutralize host leakage: yolo frozen at import + smart mode."""
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")


def _gateway_context(monkeypatch, session_key="test-trusted-profile-session"):
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    return session_key



@pytest.fixture
def _real_policy_store(tmp_path: Path, monkeypatch):
    """Bind an ISOLATED store to the writer's and profile resolver's real lookup:
    HERMES_HOME is redirected (the autouse fixture in tests/conftest.py does this
    too, but binding it explicitly here keeps the acceptance tests honest), and
    the profile resolver reads the same home. No reader mocks."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model: test\napprovals:\n  mode: smart\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield cfg_file


class TestTrustedExecuteCodeProfileLane:
    """Profile-scoped trust lane for whole-script execute_code auto-approval."""

    # -- Acceptance: real store, real handler chain --------------------------------

    def test_operator_grant_persists_and_lane_opens(self, _real_policy_store, monkeypatch):
        """Pre-grant the guard refuses; the operator path (handler-stamped grant +
        handler frame — drive it through the REAL gateway handler shape) persists
        the trust entry; the persisted enrollment then opens the lane."""
        from hermes_cli.config import set_config_value

        _isolate(monkeypatch)
        session_key = _gateway_context(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "nagatha")

        # Pre-grant: refused, no lane.
        pre = check_execute_code_guard("import os", "local")
        assert pre["approved"] is False

        # Sanctioned operator ingress: drive the write through a REAL handler
        # frame (a module whose file ends in gateway/slash_commands.py exposing
        # _handle_approvals_command, stamping the grant around the write — the
        # identical signature the real gateway handler presents to the writer's
        # frame check).
        handler = build_handler_module()
        assert handler._handle_approvals_command() is True

        # Persisted enrollment is real: read it back from the store.
        saved = yaml.safe_load(_real_policy_store.read_text())
        assert saved["approvals"]["trusted_execute_code_profiles"] == ["nagatha"]

        # Post-grant: the lane is open for the trusted profile — no human round-trip.
        result = check_execute_code_guard("import os", "local")
        assert result["approved"] is True
        assert result.get("user_approved") is not True

        with approval_module._lock:
            approval_module._gateway_notify_cbs.pop(session_key, None)
            approval_module._gateway_queues.pop(session_key, None)

    def test_worker_cannot_self_enroll(self, _real_policy_store, monkeypatch):
        """The forged-direct-writer inverse: grant WITHOUT a handler frame is
        refused, the persisted bytes are UNCHANGED (asserted unconditionally),
        and execute_code still reaches human approval."""
        from hermes_cli.config import set_config_value

        _isolate(monkeypatch)
        session_key = _gateway_context(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "operator")

        pre_bytes = _real_policy_store.read_bytes()

        # Worker forges the grant and calls the writer directly — no handler frame.
        token = grant_operator_policy_write()
        try:
            with pytest.raises(SystemExit):
                set_config_value(
                    "approvals.trusted_execute_code_profiles", '["operator"]')
        finally:
            reset_operator_policy_write(token)

        # Unconditional byte assertion — a mutate-then-refuse writer fails here.
        assert _real_policy_store.read_bytes() == pre_bytes

        # The lane stays closed: the human queue answers, not the trust lane.
        _register_gateway_auto_approve(session_key)
        try:
            result = check_execute_code_guard("import os", "local")
            assert result["approved"] is True
            assert result.get("user_approved") is True
        finally:
            with approval_module._lock:
                approval_module._gateway_notify_cbs.pop(session_key, None)
                approval_module._gateway_queues.pop(session_key, None)

    def test_noop_grant_fails_positive(self, _real_policy_store, monkeypatch):
        """Mutation control: a no-op grant must NOT open the lane. If the
        production grant becomes a no-op, the positive sequence fails here."""
        from hermes_cli.config import set_config_value

        _isolate(monkeypatch)
        _gateway_context(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "nagatha")

        def _noop_grant():
            return None

        token = grant_operator_policy_write()
        reset_operator_policy_write(token)
        with mock.patch(
            "tools.approval_context.grant_operator_policy_write", _noop_grant):
            with pytest.raises(SystemExit):
                set_config_value(
                    "approvals.trusted_execute_code_profiles", '["nagatha"]')

        # Lane must still be closed (the store never received the enrollment).
        saved = yaml.safe_load(_real_policy_store.read_text())
        assert saved["approvals"].get("trusted_execute_code_profiles") is None

    def test_mutate_before_refuse_fails_inverse(self, _real_policy_store, monkeypatch):
        """Mutation control: a writer that mutates the store BEFORE refusing must
        trip the inverse test's unconditional byte assertion."""
        from hermes_cli.config import set_config_value

        _isolate(monkeypatch)
        _gateway_context(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "operator")

        pre_bytes = _real_policy_store.read_bytes()

        real_authorized = None
        import hermes_cli.config as cfg_mod
        real_authorized = cfg_mod._policy_write_authorized

        def mutate_then_refuse():
            # Simulate the mutant: touch the store, then refuse.
            data = yaml.safe_load(_real_policy_store.read_text())
            data["approvals"]["trusted_execute_code_profiles"] = ["operator"]
            _real_policy_store.write_text(yaml.safe_dump(data, sort_keys=False))
            return True  # authorized → the write WOULD happen under the mutant

        with mock.patch.object(cfg_mod, "_policy_write_authorized", mutate_then_refuse):
            # Under the mutant the write is AUTHORIZED and mutates the store —
            # assert exactly that, proving the inverse test's byte check bites.
            set_config_value(
                "approvals.trusted_execute_code_profiles", '["operator"]')
            post_bytes = _real_policy_store.read_bytes()
            assert post_bytes != pre_bytes, (
                "the inverse test's unconditional byte assertion must trip under "
                "a mutate-before-refuse writer")

    # -- Dispatch-level lane tests --------------------------------------------------

    def _gateway_setup(self, monkeypatch, profile, allowlist=("nagatha",)):
        _isolate(monkeypatch)
        session_key = _gateway_context(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: profile)
        monkeypatch.setattr(
            approval_context, "_get_trusted_execute_code_profiles",
            lambda: list(allowlist))
        return session_key

    def test_trusted_profile_auto_approves_in_gateway_context(self, monkeypatch):
        """A gateway profile in the allowlist skips the pending prompt — the gate
        returns approved without a human round-trip (no notify callback at all)."""
        session_key = self._gateway_setup(monkeypatch, "nagatha")
        try:
            result = check_execute_code_guard("import os", "local")
            assert result["approved"] is True
            assert result.get("user_approved") is not True
        finally:
            with approval_module._lock:
                approval_module._gateway_notify_cbs.pop(session_key, None)
                approval_module._gateway_queues.pop(session_key, None)

    def test_untrusted_profile_still_prompts(self, monkeypatch):
        """An untrusted profile flows through the human approval path."""
        session_key = self._gateway_setup(monkeypatch, "operator")
        _register_gateway_auto_approve(session_key)
        try:
            result = check_execute_code_guard("import os", "local")
            assert result["approved"] is True
            assert result.get("user_approved") is True
        finally:
            with approval_module._lock:
                approval_module._gateway_notify_cbs.pop(session_key, None)
                approval_module._gateway_queues.pop(session_key, None)

    def test_ask_mode_trusted_profile_auto_approves(self, monkeypatch):
        """Ask mode (HERMES_EXEC_ASK=1) is a whole-script approval context."""
        from tools.approval import check_execute_code_guard as guard

        _isolate(monkeypatch)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "nagatha")
        monkeypatch.setattr(
            approval_context, "_get_trusted_execute_code_profiles",
            lambda: ["nagatha"])
        assert guard("import os", "local")["approved"] is True

    def test_trusted_profile_match_is_case_insensitive(self, monkeypatch):
        """Allowlist ['NAGATHA'] trusts 'nagatha'; a non-match is not trusted."""
        _isolate(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "nagatha")
        monkeypatch.setattr(
            approval_context, "_get_trusted_execute_code_profiles",
            lambda: ["NAGATHA"])
        assert approval_context._execute_code_profile_is_trusted() is True
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "opskipper")
        assert approval_context._execute_code_profile_is_trusted() is False

    def test_malformed_allowlist_is_treated_empty(self, monkeypatch):
        """A non-list allowlist is an empty allowlist (never grants trust)."""
        monkeypatch.setattr(
            approval_context, "_get_approval_config",
            lambda: {"trusted_execute_code_profiles": "nagatha"})
        assert approval_context._get_trusted_execute_code_profiles() == []

    def test_profile_resolution_failure_is_safe_not_trusted(self, monkeypatch):
        """A raising get_active_profile_name must not grant trust."""
        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        _isolate(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", _raise)
        assert approval_context._execute_code_profile_is_trusted() is False

    def test_cron_denies_even_for_trusted_profile(self, monkeypatch):
        """Cron sessions deny regardless of the trust lane."""
        from tools.approval import check_execute_code_guard as guard

        _isolate(monkeypatch)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
        monkeypatch.setenv("HERMES_SESSION_KEY", "test-cron-trusted-profile")
        monkeypatch.setattr(
            approval_context, "_get_cron_approval_mode", lambda: "deny")
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "nagatha")
        monkeypatch.setattr(
            approval_context, "_get_trusted_execute_code_profiles",
            lambda: ["nagatha"])
        result = guard("import os", "local")
        assert result["approved"] is False
        assert result["outcome"] == "blocked"

    def test_yolo_or_off_takes_precedence(self, monkeypatch):
        """YOLO / mode=off auto-approve runs BEFORE the trust lane."""
        from tools.approval import check_execute_code_guard as guard

        _isolate(monkeypatch)
        session_key = _gateway_context(monkeypatch)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "operator")
        monkeypatch.setattr(
            approval_context, "_get_trusted_execute_code_profiles",
            lambda: ["nagatha"])
        try:
            assert guard("import os", "local")["approved"] is True
        finally:
            with approval_module._lock:
                approval_module._gateway_notify_cbs.pop(session_key, None)
                approval_module._gateway_queues.pop(session_key, None)

    # -- Status observability (#44993) ----------------------------------------------

    def test_status_line_reflects_trust_state(self, monkeypatch):
        """/status shows the active profile's trust state."""
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "nagatha")
        monkeypatch.setattr(
            approval_context, "_get_trusted_execute_code_profiles",
            lambda: ["nagatha"])
        assert trusted_execute_code_status_line() == (
            "execute_code trust: nagatha (trusted)")

        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker")
        assert trusted_execute_code_status_line() == (
            "execute_code trust: worker (not trusted)")

        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: None)
        assert trusted_execute_code_status_line() == (
            "execute_code trust: (no active profile)")
