"""A ``command_approval_required`` review-policy transition revokes the superseded grant.

The approvals config is live (mtime-keyed reload), and a grant's key carries the rule's ``review``.
Qualifying the key alone only *hides* a grant while the other policy is active: after
``smart -> human -> smart`` the original key is current again and the pre-tightening grant would
run the command with no guardian call and no prompt. Contract (PR #106779 review, F-001): the moment
the process sees the rule under a new policy — any guarded command, matching or not, or the allowlist
load at startup — it discards the other policy's session AND permanent grants, so the restored policy
starts from zero, including after a restart that reloads ``command_allowlist``.

Driven through the REAL config path (temp ``HERMES_HOME``, file rewrites, cache invalidated by
mtime), not a patched ``_get_approval_config``.
"""

import os
import time

import yaml
import pytest

import hermes_cli.config as hc
import tools.approval as approval_module
from tools import approval_floors, approval_smart
from tools.approval import check_all_command_guards, check_dangerous_command
from tools.approval_context import reset_current_session_key, set_current_session_key

KUBECTL = "kubectl --context admin -n vault exec vault-0 -c vault -- du -sh /vault/data"
PATTERN = "kubectl *--context[= ]admin*"


@pytest.fixture
def live_config(tmp_path, monkeypatch):
    """Real ``config.yaml`` under a temp ``HERMES_HOME``; returns ``set_review(review)`` which rewrites
    the rule in place (keeping whatever ``command_allowlist`` the approval flow persisted)."""
    home = tmp_path / "hermes"
    home.mkdir()
    path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def set_review(review, mode="smart"):
        current = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        current.update({
            "approvals": {"mode": mode, "command_approval_required": [
                {"pattern": PATTERN, "description": "kubectl on admin", "review": review}]},
            "security": {"tirith_enabled": False},
        })
        path.write_text(yaml.safe_dump(current), encoding="utf-8")
        # The reload is keyed on (mtime_ns, size) and "smart" -> "human" keeps the size: stamp a
        # strictly increasing mtime so the test proves the live reload path instead of depending on
        # filesystem timestamp resolution — the cache is never cleared by hand after setup.
        clock["ns"] += 1_000_000_000
        os.utime(path, ns=(clock["ns"], clock["ns"]))

    clock = {"ns": time.time_ns()}
    hc._LOAD_CONFIG_CACHE.clear()
    set_review.path = path
    set_review("smart")
    yield set_review
    hc._LOAD_CONFIG_CACHE.clear()


@pytest.fixture
def isolated_state(monkeypatch):
    session = "test:approval-required:transition"
    token = set_current_session_key(session)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_module, "_permanent_approved", set())
    monkeypatch.setattr(approval_module, "_session_approved", {})
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None, raising=False)
    approval_floors._observed_review.clear()
    try:
        yield session
    finally:
        reset_current_session_key(token)
        approval_module.clear_session(session)


@pytest.fixture
def guardian(monkeypatch):
    """Guardian stub that always ESCALATEs (so every smart interval reaches a person); records commands."""
    calls = []

    def fake(command, description):
        calls.append(command)
        return "escalate"

    monkeypatch.setattr(approval_smart, "_smart_approve", fake)
    return calls


def _cli_prompt(monkeypatch, *choices):
    seen = []
    it = iter(choices)

    def fake(command, description, **kwargs):
        seen.append(kwargs.get("allow_permanent"))
        return next(it)

    monkeypatch.setattr(approval_module, "prompt_dangerous_approval", fake)
    return seen


def _rule_key(review):
    return approval_floors._RequiredRule(PATTERN, "kubectl on admin", review).key


def _simulate_restart(monkeypatch):
    """Drop in-memory state and reload the persisted allowlist the way a new process does."""
    monkeypatch.setattr(approval_module, "_permanent_approved", set())
    monkeypatch.setattr(approval_module, "_session_approved", {})
    approval_floors._observed_review.clear()
    approval_module.load_permanent_allowlist()


class TestSessionGrantDoesNotReviveAfterRoundTrip:

    def test_cli(self, live_config, isolated_state, guardian, monkeypatch):
        seen = _cli_prompt(monkeypatch, "session", "deny", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        assert _rule_key("smart") in approval_module._session_approved[isolated_state]

        live_config("human")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert _rule_key("smart") not in approval_module._session_approved.get(isolated_state, set()), \
            "the human policy revoked the smart grant instead of hiding it"

        live_config("smart")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL], "the restored smart policy consulted the guardian again"
        assert seen == [True, False, True], "every policy interval produced its own person decision"

    def test_gateway(self, live_config, isolated_state, guardian, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        answers = iter(["session", "deny", "deny"])
        notified = []

        def notify(data):
            notified.append(data["allow_permanent"])
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            live_config("human")
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
            live_config("smart")
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert guardian == [KUBECTL, KUBECTL]
        assert notified == [True, False, True]


class TestPermanentGrantDoesNotReviveAfterRoundTrip:

    def test_cli_across_restart(self, live_config, isolated_state, guardian, monkeypatch):
        seen = _cli_prompt(monkeypatch, "always", "deny", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        _simulate_restart(monkeypatch)
        assert _rule_key("smart") in approval_module._permanent_approved, "Always was persisted to config"

        live_config("human")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        _simulate_restart(monkeypatch)
        assert _rule_key("smart") not in approval_module._permanent_approved, \
            "the human policy removed the smart grant from the persisted allowlist"

        live_config("smart")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL]
        assert seen == [True, False, True]

    def test_gateway_across_restart(self, live_config, isolated_state, guardian, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        answers = iter(["always", "deny", "deny"])
        notified = []

        def notify(data):
            notified.append(data["allow_permanent"])
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            _simulate_restart(monkeypatch)
            live_config("human")
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
            _simulate_restart(monkeypatch)
            live_config("smart")
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert guardian == [KUBECTL, KUBECTL]
        assert notified == [True, False, True]


def test_same_policy_keeps_its_grant(live_config, isolated_state, guardian, monkeypatch):
    """Revocation fires on a policy *change* only: re-reading an unchanged rule is not a transition."""
    seen = _cli_prompt(monkeypatch, "session")
    assert check_all_command_guards(KUBECTL, "local")["approved"] is True
    live_config("smart")  # rewrite, same policy
    assert check_all_command_guards(KUBECTL, "local")["approved"] is True
    assert seen == [True] and guardian == [KUBECTL]


UNRELATED = "ls -la /tmp"


class TestRoundTripWithNoMatchingCommandInBetween:
    """The human interval is observed by an unrelated command only: the transition, not a match, revokes."""

    def test_session_grant_cli(self, live_config, isolated_state, guardian, monkeypatch):
        seen = _cli_prompt(monkeypatch, "session", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True

        live_config("human")
        assert check_all_command_guards(UNRELATED, "local")["approved"] is True
        assert _rule_key("smart") not in approval_module._session_approved.get(isolated_state, set()), \
            "seeing the rule under human revoked the smart grant without a matching command"

        live_config("smart")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL] and seen == [True, True]

    def test_session_grant_gateway(self, live_config, isolated_state, guardian, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        answers = iter(["session", "deny"])
        notified = []

        def notify(data):
            notified.append(data["allow_permanent"])
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            live_config("human")
            assert check_all_command_guards(UNRELATED, "local")["approved"] is True
            live_config("smart")
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert guardian == [KUBECTL, KUBECTL] and notified == [True, True]

    def test_permanent_grant_across_restart_cli(self, live_config, isolated_state, guardian, monkeypatch):
        """The policy flips while the process is down: the allowlist load at startup is the observation."""
        seen = _cli_prompt(monkeypatch, "always", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True

        live_config("human")
        _simulate_restart(monkeypatch)
        assert _rule_key("smart") not in approval_module._permanent_approved, \
            "loading the allowlist under the human policy dropped the smart Always"
        assert _rule_key("smart") not in (yaml.safe_load(live_config.path.read_text(encoding="utf-8")).get("command_allowlist") or []), \
            "and rewrote config.yaml without it"

        live_config("smart")
        _simulate_restart(monkeypatch)
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL] and seen == [True, True]

    def test_permanent_grant_across_restart_gateway(self, live_config, isolated_state, guardian, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        answers = iter(["always", "deny"])
        notified = []

        def notify(data):
            notified.append(data["allow_permanent"])
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            live_config("human")
            _simulate_restart(monkeypatch)
            live_config("smart")
            _simulate_restart(monkeypatch)
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert guardian == [KUBECTL, KUBECTL] and notified == [True, True]


class TestRoundTripUnderBypass:
    """The human interval is only ever seen while a bypass is active — session ``/yolo`` or
    ``approvals.mode: off`` — which returns before any rule matching. The transition must still be
    observed there (PR #106779 review, round 4): a bypass hides the interval, it must not hide the
    revocation. Every case: grant under smart, bypass on, flip to human, run a guarded command while
    bypassed, flip back to smart, bypass off, match again — and the match needs a fresh decision."""

    def _yolo_interval(self, live_config, session, check=check_all_command_guards):
        approval_module.enable_session_yolo(session)
        live_config("human")
        assert check(KUBECTL, "local")["approved"] is True, "yolo bypassed the human prompt"
        live_config("smart")
        approval_module.disable_session_yolo(session)

    def _mode_off_interval(self, live_config):
        live_config("human", mode="off")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True, "mode: off bypassed the prompt"
        live_config("smart", mode="smart")

    def test_session_grant_yolo_cli(self, live_config, isolated_state, guardian, monkeypatch):
        seen = _cli_prompt(monkeypatch, "session", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        self._yolo_interval(live_config, isolated_state)
        assert _rule_key("smart") not in approval_module._session_approved.get(isolated_state, set()), \
            "the human interval was observed through the yolo bypass"
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL] and seen == [True, True]

    def test_session_grant_yolo_gateway(self, live_config, isolated_state, guardian, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        answers = iter(["session", "deny"])
        notified = []

        def notify(data):
            notified.append(data["allow_permanent"])
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            self._yolo_interval(live_config, isolated_state)
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert guardian == [KUBECTL, KUBECTL] and notified == [True, True]

    def test_session_grant_yolo_pattern_gate(self, live_config, isolated_state, guardian, monkeypatch):
        """``check_dangerous_command`` has the same yolo early return."""
        seen = _cli_prompt(monkeypatch, "session", "deny")
        assert check_dangerous_command(KUBECTL, "local")["approved"] is True
        self._yolo_interval(live_config, isolated_state, check=check_dangerous_command)
        assert check_dangerous_command(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL] and seen == [True, True]

    def test_session_grant_mode_off_cli(self, live_config, isolated_state, guardian, monkeypatch):
        seen = _cli_prompt(monkeypatch, "session", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        self._mode_off_interval(live_config)
        assert _rule_key("smart") not in approval_module._session_approved.get(isolated_state, set()), \
            "the human interval was observed through the mode: off bypass"
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL] and seen == [True, True]

    def test_permanent_grant_yolo_across_restart_cli(self, live_config, isolated_state, guardian, monkeypatch):
        seen = _cli_prompt(monkeypatch, "always", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        self._yolo_interval(live_config, isolated_state)
        assert _rule_key("smart") not in (yaml.safe_load(live_config.path.read_text(encoding="utf-8")).get("command_allowlist") or []), \
            "the persisted Always was dropped while yolo was active"
        _simulate_restart(monkeypatch)
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert guardian == [KUBECTL, KUBECTL] and seen == [True, True]

    def test_permanent_grant_mode_off_across_restart_gateway(self, live_config, isolated_state, guardian, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        answers = iter(["always", "deny"])
        notified = []

        def notify(data):
            notified.append(data["allow_permanent"])
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            self._mode_off_interval(live_config)
            _simulate_restart(monkeypatch)
            assert _rule_key("smart") not in approval_module._permanent_approved
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert guardian == [KUBECTL, KUBECTL] and notified == [True, True]
