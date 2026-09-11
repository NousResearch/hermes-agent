"""Tests for ``approvals.command_approval_required`` (#5528) and the per-rule ``review`` field.

Operator rules that force a terminal command through the approval flow even when no built-in
dangerous pattern fires. Contract (from the #56011 review, teknium1):

* ``review: human`` (default, and every bare-string rule): NO smart-approval guardian, NO permanent
  ``[a]lways`` — a person sees every match, in every session, on the CLI and the gateway.
* ``review: smart``: the rule behaves like a built-in dangerous pattern — under ``approvals.mode:
  smart`` the guardian answers first (APPROVE runs, DENY/ESCALATE reach the human), and
  ``[a]lways`` persists.
* The same ``review: smart`` marker works for a plugin ``pre_tool_call`` ``approve`` directive
  through :func:`tools.approval.request_tool_approval`.

The structured-allowlist / ``args_glob`` half of #56011 is out of scope here.

Test structure adapted from ``TestConfigurableCommandPolicy`` in PR #56011 by @konsisumer.
"""

from unittest.mock import patch as mock_patch

import pytest

import tools.approval as approval_module
from tools import approval_context, approval_floors, approval_smart
from tools.approval import check_all_command_guards, check_dangerous_command, request_tool_approval
from tools.approval_context import reset_current_session_key, set_current_session_key

RESTART = "systemctl --user restart hermes-gateway.service"
RESTART_RULE = "systemctl --user restart hermes-gateway*"
KUBECTL = "kubectl --context admin -n vault exec vault-0 -c vault -- du -sh /vault/data"
KUBECTL_RULE = {"pattern": "kubectl *--context[= ]admin*", "description": "kubectl on admin", "review": "smart"}


@pytest.fixture
def approvals_config(monkeypatch):
    """Install an ``approvals`` block; returns a setter ``set(rules, mode=...)``."""
    state = {"config": {"mode": "manual", "command_approval_required": []}}

    def set_rules(rules, mode="manual", **extra):
        state["config"] = {"mode": mode, "command_approval_required": rules, **extra}

    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: state["config"])
    approval_floors._warned_review_values.clear()
    approval_floors._observed_review.clear()
    return set_rules


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    """Fresh session key, empty session/permanent stores, non-yolo, tirith allows, no on-disk allowlist writes."""
    session = "test:approval-required"
    token = set_current_session_key(session)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_module, "_permanent_approved", set())
    monkeypatch.setattr(approval_module, "_session_approved", {})
    monkeypatch.setattr(approval_module, "save_permanent_allowlist", lambda patterns: None)
    monkeypatch.setattr("tools.tirith_security.check_command_security",
                        lambda _: {"action": "allow", "findings": [], "summary": ""})
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None, raising=False)
    try:
        yield session
    finally:
        reset_current_session_key(token)
        approval_module.clear_session(session)


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")


class _Guardian(list):
    """Recorder of ``(command, description)`` guardian calls with a settable verdict."""
    _verdict = "approve"

    def verdict(self, value):
        self._verdict = value


@pytest.fixture
def guardian(monkeypatch):
    """Stub the guardian LLM; returns the recorder list of ``(command, description)`` calls."""
    calls = _Guardian()

    def fake(command, description):
        calls.append((command, description))
        return calls._verdict

    monkeypatch.setattr(approval_smart, "_smart_approve", fake)
    return calls


def _prompt(monkeypatch, *choices):
    """Stub the CLI prompt; returns the recorder of ``allow_permanent`` kwargs it was shown."""
    seen = []
    it = iter(choices)

    def fake(command, description, **kwargs):
        seen.append(kwargs.get("allow_permanent"))
        return next(it)

    monkeypatch.setattr(approval_module, "prompt_dangerous_approval", fake)
    return seen


# --- rule parsing / matching -------------------------------------------------------------------------------------

class TestRuleParsing:
    def test_string_rule_is_human_glob(self, approvals_config):
        approvals_config([RESTART_RULE])
        rule = approval_floors._match_approval_required_rule(RESTART)
        assert rule is not None
        assert (rule.pattern, rule.review, rule.description) == (RESTART_RULE, "human", RESTART_RULE)

    def test_dict_rule_carries_review_and_description(self, approvals_config):
        approvals_config([KUBECTL_RULE])
        rule = approval_floors._match_approval_required_rule(KUBECTL)
        assert rule is not None
        assert (rule.review, rule.description) == ("smart", "kubectl on admin")
        assert rule.key.startswith("approval_required:")

    def test_unknown_review_falls_back_to_human(self, approvals_config, caplog):
        approvals_config([{"pattern": "glab mr merge*", "review": "yolo"}])
        rule = approval_floors._match_approval_required_rule("glab mr merge 638 --yes")
        assert rule is not None and rule.review == "human"
        assert "unknown review" in caplog.text

    @pytest.mark.parametrize("raw", [None, [], "not a list at all", 42, [None, 7, {"description": "no pattern"}]])
    def test_malformed_config_matches_nothing(self, approvals_config, raw):
        approvals_config(raw)
        assert approval_floors._match_approval_required_rule(RESTART) is None

    def test_config_read_failure_matches_nothing(self, monkeypatch):
        def boom():
            raise RuntimeError("config unavailable")
        monkeypatch.setattr(approval_context, "_get_approval_config", boom)
        assert approval_floors._match_approval_required_rule(RESTART) is None

    @pytest.mark.parametrize("command", [
        "cd /srv/app && " + KUBECTL,                      # compound: rule matches the executable segment
        "timeout 40 " + KUBECTL,                          # launcher prefix
        "kubectl --context=admin get pods",               # = vs space, matched by the [= ] class
        'kube""ctl --context admin get pods',             # quote splicing
    ])
    def test_matching_uses_detector_variants(self, approvals_config, command):
        approvals_config([KUBECTL_RULE])
        assert approval_floors._match_approval_required_rule(command) is not None

    def test_first_matching_rule_wins(self, approvals_config):
        approvals_config([{"pattern": "kubectl*", "review": "human"}, KUBECTL_RULE])
        assert approval_floors._match_approval_required_rule(KUBECTL).review == "human"


# --- review: human — the #5528 contract ---------------------------------------------------------------------------

class TestHumanReview:
    def test_rule_overrides_broad_allowlist(self, approvals_config, isolated_state, cli, monkeypatch):
        approvals_config([RESTART_RULE])
        approval_module.load_permanent({"systemctl *"})
        seen = _prompt(monkeypatch, "once")
        result = check_all_command_guards(RESTART, "local")
        assert result["approved"] is True and result.get("user_approved") is True
        assert seen == [False], "the prompt must not offer [a]lways"

    def test_bypasses_smart_approval(self, approvals_config, isolated_state, cli, monkeypatch, guardian):
        approvals_config([RESTART_RULE], mode="smart")
        seen = _prompt(monkeypatch, "once")
        result = check_all_command_guards(RESTART, "local")
        assert result["approved"] is True
        assert guardian == [], "review: human must never consult the guardian"
        assert seen == [False]

    def test_always_is_session_only(self, approvals_config, isolated_state, cli, monkeypatch):
        """A second matching invocation in a NEW session prompts again even after 'always'."""
        approvals_config([RESTART_RULE])
        seen = _prompt(monkeypatch, "always", "deny")
        first = check_all_command_guards(RESTART, "local")
        assert first["approved"] is True
        assert approval_module._permanent_approved == set()
        # Same session: the session grant holds, no re-prompt.
        assert check_all_command_guards(RESTART, "local")["approved"] is True
        # New session: prompted again; the 'deny' answer blocks.
        token = set_current_session_key("test:approval-required:second")
        try:
            second = check_all_command_guards(RESTART, "local")
        finally:
            reset_current_session_key(token)
            approval_module.clear_session("test:approval-required:second")
        assert second["approved"] is False
        assert seen == [False, False]

    def test_permanent_allowlist_entry_for_its_key_is_ignored(self, approvals_config, isolated_state, cli,
                                                              monkeypatch):
        """Even a hand-written command_allowlist entry naming the rule's key cannot silence it."""
        approvals_config([RESTART_RULE])
        rule = approval_floors._match_approval_required_rule(RESTART)
        approval_module.load_permanent({rule.key})
        seen = _prompt(monkeypatch, "deny")
        assert check_all_command_guards(RESTART, "local")["approved"] is False
        assert seen == [False]

    def test_gateway_payload_hides_always(self, approvals_config, isolated_state, monkeypatch):
        approvals_config([RESTART_RULE], mode="smart")
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setattr(approval_smart, "_smart_approve",
                            lambda *_: pytest.fail("guardian must not run for review: human"))
        notified = []

        def notify(data):
            notified.append(data)
            approval_module.resolve_gateway_approval(isolated_state, "always")

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            result = check_all_command_guards(RESTART, "local")
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert result["approved"] is True
        assert notified[0]["allow_permanent"] is False
        assert notified[0]["allow_session"] is True
        assert approval_module._permanent_approved == set()

    def test_pattern_gate_prompts_without_always(self, approvals_config, isolated_state, cli, monkeypatch):
        """``check_dangerous_command`` (the pattern-only gate) honours the rule too."""
        approvals_config([RESTART_RULE])
        seen = _prompt(monkeypatch, "always")
        assert check_dangerous_command(RESTART, "local")["approved"] is True
        assert seen == [False]
        assert approval_module._permanent_approved == set()

    def test_cron_deny_blocks_on_rule(self, approvals_config, isolated_state, monkeypatch):
        approvals_config([RESTART_RULE], cron_mode="deny")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        result = check_all_command_guards(RESTART, "local")
        assert result["approved"] is False
        assert "approval-required rule" in result["message"]


# --- review: smart — guardian first, human on ESCALATE/DENY, Always allowed ----------------------------------------

class TestSmartReview:
    def test_guardian_approve_runs_without_prompt(self, approvals_config, isolated_state, cli, monkeypatch,
                                                  guardian):
        approvals_config([KUBECTL_RULE], mode="smart")
        monkeypatch.setattr(approval_module, "prompt_dangerous_approval",
                            lambda *a, **k: pytest.fail("guardian APPROVE must not prompt"))
        result = check_all_command_guards(KUBECTL, "local")
        assert result["approved"] is True and result.get("smart_approved") is True
        assert len(guardian) == 1
        assert "kubectl on admin" in guardian[0][1], "the rule description reaches the guardian"

    def test_guardian_escalate_prompts_human(self, approvals_config, isolated_state, cli, monkeypatch, guardian):
        approvals_config([KUBECTL_RULE], mode="smart")
        guardian.verdict("escalate")
        seen = _prompt(monkeypatch, "once")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        assert seen == [True], "[a]lways stays available for review: smart"

    def test_smart_session_grant_does_not_survive_tightening_to_human(self, approvals_config, isolated_state, cli,
                                                                      monkeypatch, guardian):
        """A session grant taken under ``review: smart`` must not satisfy the same pattern after the live
        config tightens it to ``review: human``: the stricter policy asks a person again."""
        approvals_config([KUBECTL_RULE], mode="smart")
        guardian.verdict("escalate")
        seen = _prompt(monkeypatch, "session", "deny")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True, "same policy: grant holds"
        approvals_config([{**KUBECTL_RULE, "review": "human"}], mode="smart")
        guardian.verdict("approve")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        assert seen == [True, False], "the human policy prompted again, without [a]lways"
        assert guardian[-1][0] == KUBECTL and len(guardian) == 1, "the guardian never sees a review: human rule"

    def test_smart_session_grant_does_not_survive_tightening_on_gateway(self, approvals_config, isolated_state,
                                                                        monkeypatch, guardian):
        approvals_config([KUBECTL_RULE], mode="smart")
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        guardian.verdict("escalate")
        answers = iter(["session", "deny"])
        notified = []

        def notify(data):
            notified.append(data)
            approval_module.resolve_gateway_approval(isolated_state, next(answers))

        approval_module.register_gateway_notify(isolated_state, notify)
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
            approvals_config([{**KUBECTL_RULE, "review": "human"}], mode="smart")
            assert check_all_command_guards(KUBECTL, "local")["approved"] is False
        finally:
            approval_module.unregister_gateway_notify(isolated_state)
        assert [n["allow_permanent"] for n in notified] == [True, False]

    def test_guardian_deny_lets_owner_override_once_only(self, approvals_config, isolated_state, cli, monkeypatch,
                                                         guardian):
        approvals_config([KUBECTL_RULE], mode="smart")
        guardian.verdict("deny")
        seen = _prompt(monkeypatch, "once")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        assert seen == [False], "a smart DENY override is one operation, never persistent"

    def test_always_persists(self, approvals_config, isolated_state, cli, monkeypatch, guardian):
        approvals_config([KUBECTL_RULE], mode="smart")
        guardian.verdict("escalate")
        _prompt(monkeypatch, "always")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        rule = approval_floors._match_approval_required_rule(KUBECTL)
        assert rule.key in approval_module._permanent_approved
        # New session, no guardian, no prompt: the permanent grant applies.
        guardian.clear()
        monkeypatch.setattr(approval_module, "prompt_dangerous_approval",
                            lambda *a, **k: pytest.fail("permanently approved rule must not prompt"))
        token = set_current_session_key("test:approval-required:second")
        try:
            assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        finally:
            reset_current_session_key(token)
            approval_module.clear_session("test:approval-required:second")
        assert guardian == []

    def test_manual_mode_prompts_human_directly(self, approvals_config, isolated_state, cli, monkeypatch,
                                                guardian):
        approvals_config([KUBECTL_RULE], mode="manual")
        seen = _prompt(monkeypatch, "once")
        assert check_all_command_guards(KUBECTL, "local")["approved"] is True
        assert guardian == [] and seen == [True]

    def test_human_rule_in_same_prompt_disables_guardian(self, approvals_config, isolated_state, cli, monkeypatch,
                                                         guardian):
        """A command hitting a review: human rule AND a built-in dangerous pattern: the human wins."""
        approvals_config([{"pattern": "rm -rf /srv/app/*", "review": "human"}], mode="smart")
        seen = _prompt(monkeypatch, "once")
        result = check_all_command_guards("rm -rf /srv/app/cache", "local")
        assert result["approved"] is True
        assert guardian == []
        assert seen == [False]


# --- plugin pre_tool_call approve directives ------------------------------------------------------------------------

    def test_pattern_gate_consults_guardian_first(self, approvals_config, isolated_state, cli, monkeypatch,
                                                  guardian):
        """``check_dangerous_command`` (the pattern-only gate) runs a ``review: smart`` rule through the
        guardian like a built-in dangerous pattern: APPROVE runs it with no prompt."""
        approvals_config([KUBECTL_RULE], mode="smart")
        monkeypatch.setattr(approval_module, "prompt_dangerous_approval",
                            lambda *a, **k: pytest.fail("guardian APPROVE must not prompt"))
        result = check_dangerous_command(KUBECTL, "local")
        assert result["approved"] is True and result.get("smart_approved") is True
        assert len(guardian) == 1 and "kubectl on admin" in guardian[0][1]

    def test_pattern_gate_escalate_prompts_with_always(self, approvals_config, isolated_state, cli, monkeypatch,
                                                       guardian):
        approvals_config([KUBECTL_RULE], mode="smart")
        guardian.verdict("escalate")
        seen = _prompt(monkeypatch, "once")
        assert check_dangerous_command(KUBECTL, "local")["approved"] is True
        assert len(guardian) == 1 and seen == [True]


class TestPluginReview:
    def test_default_review_never_consults_guardian(self, approvals_config, isolated_state, cli, monkeypatch,
                                                    guardian):
        approvals_config([], mode="smart")
        seen = _prompt(monkeypatch, "once")
        res = request_tool_approval("terminal", "kubectl mutation on admin", rule_key="diia:kubectl")
        assert res["approved"] is True
        assert guardian == [] and seen == [True]

    def test_smart_review_guardian_approve_runs(self, approvals_config, isolated_state, cli, monkeypatch, guardian):
        approvals_config([], mode="smart")
        monkeypatch.setattr(approval_module, "prompt_dangerous_approval",
                            lambda *a, **k: pytest.fail("guardian APPROVE must not prompt"))
        res = request_tool_approval("terminal", "kubectl mutation on admin", rule_key="diia:kubectl",
                                    review="smart")
        assert res["approved"] is True and res.get("smart_approved") is True
        assert guardian[0][1] == "kubectl mutation on admin"

    def test_smart_review_guardian_escalate_prompts(self, approvals_config, isolated_state, cli, monkeypatch,
                                                    guardian):
        approvals_config([], mode="smart")
        guardian.verdict("escalate")
        seen = _prompt(monkeypatch, "deny")
        res = request_tool_approval("terminal", "kubectl mutation on admin", rule_key="diia:kubectl",
                                    review="smart")
        assert res["approved"] is False and seen == [True]

    def test_smart_review_needs_smart_mode(self, approvals_config, isolated_state, cli, monkeypatch, guardian):
        approvals_config([], mode="manual")
        seen = _prompt(monkeypatch, "once")
        res = request_tool_approval("terminal", "why", rule_key="k", review="smart")
        assert res["approved"] is True and guardian == [] and seen == [True]

    def test_hook_directive_carries_review_to_gate(self, monkeypatch):
        from hermes_cli.plugins_pre_tool_call import resolve_pre_tool_block

        seen = {}
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kw: [
            {"action": "approve", "message": "why", "rule_key": "k", "review": "SMART"}])

        def _approve(tool_name, reason, **kwargs):
            seen.update(kwargs)
            return {"approved": True, "message": None}

        monkeypatch.setattr("tools.approval.request_tool_approval", _approve)
        assert resolve_pre_tool_block("terminal", {}) is None
        assert seen == {"rule_key": "k", "review": "smart"}

    @pytest.mark.parametrize("review", [None, "", "guardian", 7])
    def test_hook_directive_unknown_review_is_human(self, monkeypatch, review):
        from hermes_cli.plugins_pre_tool_call import resolve_pre_tool_block

        seen = {}
        directive = {"action": "approve", "message": "why"}
        if review is not None:
            directive["review"] = review
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kw: [directive])
        monkeypatch.setattr("tools.approval.request_tool_approval",
                            lambda tool_name, reason, **kw: seen.update(kw) or {"approved": True, "message": None})
        assert resolve_pre_tool_block("terminal", {}) is None
        assert seen["review"] == "human"


# --- hermes approvals test (dry run) --------------------------------------------------------------------------------

class TestDryRun:
    def test_rule_reported_as_ask(self, approvals_config, isolated_state):
        from hermes_cli.approvals_test import evaluate_command
        approvals_config([RESTART_RULE, KUBECTL_RULE])
        approval_module.load_permanent({"systemctl *"})
        with mock_patch.object(approval_module, "load_permanent_allowlist", lambda: set()):
            human = evaluate_command(RESTART)
            smart = evaluate_command(KUBECTL)
        assert human["verdict"] == "ask-approval" and human["rule"] == RESTART_RULE
        assert "never the smart guardian" in human["detail"]
        assert smart["verdict"] == "ask-approval" and "guardian first" in smart["detail"]
