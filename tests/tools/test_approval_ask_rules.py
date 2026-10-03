"""User-defined ask rules (``approvals.ask``): the tier between ``deny`` and the allowlist.

A match prompts instead of blocking, keeps prompting under the yolo / mode=off bypass, and fails
closed where nobody can answer. Inspired by Claude Cowork's "Restrict to Ask" tool policy, which
keeps prompting inside "Skip all approvals" mode.
"""

import pytest

from tools import approval as mod
from tools import approval_context


@pytest.fixture
def approvals_config(monkeypatch):
    state = {"config": {"mode": "manual", "ask": []}}

    def set_config(**cfg):
        state["config"] = {"mode": "manual", **cfg}

    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: state["config"])
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                "HERMES_INTERACTIVE", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    return set_config


def test_ask_rule_prompts_through_every_bypass_and_persists_per_rule(approvals_config, monkeypatch):
    """mode=off + process yolo + a permanent-allowlist hit: a matching command still reaches the
    human prompt (and only that command does); a session grant on one rule covers that rule alone."""
    approvals_config(mode="off", ask=["ssh *"])
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setattr(mod, "_command_matches_permanent_allowlist", lambda command: True)
    monkeypatch.setattr(mod, "_tirith_scan", lambda command: {"action": "allow", "findings": []})
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    prompts, answer = [], {"choice": "deny"}

    def human(command, description, **kwargs):
        prompts.append((command, description))
        return answer["choice"]

    denied = mod.check_all_command_guards("ssh prod uptime", "local", approval_callback=human)
    assert denied["approved"] is False and denied["pattern_key"] == "ask_rule:ssh *"
    assert prompts == [("ssh prod uptime", "matches your approvals.ask rule 'ssh *'")]

    # Non-matching commands keep the bypass the user asked for — even dangerous ones.
    assert mod.check_all_command_guards("rm -rf build/", "local", approval_callback=human)["approved"] is True
    assert len(prompts) == 1

    answer["choice"] = "session"
    assert mod.check_all_command_guards("ssh prod uptime", "local", approval_callback=human)["approved"] is True
    assert mod.check_all_command_guards("ssh prod reboot", "local", approval_callback=human)["approved"] is True
    assert len(prompts) == 2  # the per-rule grant covers the second ssh without a new prompt
    mod.clear_session(mod.get_current_session_key())


def test_ask_rule_blocks_where_no_human_can_answer(approvals_config, monkeypatch):
    """cron_mode: approve waives the heuristic prompt, never an explicit ask rule; the same rule in
    an isolated container prompts (deny-style intent about what the agent may DO, not reach)."""
    approvals_config(ask=["ssh *"])
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    blocked = mod.check_all_command_guards("ssh prod uptime", "local")
    assert blocked["approved"] is False and blocked.get("user_ask") is True
    assert "approvals.ask rule 'ssh *'" in blocked["message"]
    assert mod.check_all_command_guards("rm -rf build/", "local")["approved"] is True  # cron approve-mode intact

    assert mod.check_all_command_guards("ssh prod uptime", "modal").get("user_ask") is True
    assert mod.check_all_command_guards("rm -rf build/", "modal")["approved"] is True
