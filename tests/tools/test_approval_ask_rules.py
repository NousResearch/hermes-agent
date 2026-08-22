"""Tests for user-defined ask rules (approvals.ask in config.yaml).

approvals.ask is the middle tier of the user-editable rule lists: deny
blocks unconditionally, command_allowlist auto-approves, and ask forces the
human-approval prompt BEFORE the --yolo / /yolo / mode=off bypass — the
user saying "always show me this, even when approvals are bypassed".
"""

import pytest

from tools import approval as mod


@pytest.fixture
def ask_config(monkeypatch):
    """Install an approvals.ask list into the config and return a setter."""

    state = {"config": {"mode": "manual", "ask": []}}

    def set_ask(patterns, **extra):
        state["config"] = {
            "mode": extra.pop("mode", "manual"),
            "deny": extra.pop("deny", []),
            "ask": list(patterns),
            **{k: v for k, v in extra.items() if k != "mode"},
        }

    monkeypatch.setattr(mod, "_get_approval_config", lambda: state["config"])
    return set_ask


@pytest.fixture
def clean_env(monkeypatch):
    """Non-interactive, non-gateway, non-cron, non-yolo baseline."""
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION",
                "HERMES_CRON_SESSION", "HERMES_INTERACTIVE",
                "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)


class TestMatchUserAskRule:
    def test_empty_list_is_noop(self, ask_config):
        ask_config([])
        assert mod._match_user_ask_rule("ssh host") is None

    def test_missing_key_is_noop(self, monkeypatch):
        monkeypatch.setattr(mod, "_get_approval_config",
                            lambda: {"mode": "manual"})
        assert mod._match_user_ask_rule("ssh host") is None

    def test_config_load_failure_fails_open(self, monkeypatch):
        def boom():
            raise RuntimeError("config unavailable")
        monkeypatch.setattr(mod, "_get_approval_config", boom)
        assert mod._match_user_ask_rule("ssh host") is None

    def test_glob_match(self, ask_config):
        ask_config(["ssh *"])
        assert mod._match_user_ask_rule(
            "ssh -i key ubuntu@host 'echo hi'") == "ssh *"
        assert mod._match_user_ask_rule("ls -la /tmp") is None

    def test_quote_obfuscation_still_matches(self, ask_config):
        ask_config(["ssh *"])
        assert mod._match_user_ask_rule('ss""h host') is not None


class TestAskPromptsUnderBypass:
    def test_ask_beats_yolo_frozen(self, ask_config, clean_env, monkeypatch):
        ask_config(["ssh *"], mode="off")
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)

        result = mod.check_dangerous_command("ssh ubuntu@host 'echo hi'",
                                             "local")
        # The gate ran and, with no human present, failed closed — the
        # command was NOT silently approved by the yolo bypass.
        assert result["approved"] is False
        assert "approvals.ask" in (result.get("message") or "")

    def test_ask_beats_mode_off(self, ask_config, clean_env):
        ask_config(["scp *"], mode="off")

        result = mod.check_dangerous_command("scp f.txt ubuntu@host:/tmp/",
                                             "local")
        assert result["approved"] is False
        assert "approvals.ask" in result["message"]

    def test_non_matching_command_allowed_under_off(
            self, ask_config, clean_env):
        ask_config(["ssh *"], mode="off")
        result = mod.check_dangerous_command("echo hello", "local")
        assert result == {"approved": True, "message": None}


class TestDenyBeatsAsk:
    def test_deny_wins_when_both_match(self, monkeypatch, clean_env):
        state = {"config": {"mode": "off", "deny": ["ssh *"],
                            "ask": ["ssh *"]}}
        monkeypatch.setattr(mod, "_get_approval_config",
                            lambda: state["config"])

        result = mod.check_dangerous_command("ssh host", "local")
        assert result["approved"] is False
        assert result.get("user_deny") is True


class TestNoHumanFailsClosed:
    def test_bare_script_blocked_not_autoapproved(
            self, ask_config, clean_env):
        ask_config(["ssh *"])
        result = mod.check_dangerous_command("ssh host", "local")
        assert result["approved"] is False
        assert "no interactive user or gateway" in result["message"]
