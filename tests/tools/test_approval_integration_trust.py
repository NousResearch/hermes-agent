"""Integration trust levels through the real approval gate: config in, decision out, log on disk."""

import pytest

from agent import integration_trust as trust
from tools import approval as mod
from tools import approval_context

SEND = 'python skills/productivity/google-workspace/scripts/google_api.py gmail send --to anna@firma.pl --subject "Umowa" --body "Tajna treść"'
READ = "python skills/productivity/google-workspace/scripts/google_api.py gmail search 'is:unread'"


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    state = {"cfg": {"mode": "manual"}}
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: state["cfg"])

    def configure(level=None, **extra):
        state["cfg"] = {"mode": "manual", **extra}
        if level:
            state["cfg"]["integrations"] = {"google": level}

    return configure


def ask(answer, seen=None):
    def callback(command, description, **kwargs):
        if seen is not None:
            seen.append(description)
        return answer
    return callback


def test_classification_reads_freely_and_previews_a_write_without_its_body():
    assert trust.classify_command(READ) is None

    action = trust.classify_command(f"cd /tmp && {SEND} && echo done")

    assert action.action == "gmail.send"
    assert "anna@firma.pl" in action.describe() and "Umowa" in action.describe()
    assert "Tajna" not in action.describe()
    assert "Wysłanie maila" in action.describe("pl")


@pytest.mark.parametrize("level", ["read", "propose"])
def test_read_and_propose_block_a_write_even_under_yolo_but_never_a_read(gate, monkeypatch, level):
    gate(level)
    monkeypatch.setattr(mod, "_yolo_active", lambda: True)

    blocked = mod.check_all_command_guards(SEND, "local", approval_callback=ask("once"))

    assert blocked["approved"] is False and blocked["integration_block"] is True
    assert ("propose" in blocked["message"]) == (level == "propose")
    assert mod.check_all_command_guards(READ, "local", approval_callback=ask("once"))["approved"] is True


def test_default_is_ask_and_ask_reaches_the_human_even_under_yolo(gate, monkeypatch):
    gate()
    monkeypatch.setattr(mod, "_yolo_active", lambda: True)
    seen = []

    assert mod.check_all_command_guards(SEND, "local", approval_callback=ask("deny", seen))["approved"] is False
    assert seen and "anna@firma.pl" in seen[0]
    assert mod.check_all_command_guards(SEND, "local", approval_callback=ask("once"))["approved"] is True


def test_auto_acts_without_asking(gate):
    gate("auto")

    result = mod.check_all_command_guards(SEND, "local", approval_callback=ask("deny"))

    assert result["approved"] is True


def test_a_write_with_nobody_to_ask_fails_closed(gate, monkeypatch):
    gate()
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    result = mod.check_all_command_guards(SEND, "local", approval_callback=None)

    assert result["approved"] is False


def test_every_outcome_lands_in_the_log_without_the_body(gate, tmp_path):
    gate("ask")
    mod.check_all_command_guards(SEND, "local", approval_callback=ask("deny"))
    mod.check_all_command_guards(SEND, "local", approval_callback=ask("once"))
    gate("read")
    mod.check_all_command_guards(SEND, "local", approval_callback=ask("once"))

    log = trust.read_log()

    assert [entry["decision"] for entry in log] == ["blocked", "approved", "denied"]
    assert "Tajna" not in (tmp_path / trust.LOG_FILE).read_text(encoding="utf-8")


def test_unknown_level_falls_back_to_ask():
    assert trust.trust_level("google", {"integrations": {"google": "yolo!"}}) == "ask"
    assert trust.trust_level("google", {}) == "ask"
