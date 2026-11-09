"""The agent's terminal cannot flip the approvals.* host power keys (allow_reboot, allow_shutdown,
reboot_without_approval, shutdown_without_approval): they decide whether reboot/shutdown leave the
hardline floor, and the config cache is mtime-keyed, so a write would take effect mid-session."""

import json
from unittest import mock

import pytest

from tools import approval, approval_context
from tools.approval import check_all_command_guards, clear_session, disable_session_yolo, enable_session_yolo
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_detection import detect_hardline_command

_DESC = "agent edit of host power approval settings (approvals.allow_reboot/...)"

FLIP_CMDS = [
    "hermes config set approvals.allow_reboot true",
    "hermes config set approvals.reboot_without_approval true",
    "hermes config set approvals.allow_shutdown yes",
    "hermes config set approvals.shutdown_without_approval 1",
    "hermes config unset approvals.allow_reboot",
    "hermes --profile work config set approvals.allow_reboot true",
    "~/.local/bin/hermes config set approvals.allow_reboot true",
    "/usr/local/bin/hermes config set approvals.allow_reboot true",
    "python3 -m hermes_cli.main config set approvals.allow_reboot true",
    "sudo hermes config set approvals.allow_reboot true",
    "ls && hermes config set approvals.allow_reboot true",
    "hermes config set approvals.allow_reboot true; sudo reboot",
    "echo '  allow_reboot: true' >> ~/.hermes/config.yaml",
    "printf 'approvals:\\n  reboot_without_approval: true\\n' | tee -a ~/.hermes/config.yaml",
    "sed -i 's/allow_reboot: false/allow_reboot: true/' ~/.hermes/config.yaml",
    "sed -i 's/allow_shutdown: false/allow_shutdown: true/' $HERMES_HOME/config.yaml",
    "perl -pi -e 's/allow_reboot: false/allow_reboot: true/' ~/.hermes/config.yaml",
    "yq -i '.approvals.allow_reboot = true' ~/.hermes/config.yaml",
    "bash -c 'hermes config set approvals.allow_reboot true'",
]

# Reads, prose and other keys stay out of the rule.
SAFE_CMDS = [
    "hermes config get approvals.allow_reboot",
    "hermes config show",
    "hermes config set approvals.mode smart",
    "grep allow_reboot ~/.hermes/config.yaml",
    "cat ~/.hermes/config.yaml",
    "echo 'set approvals.allow_reboot in config.yaml yourself'",
    "git commit -m 'docs: hermes config set approvals.allow_reboot true is operator-only'",
    "echo allow_reboot > /tmp/notes.txt",
]


@pytest.mark.parametrize("cmd", FLIP_CMDS)
def test_power_setting_write_is_hardline(cmd):
    assert detect_hardline_command(cmd) == (True, _DESC)


@pytest.mark.parametrize("cmd", FLIP_CMDS)
def test_power_setting_write_stays_hardline_with_every_opt_in(cmd):
    # Opting reboot/shutdown in must not open the settings rule: it is not an opt-in category.
    assert detect_hardline_command(cmd, frozenset({"reboot", "shutdown"})) == (True, _DESC)


@pytest.mark.parametrize("cmd", SAFE_CMDS)
def test_reads_and_prose_are_not_power_setting_writes(cmd):
    assert detect_hardline_command(cmd)[1] != _DESC


@pytest.fixture
def guard_env(monkeypatch):
    for var in ("HERMES_YOLO_MODE", "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
                "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    token = set_current_session_key("power_settings_guard_test")
    disable_session_yolo("power_settings_guard_test")
    clear_session("power_settings_guard_test")
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"*"})
    monkeypatch.setattr(approval, "_smart_verdict", mock.Mock(side_effect=AssertionError("guardian called")))

    def configure(**cfg):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
        monkeypatch.setattr(approval_context, "_get_power_approval_config", lambda: cfg)
    try:
        yield configure
    finally:
        disable_session_yolo("power_settings_guard_test")
        clear_session("power_settings_guard_test")
        reset_current_session_key(token)


@pytest.mark.parametrize("mode", ["manual", "smart", "off"])
def test_guard_chain_blocks_flip_under_yolo_with_a_human_present(guard_env, monkeypatch, mode):
    """Interactive session, yolo on, wildcard allowlist, reboot already opted in: still hard-blocked,
    and the human is never even asked (the floor runs before any approval)."""
    guard_env(mode=mode, allow_reboot=True)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    enable_session_yolo("power_settings_guard_test")
    asked = []

    def approve(command, description, **_kw):
        asked.append(command)
        return "once"
    result = check_all_command_guards(
        "hermes config set approvals.reboot_without_approval true", "local", approval_callback=approve)
    assert result["approved"] is False and result.get("hardline") is True
    assert not asked


def test_terminal_tool_refuses_and_never_runs_the_flip(guard_env, monkeypatch):
    """End to end through the terminal tool's own approval step: the command is rejected before
    any environment executes it, so config.yaml is never touched."""
    from tools import terminal_tool

    guard_env(mode="off")
    enable_session_yolo("power_settings_guard_test")
    with pytest.raises(terminal_tool._Rejected) as rejected:
        terminal_tool._run_approval_guards(
            "hermes config set approvals.allow_reboot true", "local", {}, force=False)
    payload = json.dumps(json.loads(rejected.value.args[0]))
    assert "BLOCKED" in payload and "host power approval settings" in payload


def test_hardline_message_names_setting_without_a_command_to_flip_it(guard_env):
    guard_env()
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False and result.get("hardline") is True
    assert "approvals.allow_reboot" in result["message"]
    assert "hermes config set" not in result["message"]
