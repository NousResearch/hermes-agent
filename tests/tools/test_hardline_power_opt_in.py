"""Opt-in hardline categories for host reboot / shutdown (approvals.allow_reboot /
approvals.allow_shutdown). Both default false so upstream behaviour is unchanged; enabling one
lifts only that category off the unconditional floor and the command still needs approval."""

import pytest

from tools import approval_context
from tools.approval import check_all_command_guards, disable_session_yolo, enable_session_yolo
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_detection import (
    HARDLINE_OPT_IN_SETTINGS, detect_dangerous_command, detect_hardline_command, hardline_opt_in_setting)

REBOOT_CMDS = [
    "reboot", "sudo reboot", "reboot -f", "nohup reboot", "ls; reboot",
    "shutdown -r now", "sudo shutdown -r +5", "shutdown --reboot now", "shutdown -Fr now",
    "systemctl reboot", "sudo systemctl --force reboot", "systemctl kexec",
    "init 6", "telinit 6",
]
SHUTDOWN_CMDS = [
    "poweroff", "sudo poweroff", "halt", "halt -p", "shutdown", "shutdown now",
    "shutdown -h now", "sudo shutdown -P +1", "shutdown -H now", "shutdown --poweroff now",
    "systemctl poweroff", "systemctl halt", "sudo systemctl -i poweroff",
    "init 0", "telinit 0",
]
# Could plausibly halt OR reboot: blocked unless both categories are enabled.
AMBIGUOUS_CMDS = ["shutdown -rh now", "reboot -p", "poweroff --reboot"]
OTHER_HARDLINE = [
    "rm -rf /", "sudo rm -rf --no-preserve-root /", "rm -rf /etc", "rm -rf ~",
    "mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=/dev/sda", "cat img > /dev/nvme0n1",
    ":(){ :|:& };:", "kill -9 -1",
]
BOTH = {"reboot", "shutdown"}


@pytest.mark.parametrize("cmd", REBOOT_CMDS + SHUTDOWN_CMDS + AMBIGUOUS_CMDS)
def test_default_blocks_every_power_command(cmd):
    assert detect_hardline_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", REBOOT_CMDS)
def test_reboot_commands_classified_as_reboot(cmd):
    _, desc = detect_hardline_command(cmd)
    assert hardline_opt_in_setting(desc) == "approvals.allow_reboot", (cmd, desc)
    assert detect_hardline_command(cmd, {"reboot"}) == (False, None)
    assert detect_hardline_command(cmd, {"shutdown"})[0]


@pytest.mark.parametrize("cmd", SHUTDOWN_CMDS)
def test_shutdown_commands_classified_as_shutdown(cmd):
    _, desc = detect_hardline_command(cmd)
    assert hardline_opt_in_setting(desc) == "approvals.allow_shutdown", (cmd, desc)
    assert detect_hardline_command(cmd, {"shutdown"}) == (False, None)
    assert detect_hardline_command(cmd, {"reboot"})[0]


def test_shutdown_dash_r_is_reboot_not_shutdown():
    assert detect_hardline_command("shutdown -r now")[1] == "system reboot"
    assert detect_hardline_command("shutdown -r now", {"reboot"}) == (False, None)


@pytest.mark.parametrize("cmd", AMBIGUOUS_CMDS)
def test_ambiguous_power_commands_need_both(cmd):
    assert detect_hardline_command(cmd, {"reboot"})[0]
    assert detect_hardline_command(cmd, {"shutdown"})[0]
    assert detect_hardline_command(cmd, BOTH) == (False, None)


@pytest.mark.parametrize("cmd", REBOOT_CMDS + SHUTDOWN_CMDS + AMBIGUOUS_CMDS)
def test_opted_in_power_commands_are_still_dangerous(cmd):
    """Lifted off the floor, they must land in the normal approval flow, not run silently."""
    assert detect_dangerous_command(cmd)[0], cmd


@pytest.mark.parametrize("cmd", OTHER_HARDLINE)
def test_other_hardline_rules_unaffected_by_opt_ins(cmd):
    assert detect_hardline_command(cmd, BOTH)[0], cmd
    assert hardline_opt_in_setting(detect_hardline_command(cmd, BOTH)[1]) is None


@pytest.mark.parametrize("cmd", ["echo reboot", "git commit -m 'shutdown -h now later'",
                                 "grep -n reboot /var/log/syslog", "man shutdown"])
def test_prose_still_not_power_commands(cmd):
    assert detect_hardline_command(cmd) == (False, None)
    assert not detect_dangerous_command(cmd)[0]


def test_config_defaults_are_false():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["approvals"]["allow_reboot"] is False
    assert DEFAULT_CONFIG["approvals"]["allow_shutdown"] is False
    assert set(HARDLINE_OPT_IN_SETTINGS.values()) == {"approvals.allow_reboot", "approvals.allow_shutdown"}


@pytest.mark.parametrize("cfg, expected", [
    ({}, frozenset()),
    ({"allow_reboot": True}, frozenset({"reboot"})),
    ({"allow_shutdown": "true"}, frozenset({"shutdown"})),
    ({"allow_reboot": True, "allow_shutdown": True}, frozenset(BOTH)),
    ({"allow_reboot": False, "allow_shutdown": "no"}, frozenset()),
])
def test_opt_ins_read_from_approvals_config(monkeypatch, cfg, expected):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
    assert approval_context._get_hardline_opt_ins() == expected


# ---- Full guard chain -------------------------------------------------------------------------

@pytest.fixture
def guard_env(monkeypatch):
    for var in ("HERMES_YOLO_MODE", "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
                "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    token = set_current_session_key("power_opt_in_test")
    disable_session_yolo("power_opt_in_test")

    def configure(**cfg):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
    try:
        yield configure
    finally:
        disable_session_yolo("power_opt_in_test")
        reset_current_session_key(token)


@pytest.mark.parametrize("cmd", ["sudo reboot", "shutdown -h now"])
def test_default_guard_chain_hard_blocks_with_setting_hint(guard_env, cmd):
    guard_env()
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is False and result.get("hardline") is True
    setting = "approvals.allow_reboot" if "reboot" in cmd else "approvals.allow_shutdown"
    assert setting in result["message"]


def test_yolo_still_cannot_bypass_power_floor_by_default(guard_env):
    guard_env(mode="off")
    enable_session_yolo("power_opt_in_test")
    for cmd in ("reboot", "poweroff"):
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False and result.get("hardline") is True


def test_reboot_enabled_reaches_approval_flow_but_poweroff_blocked(guard_env, monkeypatch):
    """With allow_reboot, `reboot` leaves the floor but is NOT auto-run: a cron session with the
    default cron_mode=deny denies it through the normal unattended flow (not the hardline floor).
    poweroff stays hardline-blocked."""
    guard_env(allow_reboot=True, mode="manual")
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False
    assert not result.get("hardline")
    assert "approvals.allow" not in result.get("message", "")
    blocked = check_all_command_guards("poweroff", "local")
    assert blocked["approved"] is False and blocked.get("hardline") is True
    assert "approvals.allow_shutdown" in blocked["message"]


def test_reboot_enabled_prompts_interactive_user(guard_env, monkeypatch):
    guard_env(allow_reboot=True, mode="manual")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    asked = []

    def deny(command, description, **_kw):
        asked.append((command, description))
        return "deny"
    result = check_all_command_guards("systemctl reboot", "local", approval_callback=deny)
    assert asked and asked[0][0] == "systemctl reboot"
    assert result["approved"] is False and not result.get("hardline")


def test_reboot_enabled_with_yolo_runs(guard_env):
    """Opting in hands the command to the existing approval mode — yolo approves it as usual."""
    guard_env(allow_reboot=True, mode="off")
    assert check_all_command_guards("reboot", "local")["approved"] is True


def test_shutdown_enabled_allows_poweroff_but_blocks_reboot(guard_env):
    guard_env(allow_shutdown=True, mode="off")
    assert check_all_command_guards("poweroff", "local")["approved"] is True
    blocked = check_all_command_guards("shutdown -r now", "local")
    assert blocked["approved"] is False and blocked.get("hardline") is True
    assert "approvals.allow_reboot" in blocked["message"]


@pytest.mark.parametrize("cmd", OTHER_HARDLINE)
def test_other_hardline_blocked_with_both_enabled_and_yolo(guard_env, cmd):
    guard_env(allow_reboot=True, allow_shutdown=True, mode="off")
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is False and result.get("hardline") is True
    assert "approvals.allow" not in result["message"]
