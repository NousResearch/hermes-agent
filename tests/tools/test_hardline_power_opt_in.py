"""Opt-in hardline categories for host reboot / shutdown (approvals.allow_reboot /
approvals.allow_shutdown). Both default false so upstream behaviour is unchanged; enabling one
lifts only that category off the unconditional floor, and the command then needs a live human
approval EVERY time: smart mode, allowlists, session approvals and yolo/off cannot approve it, and
contexts with nobody to ask deny it. No test here ever runs a real power command."""

from unittest import mock

import pytest

from tools import approval_context
from tools import approval
from tools.approval import (
    check_all_command_guards, check_dangerous_command, clear_session, disable_session_yolo, enable_session_yolo)
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_detection import detect_dangerous_command, detect_hardline_command, detect_power_state_command
from tools.approval_detection_power import HARDLINE_OPT_IN_SETTINGS, hardline_opt_in_setting

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
    monkeypatch.setattr(approval_context, "_get_power_approval_config", lambda: cfg)
    assert approval_context._get_hardline_opt_ins() == expected


# ---- Full guard chain -------------------------------------------------------------------------

@pytest.fixture
def guard_env(monkeypatch):
    for var in ("HERMES_YOLO_MODE", "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
                "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    token = set_current_session_key("power_opt_in_test")
    disable_session_yolo("power_opt_in_test")
    clear_session("power_opt_in_test")
    # Isolated permanent allowlist; never the real config.
    monkeypatch.setattr(approval, "_permanent_set", lambda: set())
    # The guardian LLM is never called for real; a test that reaches it fails loudly.
    monkeypatch.setattr(approval, "_smart_verdict", mock.Mock(side_effect=AssertionError("guardian called")))

    def configure(**cfg):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
        monkeypatch.setattr(approval_context, "_get_power_approval_config", lambda: cfg)
    try:
        yield configure
    finally:
        disable_session_yolo("power_opt_in_test")
        clear_session("power_opt_in_test")
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
    """With allow_reboot, `reboot` leaves the floor but is NOT auto-run: a cron session has nobody
    to ask, so the human-only gate denies it (not the hardline floor). poweroff stays hardline."""
    guard_env(allow_reboot=True, mode="manual")
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False
    assert not result.get("hardline")
    assert "requires a live human approval every time" in result["message"]
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


def test_shutdown_enabled_lifts_poweroff_but_blocks_reboot(guard_env):
    guard_env(allow_shutdown=True, mode="manual")
    result = check_all_command_guards("poweroff", "local")
    assert not result.get("hardline")
    blocked = check_all_command_guards("shutdown -r now", "local")
    assert blocked["approved"] is False and blocked.get("hardline") is True
    assert "approvals.allow_reboot" in blocked["message"]


@pytest.mark.parametrize("cmd", OTHER_HARDLINE)
def test_other_hardline_blocked_with_both_enabled_and_yolo(guard_env, cmd):
    guard_env(allow_reboot=True, allow_shutdown=True, mode="off")
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is False and result.get("hardline") is True
    assert "approvals.allow" not in result["message"]


# ---- Human-only gate for opted-in power commands ------------------------------------------------

class _Prompt:
    """Approval-callback stub: records every prompt and answers with a fixed choice."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, command, description, **kwargs):
        self.calls.append((command, description, kwargs))
        return self.answer


@pytest.fixture
def interactive(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")


def _guardian(monkeypatch, verdict):
    """Guardian stub returning *verdict*; the returned mock records whether it was consulted."""
    stub = mock.Mock(return_value=verdict)
    monkeypatch.setattr(approval, "_smart_verdict", stub)
    return stub


@pytest.mark.parametrize("cmd", ["sync; sudo reboot", "foo && reboot", "ls | systemctl reboot",
                                 "echo ok; shutdown -r now", "true || init 6", "x=$(reboot)"])
def test_chained_power_command_is_detected_as_a_whole(cmd):
    assert detect_power_state_command(cmd) is not None, cmd


@pytest.mark.parametrize("cmd", ["echo reboot", "git commit -m 'reboot later'", "man shutdown", "ls"])
def test_prose_is_not_a_power_command(cmd):
    assert detect_power_state_command(cmd) is None


@pytest.mark.parametrize("verdict", ["approve", "escalate", "deny"])
@pytest.mark.parametrize("cmd", ["sync; sudo reboot", "systemctl reboot"])
def test_smart_mode_never_approves_and_always_prompts(guard_env, interactive, monkeypatch, verdict, cmd):
    """The live failure: smart mode + allow_reboot ran `sync; sudo reboot` with no prompt. Even a
    guardian that would APPROVE is not consulted; a human is asked, and their deny blocks."""
    guard_env(allow_reboot=True, mode="smart")
    guardian = _guardian(monkeypatch, verdict)
    prompt = _Prompt("deny")
    result = check_all_command_guards(cmd, "local", approval_callback=prompt)
    assert guardian.call_count == 0
    assert len(prompt.calls) == 1 and prompt.calls[0][0] == cmd
    assert result["approved"] is False and not result.get("smart_approved")
    assert result.get("user_consent") is False


def test_smart_mode_human_once_approves_only_that_call(guard_env, interactive, monkeypatch):
    guard_env(allow_reboot=True, mode="smart")
    _guardian(monkeypatch, "approve")
    prompt = _Prompt("once")
    result = check_all_command_guards("sync; sudo reboot", "local", approval_callback=prompt)
    assert result["approved"] is True and result.get("user_approved") is True
    assert not result.get("smart_approved")


@pytest.mark.parametrize("answer", ["session", "always"])
def test_prompt_is_once_only_and_never_persists(guard_env, interactive, monkeypatch, answer):
    """The prompt offers once/deny only; an older client returning session/always still persists
    nothing, so the next power command asks again."""
    guard_env(allow_reboot=True, mode="manual")
    persisted = []
    monkeypatch.setattr(approval, "_persist_choice", lambda *a: persisted.append(a))
    prompt = _Prompt(answer)
    assert check_all_command_guards("sudo reboot", "local", approval_callback=prompt)["approved"] is True
    kwargs = prompt.calls[0][2]
    assert kwargs.get("allow_permanent") is False and kwargs.get("allow_session") is False
    assert persisted == []
    assert not approval.is_approved("power_opt_in_test", "system reboot")
    check_all_command_guards("sudo reboot", "local", approval_callback=prompt)
    assert len(prompt.calls) == 2


@pytest.mark.parametrize("entry", ["system reboot", "sudo reboot", "*reboot*", "*"])
def test_permanent_allowlist_does_not_pre_approve(guard_env, interactive, monkeypatch, entry):
    guard_env(allow_reboot=True, mode="manual")
    monkeypatch.setattr(approval, "_permanent_set", lambda: {entry})
    prompt = _Prompt("deny")
    result = check_all_command_guards("sudo reboot", "local", approval_callback=prompt)
    assert len(prompt.calls) == 1 and result["approved"] is False
    assert check_dangerous_command("sudo reboot", "local", approval_callback=prompt)["approved"] is False
    assert len(prompt.calls) == 2


def test_session_approval_does_not_pre_approve(guard_env, interactive):
    guard_env(allow_reboot=True, mode="manual")
    approval.approve_session("power_opt_in_test", "system reboot")
    prompt = _Prompt("deny")
    result = check_all_command_guards("sudo reboot", "local", approval_callback=prompt)
    assert len(prompt.calls) == 1 and result["approved"] is False


@pytest.mark.parametrize("mode", ["off", "smart", "manual"])
def test_yolo_and_mode_off_still_require_the_prompt(guard_env, interactive, mode):
    """yolo / approvals.mode=off do not widen the opt-in back: the human is still asked."""
    guard_env(allow_reboot=True, allow_shutdown=True, mode=mode)
    enable_session_yolo("power_opt_in_test")
    for cmd in ("sudo reboot", "poweroff"):
        prompt = _Prompt("deny")
        assert check_all_command_guards(cmd, "local", approval_callback=prompt)["approved"] is False
        assert check_dangerous_command(cmd, "local", approval_callback=prompt)["approved"] is False
        assert len(prompt.calls) == 2


def test_process_yolo_still_requires_the_prompt(guard_env, interactive, monkeypatch):
    guard_env(allow_reboot=True, mode="off")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    prompt = _Prompt("deny")
    assert check_all_command_guards("reboot", "local", approval_callback=prompt)["approved"] is False
    assert len(prompt.calls) == 1


@pytest.mark.parametrize("mode", ["off", "manual"])
def test_yolo_without_a_human_denies(guard_env, mode):
    guard_env(allow_reboot=True, mode=mode)
    enable_session_yolo("power_opt_in_test")
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False and "human approval" in result["message"]


@pytest.mark.parametrize("ctx_env, mode_getter", [
    ("HERMES_CRON_SESSION", "_get_cron_approval_mode"),
])
@pytest.mark.parametrize("cfg_mode", ["approve", "deny"])
def test_cron_denies_even_with_approve_mode(guard_env, monkeypatch, ctx_env, mode_getter, cfg_mode):
    guard_env(allow_reboot=True, mode="smart")
    _guardian(monkeypatch, "approve")
    monkeypatch.setenv(ctx_env, "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")  # a gateway's env leaking into a cron worker
    monkeypatch.setattr(approval_context, mode_getter, lambda: cfg_mode)
    for guard in (check_all_command_guards, check_dangerous_command):
        result = guard("sync; sudo reboot", "local")
        assert result["approved"] is False and not result.get("hardline")
        assert "human approval" in result["message"]


@pytest.mark.parametrize("cfg_mode", ["approve", "deny"])
def test_single_query_denies_even_with_approve_mode(guard_env, monkeypatch, cfg_mode):
    guard_env(allow_reboot=True, mode="manual")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")  # -q exports it for sudo prompts
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: True)
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: cfg_mode)
    prompt = _Prompt("once")
    result = check_all_command_guards("sudo reboot", "local", approval_callback=prompt)
    assert result["approved"] is False and prompt.calls == []


@pytest.mark.parametrize("cfg_mode", ["approve", "deny"])
def test_unattended_platform_denies_even_with_approve_mode(guard_env, monkeypatch, cfg_mode):
    guard_env(allow_shutdown=True, mode="manual")
    monkeypatch.setattr(approval, "_is_unattended_platform_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_get_session_platform", lambda: "webhook")
    monkeypatch.setattr(approval_context, "_get_unattended_approval_mode", lambda: cfg_mode)
    result = check_all_command_guards("poweroff", "local")
    assert result["approved"] is False and "human approval" in result["message"]


def test_headless_without_any_human_channel_denies(guard_env):
    guard_env(allow_reboot=True, mode="manual")
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False and "human approval" in result["message"]


def test_ask_mode_without_notifier_denies_instead_of_parking(guard_env, monkeypatch):
    """No live card can be shown, so it is denied now rather than left as a pending approval a
    later slash command could resolve."""
    guard_env(allow_reboot=True, mode="manual")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval, "_should_fall_through_to_cli_approval", lambda **_k: False)
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False and result.get("status") != "pending_approval"
    assert "human approval" in result["message"]


def test_gateway_card_offers_once_only_and_human_approves(guard_env, monkeypatch):
    guard_env(allow_reboot=True, mode="smart")
    guardian = _guardian(monkeypatch, "approve")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval, "_gateway_notify_cb", lambda _k: (lambda *_a: None))
    seen = {}

    def decide(session_key, notify_cb, data, *, surface):
        seen.update(data)
        return {"choice": "always", "resolved": True}
    monkeypatch.setattr(approval, "_await_gateway_decision", decide)
    persisted = []
    monkeypatch.setattr(approval, "_persist_choice", lambda *a: persisted.append(a))
    result = check_all_command_guards("sync; sudo reboot", "local")
    assert guardian.call_count == 0
    assert seen["allow_permanent"] is False and seen["allow_session"] is False
    assert result["approved"] is True and result.get("user_approved") is True
    assert persisted == []


def test_gateway_deny_blocks(guard_env, monkeypatch):
    guard_env(allow_reboot=True, mode="smart")
    _guardian(monkeypatch, "approve")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval, "_gateway_notify_cb", lambda _k: (lambda *_a: None))
    monkeypatch.setattr(approval, "_await_gateway_decision",
                        lambda *_a, **_k: {"choice": "deny", "resolved": True})
    result = check_all_command_guards("sudo reboot", "local")
    assert result["approved"] is False and result.get("user_consent") is False


@pytest.mark.parametrize("cmd", ["sync; sudo reboot", "poweroff", "foo && shutdown -h now"])
def test_defaults_still_hardline_in_every_mode(guard_env, interactive, monkeypatch, cmd):
    for mode in ("manual", "smart", "off"):
        guard_env(mode=mode)
        _guardian(monkeypatch, "approve")
        prompt = _Prompt("once")
        result = check_all_command_guards(cmd, "local", approval_callback=prompt)
        assert result["approved"] is False and result.get("hardline") is True
        assert prompt.calls == []


def test_non_power_dangerous_command_keeps_normal_smart_flow(guard_env, interactive, monkeypatch):
    """Only power commands are human-only; everything else keeps the guardian path."""
    guard_env(allow_reboot=True, mode="smart")
    guardian = _guardian(monkeypatch, "approve")
    result = check_all_command_guards("rm -rf ./build", "local", approval_callback=_Prompt("deny"))
    assert guardian.call_count == 1 and result.get("smart_approved") is True


def test_approvals_test_reports_human_only(monkeypatch):
    from hermes_cli import approvals_test
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"allow_reboot": True, "mode": "off"})
    monkeypatch.setattr(approval_context, "_get_power_approval_config", lambda: {"allow_reboot": True})
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: set())
    monkeypatch.setattr(approval, "_permanent_set", lambda: {"sync; sudo reboot"})
    verdict = approvals_test.evaluate_command("sync; sudo reboot")
    assert verdict["verdict"] == "ask-approval"
    assert "requires human approval every time (smart/allowlist/yolo cannot approve)" in verdict["detail"]


# ---- Path-prefixed spellings take the same path as the bare name --------------------------------

PATH_PREFIXED_POWER_CMDS = [
    "/sbin/reboot", "sudo /usr/sbin/shutdown -h now", "/usr/bin/systemctl reboot", "sync; /sbin/reboot",
]


@pytest.mark.parametrize("cmd", PATH_PREFIXED_POWER_CMDS)
def test_path_prefixed_power_commands_hit_the_floor_by_default(cmd):
    assert detect_hardline_command(cmd)[0], cmd
    assert detect_power_state_command(cmd) is not None, cmd


@pytest.mark.parametrize("cmd, bare", [
    ("/sbin/reboot", "reboot"), ("sudo /usr/sbin/shutdown -h now", "sudo shutdown -h now"),
    ("/usr/bin/systemctl reboot", "systemctl reboot"), ("sync; /sbin/reboot", "sync; reboot"),
])
def test_path_prefixed_classified_like_bare_name(cmd, bare):
    assert detect_hardline_command(cmd) == detect_hardline_command(bare)
    assert detect_power_state_command(cmd) == detect_power_state_command(bare)


@pytest.mark.parametrize("cmd", PATH_PREFIXED_POWER_CMDS)
def test_default_guard_chain_hard_blocks_path_prefixed(guard_env, cmd):
    guard_env()
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is False and result.get("hardline") is True


def test_path_prefixed_reboot_opted_in_goes_to_human_only_gate(guard_env, interactive, monkeypatch):
    """/sbin/reboot with allow_reboot takes the same human-only gate as bare `reboot`: an
    approving guardian is not consulted, a human is prompted, and their deny blocks."""
    guard_env(allow_reboot=True, mode="smart")
    guardian = _guardian(monkeypatch, "approve")
    prompt = _Prompt("deny")
    result = check_all_command_guards("/sbin/reboot", "local", approval_callback=prompt)
    assert guardian.call_count == 0
    assert len(prompt.calls) == 1 and prompt.calls[0][0] == "/sbin/reboot"
    assert result["approved"] is False and not result.get("hardline") and not result.get("smart_approved")
    assert result.get("user_consent") is False


@pytest.mark.parametrize("cmd", ["/opt/reboot-helper.sh", "ls /sbin/reboot"])
def test_path_lookalikes_are_not_power_commands(cmd):
    assert detect_power_state_command(cmd) is None
    assert detect_hardline_command(cmd) == (False, None)


# ---- Second level: approvals.<cat>_without_approval ------------------------------------------------

def test_no_approval_defaults_are_false():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.approval_detection_power import HARDLINE_NO_APPROVAL_SETTINGS
    assert DEFAULT_CONFIG["approvals"]["reboot_without_approval"] is False
    assert DEFAULT_CONFIG["approvals"]["shutdown_without_approval"] is False
    assert HARDLINE_NO_APPROVAL_SETTINGS == {
        "reboot": "approvals.reboot_without_approval", "shutdown": "approvals.shutdown_without_approval"}


@pytest.mark.parametrize("cfg, expected", [
    ({}, frozenset()),
    ({"reboot_without_approval": True}, frozenset()),
    ({"allow_reboot": True}, frozenset()),
    ({"allow_reboot": True, "reboot_without_approval": True}, frozenset({"reboot"})),
    ({"allow_shutdown": True, "shutdown_without_approval": "true"}, frozenset({"shutdown"})),
    ({"allow_reboot": True, "shutdown_without_approval": True}, frozenset()),
    ({"allow_reboot": True, "allow_shutdown": True, "reboot_without_approval": True,
      "shutdown_without_approval": True}, frozenset(BOTH)),
])
def test_no_approval_categories_need_both_keys(monkeypatch, cfg, expected):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
    monkeypatch.setattr(approval_context, "_get_power_approval_config", lambda: cfg)
    assert approval_context._get_power_no_approval_categories() == expected


@pytest.mark.parametrize("cmd, expected", [
    ("sudo reboot", {"reboot"}), ("poweroff", {"shutdown"}), ("sync; reboot", {"reboot"}),
    ("reboot -p", BOTH), ("shutdown -rh now", BOTH), ("poweroff --reboot", BOTH),
    ("reboot; poweroff", BOTH), ("make && shutdown -r now || halt", BOTH), ("ls", set()),
])
def test_power_state_categories(cmd, expected):
    from tools.approval_detection import detect_power_state_categories
    assert detect_power_state_categories(cmd) == frozenset(expected)


# (allow_<cat>, <cat>_without_approval) -> expected interactive outcome for that category.
_LEVELS = [
    (False, False, "hardline"),
    (False, True, "hardline"),   # the second key alone has no effect
    (True, False, "prompt"),
    (True, True, "run"),
]


def _assert_outcome(result, prompt, outcome):
    if outcome == "hardline":
        assert result["approved"] is False and result.get("hardline") is True
        assert prompt.calls == []
    elif outcome == "prompt":
        assert len(prompt.calls) == 1 and result["approved"] is False and not result.get("hardline")
    else:
        assert prompt.calls == [] and result["approved"] is True and result.get("power_no_approval") is True


@pytest.mark.parametrize("allow, no_approval, outcome", _LEVELS)
@pytest.mark.parametrize("category, cmds", [
    ("reboot", ["sudo reboot", "systemctl reboot", "sync; /sbin/reboot", "foo && shutdown -r now"]),
    ("shutdown", ["poweroff", "shutdown -h now", "ls; systemctl poweroff", "true || init 0"]),
])
def test_every_combination_per_category(guard_env, interactive, monkeypatch, allow, no_approval, outcome,
                                        category, cmds):
    guard_env(**{f"allow_{category}": allow, f"{category}_without_approval": no_approval, "mode": "manual"})
    for cmd in cmds:
        for guard in (check_all_command_guards, check_dangerous_command):
            prompt = _Prompt("deny")
            _assert_outcome(guard(cmd, "local", approval_callback=prompt), prompt, outcome)


@pytest.mark.parametrize("other_allow", [False, True])
@pytest.mark.parametrize("other_no_approval", [False, True])
def test_categories_are_independent(guard_env, interactive, other_allow, other_no_approval):
    """reboot at level 2 never lends anything to shutdown, whatever shutdown's own keys say."""
    guard_env(allow_reboot=True, reboot_without_approval=True,
              allow_shutdown=other_allow, shutdown_without_approval=other_no_approval, mode="manual")
    prompt = _Prompt("deny")
    _assert_outcome(check_all_command_guards("sudo reboot", "local", approval_callback=prompt), prompt, "run")
    expected = dict(((a, n), o) for a, n, o in _LEVELS)[(other_allow, other_no_approval)]
    prompt = _Prompt("deny")
    _assert_outcome(check_all_command_guards("poweroff", "local", approval_callback=prompt), prompt, expected)


@pytest.mark.parametrize("cmd", ["reboot -p", "shutdown -rh now", "reboot; poweroff", "sync && reboot; halt"])
@pytest.mark.parametrize("cfg, outcome", [
    ({"allow_reboot": True, "reboot_without_approval": True}, "hardline"),
    ({"allow_shutdown": True, "shutdown_without_approval": True}, "hardline"),
    ({"allow_reboot": True, "allow_shutdown": True}, "prompt"),
    ({"allow_reboot": True, "allow_shutdown": True, "reboot_without_approval": True}, "prompt"),
    ({"allow_reboot": True, "allow_shutdown": True, "shutdown_without_approval": True}, "prompt"),
    ({"allow_reboot": True, "allow_shutdown": True, "reboot_without_approval": True,
      "shutdown_without_approval": True}, "run"),
])
def test_mixed_category_command_needs_both_categories(guard_env, interactive, cmd, cfg, outcome):
    guard_env(mode="manual", **cfg)
    prompt = _Prompt("deny")
    _assert_outcome(check_all_command_guards(cmd, "local", approval_callback=prompt), prompt, outcome)


def test_no_approval_run_is_logged_at_warning(guard_env, interactive, caplog):
    import logging
    guard_env(allow_reboot=True, reboot_without_approval=True)
    with caplog.at_level(logging.WARNING, logger="tools.approval"):
        assert check_all_command_guards("sync; sudo reboot", "local", approval_callback=_Prompt("deny"))["approved"]
    records = [r for r in caplog.records if "WITHOUT approval" in r.getMessage()]
    assert records and records[0].levelno == logging.WARNING
    assert "sync; sudo reboot" in records[0].getMessage()
    assert "approvals.reboot_without_approval" in records[0].getMessage()


def test_no_approval_skips_guardian_in_smart_mode(guard_env, interactive, monkeypatch):
    guard_env(allow_shutdown=True, shutdown_without_approval=True, mode="smart")
    guardian = _guardian(monkeypatch, "deny")
    prompt = _Prompt("deny")
    result = check_all_command_guards("poweroff", "local", approval_callback=prompt)
    assert result["approved"] is True and guardian.call_count == 0 and prompt.calls == []


def test_gateway_session_runs_without_card(guard_env, monkeypatch):
    guard_env(allow_reboot=True, reboot_without_approval=True)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval, "_await_gateway_decision",
                        mock.Mock(side_effect=AssertionError("card shown")))
    assert check_all_command_guards("sudo reboot", "local")["approved"] is True


_ALL_ON = dict(allow_reboot=True, allow_shutdown=True, reboot_without_approval=True, shutdown_without_approval=True)


@pytest.mark.parametrize("cfg_mode", ["approve", "deny"])
def test_no_approval_still_denied_in_cron(guard_env, monkeypatch, cfg_mode):
    guard_env(mode="off", **_ALL_ON)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: cfg_mode)
    for cmd in ("sync; sudo reboot", "poweroff"):
        for guard in (check_all_command_guards, check_dangerous_command):
            result = guard(cmd, "local")
            assert result["approved"] is False and "human approval" in result["message"]


@pytest.mark.parametrize("cfg_mode", ["approve", "deny"])
def test_no_approval_still_denied_in_single_query(guard_env, monkeypatch, cfg_mode):
    guard_env(**_ALL_ON)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: True)
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: cfg_mode)
    assert check_all_command_guards("sudo reboot", "local", approval_callback=_Prompt("once"))["approved"] is False


def test_no_approval_still_denied_on_unattended_platform(guard_env, monkeypatch):
    guard_env(**_ALL_ON)
    monkeypatch.setattr(approval, "_is_unattended_platform_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_get_session_platform", lambda: "webhook")
    monkeypatch.setattr(approval_context, "_get_unattended_approval_mode", lambda: "approve")
    assert check_all_command_guards("poweroff", "local")["approved"] is False


def test_no_approval_still_denied_headless(guard_env):
    guard_env(**_ALL_ON)
    enable_session_yolo("power_opt_in_test")
    assert check_all_command_guards("sudo reboot", "local")["approved"] is False


def test_no_approval_ask_mode_without_notifier_denies(guard_env, monkeypatch):
    guard_env(**_ALL_ON)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval, "_should_fall_through_to_cli_approval", lambda **_k: False)
    assert check_all_command_guards("sudo reboot", "local")["approved"] is False


@pytest.mark.parametrize("no_approval, outcome", [(False, "prompt"), (True, "run")])
@pytest.mark.parametrize("mode", ["off", "smart", "manual"])
def test_yolo_follows_the_without_approval_level(guard_env, interactive, monkeypatch, mode, no_approval, outcome):
    guard_env(allow_reboot=True, reboot_without_approval=no_approval, mode=mode)
    _guardian(monkeypatch, "approve")
    enable_session_yolo("power_opt_in_test")
    prompt = _Prompt("deny")
    _assert_outcome(check_all_command_guards("sudo reboot", "local", approval_callback=prompt), prompt, outcome)


def test_yolo_cannot_lift_no_approval_without_allow(guard_env, interactive):
    guard_env(reboot_without_approval=True, mode="off")
    enable_session_yolo("power_opt_in_test")
    prompt = _Prompt("once")
    _assert_outcome(check_all_command_guards("sudo reboot", "local", approval_callback=prompt), prompt, "hardline")


# ---- hermes approvals test wording -------------------------------------------------------------

def _approvals_test(monkeypatch, cfg, cmd):
    from hermes_cli import approvals_test
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
    monkeypatch.setattr(approval_context, "_get_power_approval_config", lambda: cfg)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: set())
    monkeypatch.setattr(approval, "_permanent_set", lambda: set())
    return approvals_test.evaluate_command(cmd)


@pytest.mark.parametrize("category, cmd", [("reboot", "sync; sudo reboot"), ("shutdown", "poweroff")])
def test_approvals_test_reports_no_approval_level(monkeypatch, category, cmd):
    v = _approvals_test(monkeypatch, {f"allow_{category}": True, f"{category}_without_approval": True}, cmd)
    assert v["verdict"] == "allow" and v["exit_code"] == 0
    assert f"runs without approval (approvals.{category}_without_approval)" in v["detail"]
    assert "denied in cron" in v["detail"]


@pytest.mark.parametrize("category, cmd", [("reboot", "sync; sudo reboot"), ("shutdown", "poweroff")])
def test_approvals_test_reports_human_level(monkeypatch, category, cmd):
    v = _approvals_test(monkeypatch, {f"allow_{category}": True}, cmd)
    assert v["verdict"] == "ask-approval" and "requires human approval every time" in v["detail"]


def test_approvals_test_no_approval_alone_stays_hardline(monkeypatch):
    v = _approvals_test(monkeypatch, {"reboot_without_approval": True}, "sudo reboot")
    assert v["verdict"] == "hardline-deny" and "approvals.allow_reboot" in v["detail"]


@pytest.mark.parametrize("cfg, verdict", [
    ({"allow_reboot": True, "allow_shutdown": True, "reboot_without_approval": True}, "ask-approval"),
    (_ALL_ON, "allow"),
])
def test_approvals_test_mixed_category(monkeypatch, cfg, verdict):
    v = _approvals_test(monkeypatch, cfg, "reboot -p")
    assert v["verdict"] == verdict
    if verdict == "allow":
        assert "approvals.reboot_without_approval + approvals.shutdown_without_approval" in v["detail"]
