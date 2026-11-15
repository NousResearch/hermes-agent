"""A command the hardline scan could not read to the end is human-only.

Benign padding (a few hundred `v=$(grep -o "[^\\"]*" f)` lines) runs the hardline scan out of
budget before it reaches a later command position. The verb there must still hard-refuse (a
linear fallback over every possible command start), and what is left (the "too complex to scan"
finding) only a person may approve: not the smart-approval guardian, not YOLO / mode=off, not
cron / unattended approve-mode. Command strings are only passed to the detectors, never run.
"""
from unittest.mock import MagicMock

import pytest

from hermes_cli import approvals_test
from tools import approval
from tools import approval_context
from tools import approval_smart
from tools.approval import check_all_command_guards, check_dangerous_command
from tools.approval_detection import (
    _SCAN_INCOMPLETE_DESCRIPTION, detect_hardline_command, hardline_scan_incomplete)

SESSION = "scan-incomplete-human-only"
GREP = 'v=$(grep -o "[^\\"]*" f)'
BEAR_800 = "\n".join(rf'v=$(grep -o "\"f{i}\":\"[^\"]*\"" out.json)' if i % 2 == 0 else f'echo "step {i}"'
                     for i in range(800))
TEKNIUM_175 = "\n".join(['if [ -f "$f" ]; then v=$(grep -o "name=\\"[^\\"]*\\"" "$f" | head -1); '
                         'echo "$(date) $v" >> log.txt; fi'] * 175)
GREP_500 = "\n".join([GREP] * 500)
FORK_BOMB = ":()" + "{ :|:& };:"

# The reviewer's matrix: verbs at a `;` / `(` / `{` / `sudo` / `sh -c` / compound-keyword position
# that the budgeted scan never reached.
PADDED_HARDLINE = [
    pytest.param(800, "true; reboot", id="semicolon-reboot-800"),
    pytest.param(800, "(reboot)", id="subshell-reboot-800"),
    pytest.param(800, "{ reboot; }", id="brace-reboot-800"),
    pytest.param(800, "if true; then reboot; fi", id="then-reboot-800"),
    pytest.param(950, "sudo reboot", id="sudo-reboot-950"),
    pytest.param(950, FORK_BOMB, id="fork-bomb-950"),
    pytest.param(950, 'sh -c "mkfs.ext4 /dev/sda"', id="sh-c-mkfs-950"),
    pytest.param(950, "bash -c 'rm -rf /'", id="bash-c-rm-root-950"),
]


def _padded(lines: int, tail: str) -> str:
    return "\n".join([GREP] * lines + [tail])


@pytest.fixture
def gate(monkeypatch):
    """Manual mode, interactive CLI, isolated session/allowlist state; returns a mode setter."""
    mode = {"value": "manual"}
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: mode["value"])
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION)
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    token = approval_context.set_current_session_key(SESSION)
    saved = set(approval._permanent_approved)
    approval._permanent_approved.clear()
    approval.clear_session(SESSION)
    monkeypatch.setattr(approval, "save_permanent_allowlist", lambda *_a, **_k: None)
    yield lambda value: mode.__setitem__("value", value)
    approval.clear_session(SESSION)
    approval._permanent_approved.clear()
    approval._permanent_approved.update(saved)
    approval_context.reset_current_session_key(token)


@pytest.mark.parametrize("lines,tail", PADDED_HARDLINE)
def test_a_hardline_verb_after_budget_tripping_padding_still_hard_refuses(lines, tail, gate):
    command = _padded(lines, tail)
    assert hardline_scan_incomplete(command)
    assert detect_hardline_command(command)[0] is True
    callback = MagicMock(return_value="once")
    result = check_all_command_guards(command, "local", approval_callback=callback)
    assert result["hardline"] is True and result["approved"] is False
    callback.assert_not_called()


def _smart_approves(monkeypatch, gate):
    gate("smart")
    guardian = MagicMock(return_value="approve")
    monkeypatch.setattr(approval_smart, "_smart_approve", guardian)
    return guardian


@pytest.mark.parametrize("command", [GREP_500, BEAR_800 + "\nrm -rf build/"], ids=["grep-500", "bear-800-rm"])
def test_smart_approval_never_decides_an_unread_command(command, gate, monkeypatch):
    guardian = _smart_approves(monkeypatch, gate)
    callback = MagicMock(return_value="deny")
    result = check_all_command_guards(command, "local", approval_callback=callback)
    assert result["approved"] is False and not result.get("smart_approved")
    guardian.assert_not_called()
    callback.assert_called_once()  # the human prompt, once/deny only
    assert callback.call_args.kwargs.get("allow_session") is False


def test_smart_approval_still_decides_an_ordinary_dangerous_command(gate, monkeypatch):
    guardian = _smart_approves(monkeypatch, gate)
    result = check_all_command_guards("rm -rf build/", "local", approval_callback=MagicMock())
    assert result["approved"] is True and result.get("smart_approved") is True
    guardian.assert_called_once()


@pytest.mark.parametrize("guard", [check_all_command_guards, check_dangerous_command])
@pytest.mark.parametrize("bypass", ["yolo", "off"])
def test_yolo_and_off_refuse_an_unread_command(bypass, guard, gate, monkeypatch):
    if bypass == "yolo":
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    else:
        gate("off")
    result = guard(GREP_500, "local", approval_callback=MagicMock())
    assert result["approved"] is False
    assert "too long or complex" in result["message"] and "Split it" in result["message"]
    # An ordinary dangerous command keeps the bypass.
    assert guard("rm -rf build/", "local", approval_callback=MagicMock())["approved"] is True


@pytest.mark.parametrize("guard", [check_all_command_guards, check_dangerous_command])
def test_cron_approve_mode_refuses_an_unread_command(guard, gate, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
    result = guard(BEAR_800 + "\nrm -rf build/", "local")
    assert result["approved"] is False and "no human is present" in result["message"]
    assert guard("rm -rf build/", "local")["approved"] is True


def test_ask_mode_without_a_notifier_refuses_instead_of_parking(gate, monkeypatch):
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    result = check_all_command_guards(GREP_500, "local")
    assert result["approved"] is False and result.get("status") != "pending_approval"
    assert "no human is present" in result["message"]


def test_bear_800_line_script_still_asks_a_human_in_manual_mode(gate):
    callback = MagicMock(return_value="once")
    result = check_all_command_guards(BEAR_800 + "\nrm -rf build/", "local", approval_callback=callback)
    assert result["approved"] is True and not result.get("hardline")
    callback.assert_called_once()
    assert _SCAN_INCOMPLETE_DESCRIPTION in callback.call_args.args[1]


@pytest.mark.parametrize("command", [TEKNIUM_175, GREP_500], ids=["teknium-175", "grep-500"])
def test_ordinary_long_scripts_are_not_hard_blocked(command, gate):
    assert detect_hardline_command(command) == (False, None)
    callback = MagicMock(return_value="once")
    assert check_all_command_guards(command, "local", approval_callback=callback)["approved"] is True
    callback.assert_called_once()


def test_approvals_test_matches_the_runtime(gate, monkeypatch):
    assert approvals_test.evaluate_command(_padded(800, "true; reboot"))["verdict"] == "hardline-deny"
    assert approvals_test.evaluate_command(GREP_500)["verdict"] == "ask-approval"
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    verdict = approvals_test.evaluate_command(GREP_500)
    assert verdict["verdict"] == "scan-incomplete-deny" and verdict["exit_code"] == approvals_test.EXIT_DENY
    assert approvals_test.evaluate_command("rm -rf build/")["verdict"] == "allow"
