"""A command the scanners run out of budget on asks for approval; it is not a hardline refusal.

Hardline refusals stay for the real limits (size, segment count, real ``$(`` nesting) and for any
hardline rule, whether the full scan reached it or only the linear fallback did (see
test_approval_scan_incomplete_human_only.py). An approval for an unread command covers that one
run only: a session or always grant would wave every later unread command through.
"""
from unittest.mock import MagicMock

import pytest

from tools import approval
from tools import approval_context
from tools.approval import check_all_command_guards
from tools.approval_detection import (
    _PARSER_LIMIT_DESCRIPTION, _SCAN_INCOMPLETE_DESCRIPTION, detect_dangerous_command, detect_hardline_command)

SESSION = "scan-budget-session"
# Every `\"` inside `$(…)` leaves a `$(` open in the escape-stripped text, so a few hundred such
# lines run the scanners out of budget although bash nests none of them.
BEAR_800 = [rf'v=$(grep -o "\"f{i}\":\"[^\"]*\"" out.json)' if i % 2 == 0 else f'echo "step {i}"'
            for i in range(800)]
TEKNIUM_175 = "\n".join(['if [ -f "$f" ]; then v=$(grep -o "name=\\"[^\\"]*\\"" "$f" | head -1); '
                         'echo "$(date) $v" >> log.txt; fi'] * 175)
GREP_500 = "\n".join(['v=$(grep -o "[^\\"]*" f)'] * 500)


@pytest.fixture
def manual_cli(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION)
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    token = approval_context.set_current_session_key(SESSION)
    saved = set(approval._permanent_approved)
    approval._permanent_approved.clear()
    approval.clear_session(SESSION)
    monkeypatch.setattr(approval, "save_permanent_allowlist", lambda *_a, **_k: None)
    yield
    approval.clear_session(SESSION)
    approval._permanent_approved.clear()
    approval._permanent_approved.update(saved)
    approval_context.reset_current_session_key(token)


def test_a_long_script_that_runs_out_of_scan_budget_still_prompts_for_its_dangerous_line(manual_cli):
    callback = MagicMock(return_value="deny")
    result = check_all_command_guards("\n".join([*BEAR_800, "rm -rf build/"]), "local", approval_callback=callback)
    assert not result.get("hardline"), result["message"][:120]
    assert result["approved"] is False
    callback.assert_called_once()
    _command, description = callback.call_args.args
    assert "recursive delete" in description and _SCAN_INCOMPLETE_DESCRIPTION in description
    # Once or deny only: the prompt offers no scope that would remember an unread command.
    assert callback.call_args.kwargs.get("allow_session") is False
    assert callback.call_args.kwargs.get("allow_permanent") is False


def test_an_unread_command_is_approved_one_run_at_a_time(manual_cli):
    command = "\n".join([*BEAR_800, "rm -rf build/"])
    # A grant for the dangerous line found in the read part does not cover the unread rest.
    approval.approve_session(SESSION, "recursive delete")
    callback = MagicMock(return_value="session")
    assert check_all_command_guards(command, "local", approval_callback=callback)["approved"] is True
    callback.assert_called_once()
    # Even a "session" answer persists nothing: the same command asks again.
    assert not approval.is_approved(SESSION, _SCAN_INCOMPLETE_DESCRIPTION)
    again = MagicMock(return_value="deny")
    assert check_all_command_guards(command, "local", approval_callback=again)["approved"] is False
    again.assert_called_once()


@pytest.mark.parametrize("command", [TEKNIUM_175, GREP_500], ids=["teknium-175-lines", "grep-500"])
def test_ordinary_scripts_over_the_budget_are_not_hard_blocked(command, manual_cli):
    assert detect_hardline_command(command) == (False, None)
    # Callers without a prompt (gateway session shell, TUI shell.exec) refuse it as dangerous.
    assert detect_dangerous_command(command)[1] == _SCAN_INCOMPLETE_DESCRIPTION
    callback = MagicMock(return_value="once")
    result = check_all_command_guards(command, "local", approval_callback=callback)
    assert result["approved"] is True and not result.get("hardline")
    callback.assert_called_once()


def test_real_nesting_still_hard_refuses(manual_cli):
    command = "echo " + "$(" * 40 + "x" + ")" * 40
    assert detect_hardline_command(command) == (True, _PARSER_LIMIT_DESCRIPTION)
    callback = MagicMock(return_value="once")
    result = check_all_command_guards(command, "local", approval_callback=callback)
    assert result["hardline"] is True and result["approved"] is False
    callback.assert_not_called()


@pytest.mark.parametrize("tail", ["rm -rf /", "bash -c 'rm -rf /'", "reboot"])
def test_a_hardline_verb_found_before_the_budget_ran_out_still_hard_refuses(tail, manual_cli):
    command = "\n".join([*BEAR_800, tail])
    assert detect_hardline_command(command)[0] is True
    callback = MagicMock(return_value="once")
    result = check_all_command_guards(command, "local", approval_callback=callback)
    assert result["hardline"] is True and result["approved"] is False
    callback.assert_not_called()


def test_approvals_suggest_never_proposes_an_unread_command():
    from hermes_cli.approvals_suggest import is_unsafe_class
    assert is_unsafe_class(_SCAN_INCOMPLETE_DESCRIPTION)
