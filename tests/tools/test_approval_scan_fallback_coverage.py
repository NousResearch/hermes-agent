"""What the budget-trip fallback must still refuse, and the scan-incomplete key no grant can cover.

When benign padding runs the hardline scan out of budget, ``_hardline_fallback`` is the only
thing between a later hardline verb and a human-only once/deny prompt. Spellings glued to a quote,
to ``${...}`` or to ``$'...'`` must refuse there as they do in the full scan. Command strings are
only passed to the detectors, never run.
"""
from unittest.mock import MagicMock

import pytest

from tests.tools import test_hardline_blocklist as blocklist
from tools import approval, approval_smart
from tools.approval import check_all_command_guards, check_dangerous_command
from tools.approval_detection import (
    _MALFORMED_EXEC_DESCRIPTION, _PARSER_LIMIT_DESCRIPTION, _SCAN_INCOMPLETE_DESCRIPTION,
    _hardline_fallback, detect_hardline_command, hardline_scan_incomplete)
from tests.tools.test_approval_scan_incomplete_human_only import GREP, GREP_500, SESSION
from tests.tools.test_approval_scan_incomplete_human_only import gate as _gate_fixture

gate = _gate_fixture  # the shared manual-mode, isolated-grant-store fixture

GLUED_SPELLINGS = [
    pytest.param("true; rm -rf ${HOME}", id="semicolon-rm-home-brace"),
    pytest.param("(rm -rf ${HOME})", id="subshell-rm-home-brace"),
    pytest.param('true; dd if=/dev/zero "of=/dev/sda"', id="dd-quoted-of"),
    pytest.param("r$'m' -rf /", id="ansi-c-rm"),
    pytest.param("$'\\x72'eboot", id="ansi-c-reboot"),
]
# The unpadded full scan does not refuse these on main either (a `$'..'`-spliced verb after a
# separator); padded, the fallback's over-approximation must refuse them, as the remote head did.
PADDED_ONLY_SPELLINGS = [
    pytest.param("true; $'\\x72'eboot", id="semicolon-ansi-c-reboot"),
    pytest.param("{ $'\\x72'eboot; }", id="brace-ansi-c-reboot"),
]


def _padded(lines: int, tail: str) -> str:
    return "\n".join([GREP] * lines + [tail])


@pytest.mark.parametrize("lines", [400, 800, 1000])
@pytest.mark.parametrize("tail", GLUED_SPELLINGS + PADDED_ONLY_SPELLINGS)
def test_glued_hardline_spellings_refuse_after_padding(tail, lines):
    verdict = detect_hardline_command(_padded(lines, tail))
    assert verdict[0] is True, (lines, tail, verdict)


def _corpus() -> list[str]:
    """The hardline corpus of test_hardline_blocklist (imported lists, no source reading) plus the
    ANSI-C spellings, keeping what the full scan refuses on a real rule."""
    commands = [*blocklist._HARDLINE_BLOCK, *blocklist._COMMAND_POSITION_ROOT_WIPES,
                *blocklist._TRUE_POSITIVES_93392, *blocklist._QUOTED_NEWLINE_THREATS_BLOCK,
                *(command for command, _ in blocklist._HARDLINE_LINE_CONTINUATION),
                "r$'m' -rf /", "$'\\x72'eboot", "$'\\x72\\x6d' -rf /", "rm -rf $'\\x2f'",
                "true; rm -rf ${HOME}", 'dd if=/dev/zero "of=/dev/sda"']
    refused = []
    for command in dict.fromkeys(commands):
        hardline, description = detect_hardline_command(command)
        if hardline and description not in (_PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION):
            refused.append(command)
    return refused


@pytest.mark.parametrize("tail", GLUED_SPELLINGS)
def test_the_glued_spellings_are_refused_by_the_full_scan_unpadded(tail):
    assert detect_hardline_command(tail)[0] is True


def test_the_fallback_refuses_every_corpus_command_the_full_scan_refuses():
    corpus = _corpus()
    assert len(corpus) > 140
    missed = [command for command in corpus if not _hardline_fallback(GREP + "\n" + command)]
    assert missed == []


@pytest.mark.parametrize("store", ["permanent", "session"])
@pytest.mark.parametrize("guard", [check_all_command_guards, check_dangerous_command])
def test_a_stored_scan_incomplete_key_never_pre_approves(store, guard, gate):
    assert hardline_scan_incomplete(GREP_500)
    if store == "permanent":
        approval.load_permanent({_SCAN_INCOMPLETE_DESCRIPTION})  # a hand-edited command_allowlist
    else:
        approval.approve_session(SESSION, _SCAN_INCOMPLETE_DESCRIPTION)
    assert not approval.is_approved(SESSION, _SCAN_INCOMPLETE_DESCRIPTION)
    callback = MagicMock(return_value="deny")
    result = guard(GREP_500, "local", approval_callback=callback)
    assert result["approved"] is False
    callback.assert_called_once()  # a person was asked
    assert callback.call_args.kwargs.get("allow_session") is False


def test_a_fallback_refusal_says_where_it_came_from_and_how_to_recover(gate):
    # The fallback reads quoted text as code, so a benign `echo "shutdown complete"` behind padding
    # refuses; the block must say so and point at splitting, not claim the operation was seen.
    result = check_all_command_guards(GREP_500 + '\necho "shutdown complete"', "local")
    assert result["hardline"] is True
    assert "quoted text can match" in result["message"]
    assert "Split the work into smaller commands" in result["message"]


def test_smart_approval_never_decides_a_scan_incomplete_finding(gate, monkeypatch):
    # Detection's own budget can run out on a command the hardline scan finished: the finding,
    # not only the hardline scan, keeps the guardian LLM out.
    gate("smart")
    guardian = MagicMock(return_value="approve")
    monkeypatch.setattr(approval_smart, "_smart_approve", guardian)
    monkeypatch.setattr(approval, "detect_dangerous_command", lambda command: (
        True, _SCAN_INCOMPLETE_DESCRIPTION, _SCAN_INCOMPLETE_DESCRIPTION))
    callback = MagicMock(return_value="deny")
    assert hardline_scan_incomplete("echo ok") is False
    result = check_all_command_guards("echo ok", "local", approval_callback=callback)
    assert result["approved"] is False and not result.get("smart_approved")
    guardian.assert_not_called()
    callback.assert_called_once()
    assert callback.call_args.kwargs.get("allow_session") is False
