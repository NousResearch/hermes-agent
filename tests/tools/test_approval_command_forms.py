"""Round-9 review findings: commands hidden in ``${..}`` words / bash 5.3 command braces, and awk
programs whose string parity or option bundles the earlier check misread."""

import signal
import time

import pytest

from tools import approval as approval_mod
from tools import approval_context

from tools.approval_detection import detect_dangerous_command, detect_hardline_command

_SHELL_COMMAND_FORMS = [
    "echo ${x:-<(reboot)}",
    "echo ${x:-<(reboot)} tail",
    "echo ${x:-<(true; reboot)}",
    "echo ${x:-<(true && reboot)}",
    "echo ${x:-<(true | reboot)}",
    "echo ${x:-<((reboot))}",
    "echo ${x:-<(true\nreboot)}",
    "echo ${x:->(true; reboot)}",
    "echo ${x:-<(sudo reboot)}",
    "echo ${ true; reboot; }",
    'echo "${ true; reboot; }"',
    "${ reboot; }",
    "echo ${| reboot; }",
    "x=${ reboot; }",
    'echo "${| reboot; }"',
]


@pytest.mark.parametrize("command", _SHELL_COMMAND_FORMS)
def test_hardline_command_inside_process_substitution_or_command_brace(command):
    assert detect_hardline_command(command)[0]


@pytest.mark.parametrize("command", [
    "echo ${x:-<(bash -c 'reboot')}",
    "echo ${x:->(bash -c id)}",
    "echo ${ bash -c id; }",
    'echo "${ python3 -c pass; }"',
])
def test_dangerous_command_inside_process_substitution_or_command_brace(command):
    assert detect_dangerous_command(command)[0]


@pytest.mark.parametrize("command", [
    "echo ${x:-(a;b)}",
    "echo ${x:-<(ls)}",
    "echo ${x} reboot",
    "echo ${x:-a b} ok",
    'echo "${ ls; }"',
    "echo ${x:-reboot}",
])
def test_benign_expansions_stay_clean(command):
    assert not detect_hardline_command(command)[0]
    assert not detect_dangerous_command(command)[0]


_AWK_SHELL = [
    'awk \'BEGIN{x="a\\"b"; system("id")}\'',
    'awk \'{print "\\"" | "sh"}\'',
    'awk \'BEGIN{x="\\\\"; system("id")}\'',
    "gawk -oF 'BEGIN{system(\"id\")}' f",
    "gawk -dv 'BEGIN{system(\"id\")}' f",
    "gawk -pF 'BEGIN{system(\"id\")}' f",
    "gawk -DF 'BEGIN{system(\"id\")}' f",
    "gawk -bpF 'BEGIN{system(\"id\")}' f",
    "awk 'BEGIN{print 1 &&\n 1 | \"sh\"}'",
    "awk 'BEGIN{print 1 ||\n 1 | \"sh\"}'",
]


@pytest.mark.parametrize("command", _AWK_SHELL)
def test_awk_programs_that_run_a_shell_are_flagged(command):
    assert detect_dangerous_command(command)[0]


@pytest.mark.parametrize("command", [
    'awk \'BEGIN{x="a\\"b"; print x}\' f',
    "gawk -oF '{print $1}' f",
    "awk 'BEGIN{print 1 &&\n 1}'",
])
def test_awk_benign_forms_stay_clean(command):
    assert not detect_dangerous_command(command)[0]


def test_nested_shell_payload_under_dash_quote_reading_is_inspected():
    # dash reads the inner quote of "${x:-'}" as a literal, so the nested `sh -c reboot` runs.
    command = "sh -c 'echo \"${x:-'\"'\"'}\"; sh -c " + "re" + "boot; echo \"'\"'\"'}\"'"
    assert detect_hardline_command(command)[0]


def test_many_ansi_quoted_words_do_not_hit_the_reading_limit():
    command = "echo " + " ".join("$'a\\'b'" for _ in range(12))
    assert not detect_hardline_command(command)[0]


def test_deny_rule_still_matches_a_command_too_nested_to_parse(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "deny": ["curl *"]})
    assert approval_mod._match_user_deny_rule("curl x; echo " + '"$(' * 60) == "curl *"


@pytest.mark.parametrize("command", [
    'echo "${ { true; }; @@; }"',
    'echo "${ echo }; @@; }"',
    'echo "${ { a; { b; }; }; @@; }"',
])
def test_command_brace_closes_at_the_list_terminator_not_the_first_brace(command):
    assert detect_hardline_command(command.replace("@@", "re" + "boot"))[0]


def test_command_brace_with_nested_group_stays_clean():
    assert not detect_hardline_command('echo "${ { true; }; ls; }" ok')[0]


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="watchdog needs SIGALRM (POSIX)")
def test_comment_hidden_openers_are_not_rescanned_per_opener():
    command = "# $( #\n" * 6000
    def expired(signum, frame):
        raise AssertionError("opener rescans did not finish within 20s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(20)
    try:
        started = time.monotonic()
        detect_hardline_command(command)
        detect_dangerous_command(command)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert time.monotonic() - started < 10
