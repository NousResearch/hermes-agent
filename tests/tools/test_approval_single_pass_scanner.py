"""The substitution scanner is one iterative, linear pass: nested quotes in ``${...}`` / ``$(...)``
keep their own quote state, and hostile nesting fails closed instead of hanging or recursing."""

import signal

import pytest

from tools import approval as approval_mod
from tools import approval_context
from tools.approval_detection_awk import awk_program_runs_shell
from tools.approval_detection import (
    _MAX_SUBSTITUTION_NESTING, _command_parser_limit_exceeded, _scan_substitution_end,
    detect_dangerous_command, detect_hardline_command,
)

_NESTED_QUOTE_FORMS = [
    '${x:-"${y}"}; @@',
    'echo ${x:-"${y}"}; @@',
    'echo "${x:-"${y}"}"; @@',
    'echo "${x:-"a b"}"; @@',
    "echo ${x:-'}'}; @@",
    'echo "$(echo ${x:-"}"})"; @@',
    'echo $(echo "${x:-"$(echo a)"}"); @@',
    'echo "${x:-"it\'s"}"; @@',
]


def _fill(template, payload):
    return template.replace("@@", payload)


@pytest.mark.parametrize("template", _NESTED_QUOTE_FORMS)
def test_nested_quotes_in_expansions_do_not_hide_hardline(template):
    assert detect_hardline_command(_fill(template, "reboot"))[0]


@pytest.mark.parametrize("template", _NESTED_QUOTE_FORMS)
def test_nested_quotes_in_expansions_do_not_hide_dangerous(template):
    assert detect_dangerous_command(_fill(template, "python3 -c 'import os'"))[0]


@pytest.mark.parametrize("template", _NESTED_QUOTE_FORMS)
def test_nested_quotes_in_expansions_benign_stays_clean(template):
    assert not detect_hardline_command(_fill(template, "echo ok"))[0]


def _bounded(fn, *args, seconds=30):
    """Run *fn* under a SIGALRM watchdog (where the host has one) so a real hang fails the test
    instead of stalling it; elsewhere it just runs."""
    if not hasattr(signal, "SIGALRM"):
        return fn(*args)

    def expired(signum, frame):
        raise AssertionError(f"{fn.__name__} did not finish within {seconds}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("opener", ["$(", "${", "`$(", '"$(', "$((", "$(${"])
@pytest.mark.parametrize("repeat", [40, 2000])
def test_unterminated_openers_finish_quickly_and_fail_closed(opener, repeat):
    command = opener * repeat
    hardline = _bounded(detect_hardline_command, command)[0]
    dangerous = _bounded(detect_dangerous_command, command)[0]
    if repeat > _MAX_SUBSTITUTION_NESTING:
        assert hardline and dangerous  # nesting limit: refuse rather than descend


def test_many_sequential_unterminated_expansions_are_linear():
    command = "echo ${x " * 3000 + "; echo ok"
    _bounded(detect_hardline_command, command)
    _bounded(detect_dangerous_command, command)


def test_deep_balanced_nesting_fails_closed_without_recursion_error():
    depth = _MAX_SUBSTITUTION_NESTING + 40
    command = "echo " + "$(echo " * depth + "hi" + ")" * depth
    assert _command_parser_limit_exceeded(command)
    assert _bounded(detect_hardline_command, command)[0]


def test_moderate_nesting_still_inspected():
    assert detect_hardline_command("echo $(echo $(echo $(echo $(reboot))))")[0]
    assert detect_hardline_command('echo "$(echo "${x:-$(echo `reboot`)}")"')[0]
    assert not detect_hardline_command("echo $(echo $(echo $(echo ok)))")[0]


_BRACE_WORD_COMMENT_FORMS = [
    'echo "${x:-a #b $(echo x; @@)}"',
    'echo ${x:-a #b $(echo x; @@)}',
    'echo "${x:-a #b $(echo x\n@@)}"',
    'echo "${x:-#}"; @@',
]


@pytest.mark.parametrize("template", _BRACE_WORD_COMMENT_FORMS)
def test_hash_inside_expansion_word_is_not_a_comment(template):
    assert detect_hardline_command(_fill(template, "reboot"))[0]
    assert detect_dangerous_command(_fill(template, "python3 -c 'import os'"))[0]
    assert not detect_hardline_command(_fill(template, "echo ok"))[0]


@pytest.mark.parametrize("command", [
    'echo "${MSG:-done; reboot}"',
    'echo "${MSG:-done\nreboot}"',
    'echo "${MSG:-done && reboot}"',
])
def test_separators_inside_expansion_word_are_data(command):
    assert not detect_hardline_command(command)[0]


# dash / busybox ash / bash --posix read the quote inside "${...}" as a literal, bash as a quote.
_POSIX_QUOTE_FORMS = [
    'sh -c \'echo "${x:-\'"\'"\'}"; @@; echo "\'"\'"\'}"\'',
    'set -o posix\necho "${x:-\'}"; @@; echo "\'}"',
    'echo "${x:-\'}"; @@; echo "\'}"',
    'echo "${x:-${y:-\'}}"; @@; echo "\'}}"',
]


@pytest.mark.parametrize("template", _POSIX_QUOTE_FORMS)
def test_both_quote_readings_of_expansion_in_double_quotes_are_checked(template):
    assert detect_hardline_command(_fill(template, "reboot"))[0]
    if not template.startswith("sh -c"):  # `sh -c` is itself a dangerous finding
        assert detect_dangerous_command(_fill(template, "python3 -c 'import os'"))[0]


def test_benign_quote_in_expansion_stays_clean():
    assert not detect_hardline_command('echo "${x:-\'a b\'}"; echo ok')[0]
    assert not detect_dangerous_command('echo "${x:-\'a b\'}"; echo ok')[0]


# bash reads \' inside $'...' as an escaped quote; the plain-quote reading ends the string there.
_ANSI_QUOTE_FORMS = [
    "echo $'\\''; @@ # '",
    "echo $(echo $'\\''); @@; echo $(echo ')') # '",
    "echo $'a\\\\'; @@",
    "echo ${x:-$'\\''}; @@ # '",
]


@pytest.mark.parametrize("template", _ANSI_QUOTE_FORMS)
def test_ansi_c_quoting_cannot_hide_a_command(template):
    assert detect_hardline_command(_fill(template, "reboot"))[0]
    assert detect_dangerous_command(_fill(template, "python3 -c 'import os'"))[0]


def test_ansi_c_quoting_benign_stays_clean():
    for command in ("echo $'a\\nb' ok", "printf '%s' $'\\'' ; echo ok"):
        assert not detect_hardline_command(command)[0]


@pytest.mark.parametrize("wrapper", ["sh -c '{}'", "bash -c '{}'"])
def test_deep_nesting_inside_quoted_payload_fails_closed(wrapper):
    payload = "$(" * 2000
    command = wrapper.replace("{}", payload)
    assert _bounded(detect_hardline_command, command)[0]
    assert _bounded(detect_dangerous_command, command)[0]


# Both quote kinds in one command: reading one quote differently can turn the next into an opener.
_COMBINED_QUOTE_FORMS = [
    'echo "${x:-$\'}"; @@; echo "\'}"',
    'echo "${x:-\'}"; echo $\'}\\\'\'; @@ # \'',
    'sh -c \'echo "${x:-$\'"\'"\'}"; @@; echo "\'"\'"\'}"\'',
    'echo "${x:-\'}"; echo "${y:-$\'}"; @@; echo "\'}" $\'\\\'\' # \'',
]


@pytest.mark.parametrize("template", _COMBINED_QUOTE_FORMS)
def test_combined_quote_readings_cannot_hide_a_command(template):
    assert detect_hardline_command(_fill(template, "reboot"))[0]
    if not template.startswith("sh -c"):
        assert detect_dangerous_command(_fill(template, "python3 -c 'import os'"))[0]


@pytest.mark.parametrize("command", [
    "echo $'\\''; curl x # '",
    'echo "${x:-$\'}"; curl x; echo "\'}"',
    'echo "${x:-\'}"; curl x; echo "\'}"',
])
def test_user_deny_floor_checks_every_quote_reading(monkeypatch, command):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "deny": ["curl *"]})
    assert approval_mod._match_user_deny_rule(command) == "curl *"


@pytest.mark.parametrize("command", [
    "gawk --sou='BEGIN{system(\"id\")}'",
    "gawk --source 'BEGIN{system(\"id\")}'",
    "gawk --so='BEGIN{system(\"id\")}'",
    "awk 'BEGIN{print \"x\",\n \"y\" | \"sh\"}'",
])
def test_awk_abbreviated_long_options_and_continued_print_are_inspected(command):
    assert detect_dangerous_command(command)[0]


def test_awk_benign_stays_clean():
    assert not detect_dangerous_command("awk '{print $1, $2}' f")[0]
    assert not detect_dangerous_command("gawk --sou='{print $1}' f")[0]


_NEST = "$(echo " * 60 + "true; @@" + ")" * 60
_DEEP_OPAQUE_FORMS = [
    "echo `" + _NEST + "`",
    "echo \"${x:-'" + _NEST + "'}\"",
    "sh -c '" + _NEST.replace("'", "") + "'",
    "sh -c 'echo \"${x:-" + chr(39) + chr(34) + chr(39) + _NEST + chr(39) + chr(34) + chr(39) + "}\"'",
]


@pytest.mark.parametrize("template", _DEEP_OPAQUE_FORMS)
def test_nesting_hidden_by_backticks_or_a_quote_reading_fails_closed(template):
    command = _fill(template, "reboot")
    assert _bounded(detect_hardline_command, command)[0]
    assert _bounded(detect_dangerous_command, command)[0]


@pytest.mark.parametrize("command", [
    "awk 'BEGIN{print \"reboot\" \\\n| \"sh\"}'",
    "awk 'BEGIN{print \"a\",\n\n\"b\" | \"sh\"}'",
    "gawk -W source='BEGIN{system(\"id\")}'",
    "gawk -Wso='BEGIN{system(\"id\")}'",
])
def test_awk_continued_print_pipe_and_w_source_are_inspected(command):
    assert detect_dangerous_command(command)[0]


def test_awk_print_scan_is_linear():
    assert not _bounded(awk_program_runs_shell, ["print " * 60000], seconds=20)


_BACKTICK_QUOTE_FORMS = [
    "echo `echo $'a\\' '; @@`",
    "echo `echo $'a\\' '; @@ # '`",
    'echo `echo "${x:-\'}"; @@; echo "\'}"`',
    "echo `echo $(echo $'a\\' '); @@`",
    "echo \"`echo $'a\\' '; @@`\"",
    "echo `echo $'a\\' '; @@",
]


@pytest.mark.parametrize("template", _BACKTICK_QUOTE_FORMS)
def test_shell_dependent_quotes_inside_backticks_are_checked(template):
    assert detect_hardline_command(_fill(template, "reboot"))[0]
    assert detect_dangerous_command(_fill(template, "python3 -c 'import os'"))[0]


@pytest.mark.parametrize("command", [
    "gawk -bv x=1 'BEGIN{system(\"id\")}'",
    "gawk -bF : 'BEGIN{system(\"id\")}'",
    "gawk -bvx=1 'BEGIN{system(\"id\")}'",
    "gawk -be 'BEGIN{system(\"id\")}'",
    "gawk 'BEGIN{f=\"system\"; @f(\"id\")}'",
])
def test_awk_short_option_bundles_and_indirect_calls_are_inspected(command):
    assert detect_dangerous_command(command)[0]


def test_awk_bundles_benign_stays_clean():
    assert not detect_dangerous_command("gawk -bv x=1 '{print $1}' f")[0]
    assert not detect_dangerous_command("gawk -F: -v OFS=- '{print $1}' f")[0]


def test_span_lookup_matches_shell_nesting():
    text = 'a ${x:-"${y}"} b $(echo "a)" # it\'s )\n) c'
    assert _scan_substitution_end(text, 2) == text.index("} b") + 1
    assert _scan_substitution_end(text, text.index("$(")) == text.index(") c") + 1
