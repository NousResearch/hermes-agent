"""Command detection around comments in ``$(...)``, the ``&>`` dash/sh reading and xargs option bundles."""

import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command

_COMMENT_SUBSTS = [
    "echo \"$(echo a # it's\n)\"; {}",
    "echo \"$(echo a # it's\n)\"\n{}",
    "echo \"$(echo a # \"x\n)\"; {}",
    "echo \"$(echo a # it's\n)\" && {}",
]


@pytest.mark.parametrize("template", _COMMENT_SUBSTS)
def test_comment_in_quoted_substitution_does_not_hide_hardline(template):
    assert detect_hardline_command(template.format("reboot"))[0]


@pytest.mark.parametrize("template", _COMMENT_SUBSTS)
def test_comment_in_quoted_substitution_does_not_hide_dangerous(template):
    # `python3 -c` is only recognised at a command position, so this fails if the start is hidden.
    assert detect_dangerous_command(template.format("python3 -c 'import os'"))[0]


def test_comment_in_quoted_substitution_benign_stays_clean():
    assert not detect_hardline_command("echo \"$(echo a # it's\n)\"; echo ok")[0]


def test_ampersand_redirect_keeps_background_reading():
    payload = "reboot"
    # dash/sh reads ``a &>f b`` as ``a &`` then ``>f b``: ``b`` is a command there.
    assert detect_hardline_command(f"true &>f {payload}")[0]
    assert detect_hardline_command(f"true &> f {payload}")[0]
    assert detect_hardline_command(f"true &>>f {payload}")[0]


def test_ampersand_redirect_dangerous_second_reading():
    assert detect_dangerous_command("true &>f python3 -c 'import os'")[0]
    assert detect_dangerous_command("true &> f python3 -c 'import os'")[0]


@pytest.mark.parametrize("options", ["-tn 1", "-rP 4", "-ra f", "-0E eof", "-rI {}", "-0rd '\\n'"])
@pytest.mark.parametrize("payload", ["sh -c 'reboot'", "bash -c 'reboot'"])
def test_xargs_bundled_short_option_operand_is_skipped(options, payload):
    assert detect_hardline_command(f"echo x | xargs {options} {payload}")[0]


def test_xargs_bundled_attached_operand_not_over_skipped():
    assert detect_hardline_command("echo x | xargs -tn1 sh -c 'reboot'")[0]
    assert not detect_dangerous_command("echo x | xargs -tn 1 echo hi")[0]


@pytest.mark.parametrize("command", [
    "make &>/dev/null",
    "ls &> out.txt; echo ok",
    "docker ps &>/dev/null echo ok",
])
def test_ampersand_redirect_benign_stays_clean(command):
    assert not detect_dangerous_command(command)[0]
    assert not detect_hardline_command(command)[0]


@pytest.mark.xfail(strict=True, reason=(
    "Known gap, identical on base: an awk program read from a file (-f) or stdin cannot be inspected "
    "without reading files, and `busybox awk` dispatches through an applet word the wrapper table does "
    "not unwrap. Fixing either adds a new prompt class, not a bug fix inside this PR's design."))
@pytest.mark.parametrize("command", [
    "awk -f p.awk",
    "awk -f- <<<'BEGIN{system(\"x\")}'",
    "busybox awk 'BEGIN{system(\"x\")}'",
])
def test_awk_uninspectable_program_gap(command):
    assert detect_dangerous_command(command)[0]


@pytest.mark.parametrize("command", [
    "echo \"$(echo `true #x`)\"; {}",
    "echo \"$(echo ${{x:-a #b}})\"; {}",
    "echo ${{x:-a #b}}; {}",
    "echo \"$( (true); {} )\"",
    "echo \"$(echo a # it's\n  echo `true #x`)\"; {}",
])
def test_comment_marker_inside_backtick_or_brace_is_not_a_comment(command):
    assert detect_hardline_command(command.format("reboot"))[0]


@pytest.mark.parametrize("options", ["-in", "-is", "-rd=", "-tI=x", "--delim ,", "--max-ar 1"])
def test_xargs_attached_and_abbreviated_options(options):
    assert detect_hardline_command(f"echo x | xargs {options} sh -c 'reboot'")[0]


@pytest.mark.parametrize("command", [
    "echo \"$(echo ${{x:-'}}'}} )\"; {}",
    "echo \"$(echo ${{x:-'}}'}} )\"\n{}\n#'",
    "echo ${{x:-\"}}\"}}; {}",
    "echo ${{x:-${{y:-}}}}; {}",
    "echo `true #x` 'a\nb'\n{}",
    "echo ${{x:-a #b}} 'a\nb'\n{}",
    "echo \"$(echo `true #x` 'a\nb'\n{})\"",
])
def test_brace_closer_and_newline_mask_respect_quotes_and_units(command):
    assert detect_hardline_command(command.format("reboot"))[0]


@pytest.mark.parametrize("command", [
    "echo `true #x`; bash -c 'reboot'",
    "echo `true #x`; python3 -c 'import os'",
    "echo ${x:-a #b}; python3 -c 'import os'",
])
def test_exec_flag_scan_does_not_read_comment_inside_backtick_or_brace(command):
    assert detect_dangerous_command(command)[0]


def test_ampersand_redirect_target_is_bounded_by_the_enclosing_span():
    # The backtick closes the target; the word after it is an argument of echo, not a command start.
    assert not detect_dangerous_command("echo `a &>f` python3 -c 'import os'")[0]


@pytest.mark.parametrize("command", [
    "echo ${x:-'}'}; echo ok",
    "echo `a &>f` x",
    "echo \"$(echo 'a\nb')\"\necho hi",
])
def test_quote_and_unit_aware_scans_keep_benign_commands_clean(command):
    assert not detect_hardline_command(command)[0]
    assert not detect_dangerous_command(command)[0]


def test_ampersand_redirect_quoted_substitution_target():
    assert detect_hardline_command('true &>"$(echo "a b")" sh -c reboot')[0]
    assert detect_dangerous_command('true &>"$(echo "a b")" python3 -c "import os"')[0]
